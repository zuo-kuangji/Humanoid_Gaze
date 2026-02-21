"""
Real Robot Evaluation with SAM2 Object Tracking + GR00T Attention Visualization
Based on: eval_g1_sam.py + eval_attention_visualization.py

Features:
    1. SAM2 real-time object tracking (30% red overlay on tracked object)
    2. GR00T attention heatmap visualization (JET colormap overlay)
    3. Video recording of attention visualization

 IMPORTANT: Must run from project root /home/g1/unitree_groot1.5 directory!

Requirements:
    1. Start gaze server first (in glasses env):
       conda activate glasses
       cd /home/g1/unitree_groot1.5
       python unitree_lerobot/eval_robot/run_gaze_server.py --port 5556

    2. Then run this script (in groot1.5 env):
       conda activate groot1.5
       cd /home/g1/unitree_groot1.5
       python unitree_lerobot/eval_robot/eval_g1_sam_attention.py \
         --policy.path=unitree_lerobot/lerobot/outputs/train/groot_mask_handover_v2/checkpoints/last/pretrained_model \
         --repo_id=ZUO66/handover_mask_drinks \
         --use_sam=True \
         --gaze_port=5556 \
         --frequency=30 \
         --motion=True \
         --arm="G1_29" \
         --ee="inspire1" \
         --send_real_robot=True \
         --visualization=True \
         --use_attention_viz=True
"""

import sys
import os
import time
import torch
import logging
import zmq
import cv2
import numpy as np
from pprint import pformat
from dataclasses import asdict, dataclass, field
from torch import nn
from contextlib import nullcontext
from typing import Any

# Add Attention_Visualization to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Attention_Visualization'))

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

# Import attention visualizer
from attention_visualizer import (
    register_attention_hooks,
    remove_hooks,
    clear_attention_maps,
    compute_vision_heatmap,
    create_heatmap_overlay,
)

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


@dataclass
class EvalRealSAMAttentionConfig(EvalRealConfig):
    """Extended config with SAM2 tracking + Attention visualization options"""
    use_sam: bool = field(default=False, metadata={"help": "Enable SAM2 object tracking overlay"})
    gaze_port: int = field(default=5556, metadata={"help": "ZMQ port for gaze server"})
    sam_init_on_start: bool = field(default=True, metadata={"help": "Initialize SAM2 tracker before robot starts"})
    use_attention_viz: bool = field(default=False, metadata={"help": "Enable GR00T attention heatmap visualization"})


class GazeClient:
    """Client for communicating with SAM2 gaze server"""
    def __init__(self, port=5556, timeout=15000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.timeout = timeout
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.connect(f"tcp://localhost:{port}")
        self.initialized = False
        logger_mp.info(f"Connected to Gaze Server on port {port}")

    def ping(self):
        """Check if server is alive"""
        try:
            self.socket.send_pyobj({"cmd": "ping"})
            response = self.socket.recv_pyobj()
            return response.get("status") == "pong"
        except zmq.error.Again:
            logger_mp.warning("Gaze server ping timeout")
            return False

    def initialize_tracker(self, image_rgb):
        """Send first frame to server for user to select tracking object"""
        logger_mp.info("Sending image to gaze server for object selection...")
        logger_mp.info(">>> Please click on the object in the GUI window, then close it to confirm <<<")
        
        # Temporarily set longer timeout for user interaction (120 seconds)
        self.socket.setsockopt(zmq.RCVTIMEO, 120000)
        
        try:
            self.socket.send_pyobj({"cmd": "init", "image": image_rgb})
            response = self.socket.recv_pyobj()
            self.initialized = response.get("success", False)
            if self.initialized:
                logger_mp.info("✓ SAM2 tracker initialized successfully")
            else:
                logger_mp.warning("✗ SAM2 tracker initialization failed")
            return self.initialized
        except zmq.error.Again:
            logger_mp.error("Timeout waiting for user to select object (120s)")
            self.initialized = False
            return False
        finally:
            # Restore original timeout
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)

    def track(self, image_rgb):
        """Send frame and get back mask + overlayed image"""
        if not self.initialized:
            return None, image_rgb
        
        try:
            self.socket.send_pyobj({"cmd": "track", "image": image_rgb})
            response = self.socket.recv_pyobj()
            
            if response.get("status") != "ok":
                return None, image_rgb
            
            return response.get("mask"), response.get("overlayed_image", image_rgb)
        except zmq.error.Again:
            logger_mp.warning("[GazeClient] Tracker timeout, using previous state.")
            # Return None mask and original image to avoid crash
            return None, image_rgb

    def reset(self):
        """Reset tracker state"""
        self.socket.send_pyobj({"cmd": "reset"})
        response = self.socket.recv_pyobj()
        self.initialized = False
        return response.get("status") == "ok"

    def close(self):
        self.socket.close()
        self.context.term()


def eval_policy_with_sam_attention(
    cfg: EvalRealSAMAttentionConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    rerun_logger = None
    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Initialize SAM2 client if enabled
    gaze_client = None
    if cfg.use_sam:
        try:
            gaze_client = GazeClient(port=cfg.gaze_port)
            if not gaze_client.ping():
                logger_mp.error("Gaze server not responding! Start it with: python unitree_lerobot/eval_robot/run_gaze_server.py")
                gaze_client = None
        except Exception as e:
            logger_mp.error(f"Failed to connect to gaze server: {e}")
            gaze_client = None

    # Register attention hooks if enabled
    patched_layers = []
    if cfg.use_attention_viz:
        logger_mp.info("[AttentionVisualizer] Registering attention hooks...")
        patched_layers = register_attention_hooks(policy)
        if not patched_layers:
            logger_mp.warning("[AttentionVisualizer] Failed to register hooks, attention viz disabled")
            cfg.use_attention_viz = False

    # Video Writer Setup for attention visualization
    video_writer = None
    output_video_path = os.path.join(
        os.path.dirname(__file__), 
        "outputs", 
        "sam_attention_eval.mp4"
    )
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Heatmap persistence state
    heatmap_last = None

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Select episode to initialize pose/task.
        num_episodes = len(dataset.meta.episodes["dataset_from_index"])
        episode_index = int(cfg.episode_index)
        if episode_index < 0 or episode_index >= num_episodes:
            raise ValueError(f"episode_index out of range: {episode_index} (valid: 0..{num_episodes - 1})")
        from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
        step = dataset[from_idx]
        logger_mp.info(
            f"Using episode_index={episode_index}/{num_episodes - 1}, from_idx={from_idx}, task={step['task']}"
        )
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        # SAM2 Initialization before robot starts
        if gaze_client and cfg.sam_init_on_start:
            logger_mp.info("Waiting for first frame to initialize SAM2 tracker...")
            time.sleep(0.5)  # Wait for image server to be ready
            
            # Get a frame from the shared memory (BGR format from image server)
            init_frame_bgr = tv_img_array.copy()
            init_frame_rgb = cv2.cvtColor(init_frame_bgr, cv2.COLOR_BGR2RGB)
            
            gaze_client.initialize_tracker(init_frame_rgb)

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        
        if user_input.lower() == "s":
            # Initialize robot to starting pose
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)
            
            idx = 0
            latencies = []
            fps_list = []
            
            # Init Video Writer
            if cfg.use_attention_viz:
                video_writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    15,  # 15 FPS
                    (640, 480)
                )
                logger_mp.info(f"[AttentionVisualizer] Recording to {output_video_path} at 15 FPS")

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                
                # Clear attention maps before inference
                if cfg.use_attention_viz:
                    clear_attention_maps()
                
                # 1. Get Observations (with optional SAM2 overlay)
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                
                # Apply SAM2 tracking overlay to third-view camera (cam_head)
                overlayed_img = None
                if gaze_client and gaze_client.initialized:
                    tv_key = "observation.images.cam_head"
                    if tv_key in observation:
                        tv_img_tensor = observation[tv_key]
                        tv_img_np = tv_img_tensor.cpu().numpy().astype(np.uint8)
                        
                        # Track and get overlayed image (RGB)
                        mask, overlayed_img = gaze_client.track(tv_img_np)
                        
                        # Replace observation image with SAM2 overlayed version
                        if overlayed_img is not None:
                            observation[tv_key] = torch.from_numpy(overlayed_img)

                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                
                # 2. Get Action from Policy (this triggers attention hooks)
                inference_start = time.perf_counter()
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                inference_time = (time.perf_counter() - inference_start) * 1000
                action_np = action.cpu().numpy()
                
                # --- ATTENTION VISUALIZATION ---
                if cfg.use_attention_viz:
                    # Compute Heatmap
                    heatmap = compute_vision_heatmap(
                        image_size=(480, 640),
                        aggregation="sum",
                        head_aggregation="sum"
                    )
                    
                    # Persistence: reuse last heatmap if current is None
                    if heatmap is not None:
                        heatmap_last = heatmap
                    elif heatmap_last is not None:
                        heatmap = heatmap_last
                    
                    # Create attention overlay
                    if heatmap is not None:
                        # Get image for overlay (use SAM overlayed if available, else original)
                        img_tensor = observation["observation.images.cam_head"]
                        if img_tensor.ndim == 4:
                            img_tensor = img_tensor[0]
                        
                        img_np = img_tensor.cpu().numpy()
                        
                        # Handle CHW vs HWC
                        if img_np.shape[0] == 3:  # CHW -> HWC
                            img_np = np.transpose(img_np, (1, 2, 0))
                        
                        # Ensure uint8 [0,255]
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
                        # Create attention heatmap overlay
                        attention_overlay = create_heatmap_overlay(img_bgr, heatmap, alpha=0.3)
                        
                        # Write to video
                        if video_writer:
                            video_writer.write(attention_overlay)
                        
                        # Real-time Display
                        try:
                            cv2.imshow("Attention + SAM2 Visualization", attention_overlay)
                            cv2.waitKey(1)
                        except Exception as e:
                            if idx % 30 == 0:
                                logger_mp.warning(f"OpenCV Display Error: {e}")
                
                # Show SAM2 only if attention viz is disabled
                elif cfg.visualization and overlayed_img is not None:
                    display_img = cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("SAM2 Tracking (30% Red Overlay)", display_img)
                    cv2.waitKey(1)

                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization and rerun_logger:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)

                latencies.append(inference_time)
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                fps_list.append(1.0 / (time.perf_counter() - loop_start_time))
                
                idx += 1
                if idx % 30 == 0:
                    avg_latency, avg_fps = np.mean(latencies[-30:]), np.mean(fps_list[-30:])
                    logger_mp.info(f"[Perf] Step: {idx} | Inference: {avg_latency:.1f}ms | FPS: {avg_fps:.1f} (Max: {1000/avg_latency:.1f})")
                    
    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user")
    except Exception as e:
        logger_mp.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Cleanup
        logger_mp.info("Cleaning up...")
        cv2.destroyAllWindows()
        
        if video_writer:
            video_writer.release()
            logger_mp.info(f"Video saved to {output_video_path}")
        
        if patched_layers:
            remove_hooks(patched_layers)
            
        if gaze_client:
            gaze_client.close()
            
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealSAMAttentionConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")
    dataset = LeRobotDataset(repo_id=cfg.repo_id)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy_with_sam_attention(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
