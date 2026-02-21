"""
Real-Robot Evaluation with Attention Visualization
Based on: unitree_lerobot/eval_robot/eval_g1.py
Enhanced with: Attention_Visualization/attention_visualizer.py

Usage:
    python Attention_Visualization/eval_attention_visualization.py \
      --policy.path=... --motion=True --send_real_robot=True ...
"""

import sys
import os
import time
import torch
import logging
import cv2
import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

import logging_mp
from Attention_Visualization.attention_visualizer import (
    register_attention_hooks,
    remove_hooks,
    compute_vision_heatmap,
    create_heatmap_overlay
)

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    
    # RERUN DISABLED per user request (User wants only Attention Viz)
    # if cfg.visualization:
    #     rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # --- ATTENTION HOOKS ---
    logger_mp.info("[AttentionVisualizer] Registering attention hooks...")
    patched_layers = register_attention_hooks(policy)
    
    # Video Writer Setup
    video_writer = None
    output_video_path = os.path.join(
        os.path.dirname(__file__), 
        "outputs", 
        "real_eval_attention.mp4"
    )
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    heatmap_last = None

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array", "wrist_img_array", "tv_img_shape", 
                "wrist_img_shape", "is_binocular", "has_wrist_cam"
            ]
        )

        # Get initial pose
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        
        if user_input.lower() == "s":
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0) 
            idx = 0
            latencies = []
            fps_list = []
            
            # Init Video Writer with correct dimensions (640x480)
            if cfg.visualization:
                video_writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    15,  # 15 FPS
                    (640, 480)
                )
                logger_mp.info(f"[AttentionVisualizer] Recording to {output_video_path} at 15 FPS")

            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                
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
                
                # 2. Get Action from Policy
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
                if cfg.visualization:
                    # Compute Heatmap
                    # aggregation="sum", head_aggregation="sum" per user preference (Sigmoid)
                    heatmap = compute_vision_heatmap(
                        image_size=(480, 640),
                        aggregation="sum",
                        head_aggregation="sum"
                    )
                    
                    # Persistence
                    if heatmap is not None:
                        heatmap_last = heatmap
                    elif heatmap_last is not None:
                        heatmap = heatmap_last
                        
                    # Overlay and Write
                    if heatmap is not None:
                        # Extract image from observation
                        img_tensor = observation["observation.images.cam_head"]
                        if img_tensor.ndim == 4: img_tensor = img_tensor[0]
                        
                        img_np = img_tensor.cpu().numpy()
                        
                        # Handle CHW vs HWC
                        if img_np.shape[0] == 3:  # CHW -> HWC
                            img_np = np.transpose(img_np, (1, 2, 0))
                        
                        # Ensure uint8 [0,255]
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)

                        # Fix RGB/BGR
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        overlay = create_heatmap_overlay(img_bgr, heatmap, alpha=0.3)
                        video_writer.write(overlay)
                        
                        # Real-time Display (if supported)
                        # Real-time Display (if supported)
                        try:
                            cv2.imshow("Attention Visualization", overlay)
                            cv2.waitKey(1)
                        except Exception as e:
                            # Log error once to avoid spam, or just print it
                            if idx % 30 == 0:
                                logger_mp.warning(f"OpenCV Display Error: {e}")

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

                # Original Rerun logging (Optional, keep if user wants logs too)
                # if cfg.visualization:
                #    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)

                latencies.append(inference_time)
                
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                fps_list.append(1.0 / (time.perf_counter() - loop_start_time))
                
                idx += 1
                if idx % 30 == 0:
                    avg_latency, avg_fps = np.mean(latencies[-30:]), np.mean(fps_list[-30:])
                    logger_mp.info(f"[Perf] Step: {idx} | Inference: {avg_latency:.1f}ms | FPS: {avg_fps:.1f}")

    except Exception as e:
        logger_mp.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logger_mp.info("[AttentionVisualizer] Cleaning up...")
        if video_writer:
            video_writer.release()
            logger_mp.info(f"Video saved to {output_video_path}")
            
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        remove_hooks(patched_layers)
        
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
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
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
