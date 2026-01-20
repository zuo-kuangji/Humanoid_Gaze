"""
GR00T VLM Attention Visualization Evaluation Script

Based on eval_g1_dataset.py, with attention heatmap visualization added.
Outputs attention heatmaps overlaid on camera images via Rerun.

Usage:
    python Attention_Visualization/eval_attention.py \
      --policy.path=unitree_lerobot/lerobot/outputs/train/groot_handover/checkpoints/020000_1/pretrained_model \
      --repo_id=ZUO66/handover_drinks \
      --episodes=0 \
      --visualization=true
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unitree_lerobot', 'lerobot', 'src'))

import torch
import tqdm
import logging
import time
import numpy as np
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger

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


def eval_policy_with_attention(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
):
    """Evaluate policy while visualizing cross-attention heatmaps."""
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    logger_mp.info(f"Arguments: {cfg}")

    # Initialize Rerun logger
    rerun_logger = None
    if cfg.visualization:
        rerun_logger = RerunLogger()
        import rerun as rr
        rr.init("attention_visualization", spawn=True)

    # Initialize video writer for MP4 output
    import cv2
    video_output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(video_output_dir, exist_ok=True)
    video_path = os.path.join(video_output_dir, f"attention_ep{cfg.episodes}.mp4")
    
    # Video settings
    fps = 15  # Slower framerate for better stability analysis (was 30)
    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    logger_mp.info(f"Video will be saved to: {video_path}")

    # Heatmap Persistence State
    heatmap_last = None
    
    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Register attention hooks
    logger_mp.info("Registering attention hooks...")
    hooks = register_attention_hooks(policy)
    
    if not hooks:
        logger_mp.error("Failed to register attention hooks. Exiting.")
        video_writer.release()
        return

    # Get episode range
    from_idx = dataset.meta.episodes["dataset_from_index"][cfg.episodes]
    to_idx = dataset.meta.episodes["dataset_to_index"][cfg.episodes]
    step = dataset[from_idx]

    logger_mp.info(f"Processing episode {cfg.episodes}: steps {from_idx} to {to_idx}")

    try:
        for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc="Processing frames"):
            loop_start_time = time.perf_counter()

            # Clear previous attention maps
            clear_attention_maps()

            # Get observation from dataset
            step = dataset[step_idx]
            observation = extract_observation(step)

            # Forward pass (this triggers the hooks)
            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )

            # Compute vision attention heatmap
            # User wants to test "sum" aggregation for heads.
            heatmap = compute_vision_heatmap(
                image_size=(480, 640),
                aggregation="sum",      # Layers: Sum
                head_aggregation="max"  # Heads: Sum
            )

            # Persistence logic for Action Chunking
            # If inference is skipped (heatmap is None), reuse the last valid heatmap.
            if heatmap is not None:
                heatmap_last = heatmap
            
            # Use last valid heatmap
            current_heatmap = heatmap_last

            # Get camera image
            cam_key = "observation.images.cam_head"
            if cam_key in observation:
                image = observation[cam_key]
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                
                # Handle different image formats
                if image.ndim == 4:
                    image = image[0]  # Remove batch dim
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                # Create overlay with 30% transparency (heatmap less dominant)
                if current_heatmap is not None:
                    # Convert heatmap to colormap
                    heatmap_np = current_heatmap[0].numpy()
                    heatmap_uint8 = np.uint8(255 * heatmap_np)
                    
                    # Convert RGB image to BGR for OpenCV
                    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    
                    # Blend: 70% Original, 30% Heatmap (Clearer underlying video)
                    overlay_bgr = cv2.addWeighted(img_bgr, 0.7, heatmap_colored, 0.3, 0)
                else:
                    overlay_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Write single overlay frame
                video_writer.write(overlay_bgr)


                
                # Log to Rerun if visualization enabled
                if cfg.visualization and rerun_logger is not None:
                    import rerun as rr
                    rr.set_time_sequence("step", step_idx)
                    rr.log("camera/original", rr.Image(image))
                    rr.log("camera/attention_overlay", rr.Image(overlay))
                    
                    if heatmap is not None:
                        heatmap_vis = (heatmap[0].numpy() * 255).astype(np.uint8)
                        rr.log("attention/heatmap", rr.Image(heatmap_vis, color_model="L"))

            # Maintain frequency
            elapsed = time.perf_counter() - loop_start_time
            time.sleep(max(0, (1.0 / cfg.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user")
    finally:
        # Clean up
        video_writer.release()
        remove_hooks(hooks)
        logger_mp.info(f"Video saved to: {video_path}")
        logger_mp.info("Attention visualization complete")



@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy...")

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

    logging.info("Starting attention visualization...")

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy_with_attention(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
