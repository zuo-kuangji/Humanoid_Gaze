
"""
Evaluation script for Head-Wise AG-CAM visualization on the G1 robot.
This script focuses on visualizing the 32 individual attention heads for specific layers.
"""
import logging
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import types
from copy import copy
from contextlib import nullcontext
from typing import Any
from dataclasses import dataclass, field

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

# Project imports
import sys
# Add the project root to path so we can import Attention_Analysis
sys.path.append(str(Path(__file__).parents[2]))

from Attention_Analysis.Attention_Module.agcam_visualizer import AGCAMVisualizer
from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    EvalRealConfig,
)
import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)

# --- Helper Functions (Copied to be self-contained) ---

def monkey_patch_eagle_feature_extraction(policy):
    """
    Monkey patch exact_feature to support interpolate_pos_encoding for 448x448 input
    without modifying the installed library code.
    """
    logger.info("Applying Monkey Patch to Eagle model...")
    if not hasattr(policy, "_groot_model"):
        logger.warning("Policy does not have _groot_model, skipping patch.")
        return
    
    eagle_model = policy._groot_model.backbone.eagle_model
    
    def custom_extract_feature(self, pixel_values):
        # Force interpolate_pos_encoding=True
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, 
                output_hidden_states=False, 
                return_dict=True,
                interpolate_pos_encoding=True 
            )
            if hasattr(vit_embeds, "last_hidden_state"):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, 
                output_hidden_states=True, 
                return_dict=True,
                interpolate_pos_encoding=True
            ).hidden_states[self.select_layer]

        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    eagle_model.extract_feature = types.MethodType(custom_extract_feature, eagle_model)
    logger.info("[Monkey Patch] Fixed Eagle extract_feature to support interpolate_pos_encoding")

def unwrap_no_grad(obj, method_name):
    """
    Dynamically remove @torch.no_grad() from a method at runtime by accessing __wrapped__.
    """
    if not hasattr(obj, method_name):
        return
    
    orig_method = getattr(obj, method_name)
    if hasattr(orig_method, "__wrapped__"):
        # We must re-bind the unwrapped function to the instance, 
        # otherwise 'self' won't be passed automatically.
        unwrapped_func = orig_method.__wrapped__
        setattr(obj, method_name, types.MethodType(unwrapped_func, obj))
        logger.info(f"Unwrapped and re-bound {method_name} for {obj.__class__.__name__}")

def predict_action_with_grad(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.set_grad_enabled(True),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            if not use_dataset:
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)
        
        # Enable gradients for image tensors
        for name in observation:
            if isinstance(observation[name], torch.Tensor) and "image" in name.lower():
                if observation[name].dtype in (torch.float32, torch.bfloat16):
                    observation[name].requires_grad_(True)
        if "eagle_pixel_values" in observation:
             observation["eagle_pixel_values"].requires_grad_(True)

        # [FIX] Pass observation as dictionary with keyword argument if needed, or just positional if wrappers support it.
        # GrootPolicy.select_action expects 'batch' (observation dict).
        action = policy.select_action(observation)
        
    return action, observation

def visualize_head_grid(img_bgr, head_maps, layer_idx, step_idx):
    """
    Visualizes a 4x8 grid of 32 attention heads for a single layer.
    Returns the image as a numpy array (H, W, 3) for video writing.
    """
    num_heads = head_maps.shape[0] # Should be 32
    rows = 4
    cols = 8
    
    # Let's say each head is 224x224 in the grid.
    # 8 cols * 224 = 1792 width
    # 4 rows * 224 = 896 height
    cell_h, cell_w = 224, 224
    grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    # [Numerical Analysis]
    # Let's find where and how big the global max is
    global_max = head_maps.max()
    global_min = head_maps.min()
    max_idx = np.unravel_index(np.argmax(head_maps), head_maps.shape)
    # logger.debug(f"[Head Analysis] Global Max Value: {global_max:.4f} at {max_idx} (H, L, Y, X)")

    for i in range(num_heads):
        row_idx = i // cols
        col_idx = i % cols
        
        heatmap = head_maps[i].copy()
        
        # [OPTIONAL: SINK SUPPRESSION]
        # Most sinks occur at the edges (y=0 or y=15, x=0 or x=15 in 16x16 grid)
        # We can zero out the top row if it's dominating everything
        # [ROBUST NORMALIZATION: 5% - 95% Percentile Clipping]
        # Suppression of top 5% (sinks/outliers) and bottom 5% (background noise)
        h_min_robust = np.percentile(heatmap, 5)
        h_max_robust = np.percentile(heatmap, 95)
        
        # Clip to these robust bounds
        heatmap_clipped = np.clip(heatmap, h_min_robust, h_max_robust)
        
        # Resize heatmap/image
        heatmap_resized = cv2.resize(heatmap_clipped, (cell_w, cell_h))
        img_resized = cv2.resize(img_bgr, (cell_w, cell_h))
        
        # Normalize within the robust range [5%, 95%]
        if h_max_robust - h_min_robust > 1e-7:
            heatmap_norm = (heatmap_resized - h_min_robust) / (h_max_robust - h_min_robust)
        else:
            heatmap_norm = np.zeros_like(heatmap_resized)
            
        # applyColorMap returns BGR
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        
        # Mix BGR image with BGR heatmap
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        
        # Add Text (Head ID)
        cv2.putText(overlay, f"H{i}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Place in grid
        y1 = row_idx * cell_h
        y2 = y1 + cell_h
        x1 = col_idx * cell_w
        x2 = x1 + cell_w
        grid_img[y1:y2, x1:x2] = overlay
        
    return grid_img

# --- Main Evaluation Loop ---

@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    init_logging()
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    logger.info("Initializing Dataset...")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, video_backend="pyav")
    
    logger.info("Initializing Policy...")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    
    # Apply Patches
    monkey_patch_eagle_feature_extraction(policy)
    unwrap_no_grad(policy, "select_action")
    unwrap_no_grad(policy, "predict_action_chunk")
    if hasattr(policy, "_groot_model") and hasattr(policy._groot_model, "action_head"):
        unwrap_no_grad(policy._groot_model.action_head, "get_action")
    
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}, "rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    
    # Initialize Visualizer
    real_policy = policy._groot_model if hasattr(policy, "_groot_model") else policy
    try:
        viz = AGCAMVisualizer(real_policy)
        logger.info(f"[AG-CAM] Initialization successful. Hooked {len(viz.layer_indices)} layers.")
    except Exception as e:
        logger.error(f"[AG-CAM] Initialization failed: {e}")
        return

    # Output directory
    # Output directory relative to this script
    output_dir = Path(__file__).parents[1] / "Outputs" / "head_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process Frames
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    to_idx = dataset.meta.episodes["dataset_to_index"][0]
    
    logger.info(f"Processing full episode from frame {from_idx} to {to_idx}...")
    
    # Video Writers for Layer 0 and Layer 14
    video_writers = {}
    fps = 30.0 # Standard fps
    
    # Dimensions will be determined on first frame
    
    for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
        step = dataset[step_idx]
        observation = extract_observation(step)
        
        # Inference
        action_tensor, _ = predict_action_with_grad(
            observation, policy, device,
            preprocessor, postprocessor, policy.config.use_amp,
            step["task"], use_dataset=True, robot_type=None,
        )
        
        # Generate Head Maps
        loss_target = action_tensor.abs().sum()
        all_layer_heads = viz.generate_heatmap(
            loss_target, 
            norm_method='none', # Raw for comparison
            return_heads=True
        ) # Shape: (B, Layers, Heads, H, W)
        
        if all_layer_heads is None:
            continue
            
        # Get RGB Image for Overlay
        img_key = next((k for k in observation if "images" in k), None)
        if img_key:
            img_np = observation[img_key].permute(1, 2, 0).float().numpy() # RGB [0, 1]
            img_rgb = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
             continue

        # 0 -> Layer 0, 1 -> Layer 2, ..., 7 -> Layer 14 (captured 8 layers total)
        # We sum all of them to see the total contribution to the Action.
        total_cumulative_head_maps = np.sum(all_layer_heads[0], axis=0) # Sum over Layers: (32, 16, 16)

        name = "All_Layers_Integrated"
        
        # Generate Grid Image (BGR)
        grid_bgr = visualize_head_grid(img_bgr, total_cumulative_head_maps, name, step_idx)
        
        # Initialize/Write Video
        if name not in video_writers:
            h, w = grid_bgr.shape[:2]
            # Switch to .avi and XVID for maximum reliability on Ubuntu systems
            video_path = str(output_dir.absolute() / f"Head_Analysis_{name}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
            else:
                video_writers[name] = writer
                logger.info(f"Started video recording (XVID/AVI): {video_path}")
        
        if name in video_writers:
            video_writers[name].write(grid_bgr)
        
        # Real-time display of the final integrated state
        display_scale = 1.5 
        display_img = cv2.resize(grid_bgr, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow(f"Integrated 32-Head Grid (Layers 0+2+...+14)", display_img)
        
        # Process keystrokes
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User requested stop.")
            break
    
    # Release Writers
    cv2.destroyAllWindows()
    for name, writer in video_writers.items():
        writer.release()
        logger.info(f"Finished video: {output_dir / f'Head_Analysis_{name}.mp4'}")
    
    logger.info("Done.")

if __name__ == "__main__":
    eval_main()
