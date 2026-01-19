#!/usr/bin/env python

"""
VLM (EAGLE LLM) Attention Visualization script.
Redesigned to be standalone and not modify shared project files.
Focuses on Eagle Visual Tokens within the Qwen-based VLM.
"""

import os
from pathlib import Path
import torch
import numpy as np
import cv2
import tqdm
import types
from copy import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

# Project imports
import sys
# Add the project root to path so we can import Attention_Analysis
sys.path.append(str(Path(__file__).parents[2]))

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.processor.rename_processor import rename_stats
from Attention_Analysis.Attention_Module.agcam_visualizer import VLM_AGCAMVisualizer
from unitree_lerobot.eval_robot.utils.utils import extract_observation
import logging_mp

# Setup logging
logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)

@dataclass
class VLMVisualConfig:
    """Local config class to avoid modifying shared utils.py"""
    repo_id: str = "ZUO66/handover_drinks"
    policy: PreTrainedConfig | None = None
    
    # Visualization specific parameters
    mode: str = "summary"  # "summary" or "head"
    layer_idx: int = -4
    display_scale: float = 1.0
    headless: bool = False
    
    # Common Eval fields for compatibility
    rename_map: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"
    use_amp: bool = False

    def __post_init__(self):
        # Reuse the project's logic to load policy config from path
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logger.warning("No pretrained path provided for policy.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

def monkey_patch_eagle_feature_extraction(policy):
    """
    Monkey patch exact_feature to support interpolate_pos_encoding for 448x448 input
    without modifying the installed library code.
    """
    logger.info("Applying Monkey Patch to Eagle model...")
    real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
    if not hasattr(real_model, "backbone"):
        logger.warning("Policy does not have backbone, skipping patch.")
        return
    
    eagle_model = real_model.backbone.eagle_model
    
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
    if not hasattr(obj, method_name): return
    orig_method = getattr(obj, method_name)
    if hasattr(orig_method, "__wrapped__"):
        unwrapped_func = orig_method.__wrapped__
        setattr(obj, method_name, types.MethodType(unwrapped_func, obj))
        logger.info(f"[VLM] Unwrapped {method_name}")

def predict_action_with_grad(observation, policy, device, preprocessor, postprocessor, use_amp):
    observation = copy(observation)
    with (
        torch.set_grad_enabled(True),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to batch
        for name in observation:
            if hasattr(observation[name], "unsqueeze"):
                # Normalize images if uint8
                if "images" in name and observation[name].dtype == torch.uint8:
                    observation[name] = observation[name].float() / 255.0
                    if observation[name].ndim == 3:
                        observation[name] = observation[name].permute(2, 0, 1)
                
                # Resize to 224x224 to match LLM prompt expectations (256 tokens)
                if "images" in name:
                    if observation[name].ndim == 3:
                        observation[name] = torch.nn.functional.interpolate(
                            observation[name].unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                        ).squeeze(0)
                
                observation[name] = observation[name].unsqueeze(0).to(device)
        
        observation["task"] = ""
        observation["robot_type"] = ""
        observation = preprocessor(observation)
        
        # Enable gradients for image and eagle tensors
        for name in observation:
            tensor = observation[name]
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                if any(key in name.lower() for key in ["image", "eagle", "pixel"]):
                    tensor.requires_grad_(True)
        
        action = policy.select_action(observation)
        return action

def apply_robust_norm(heatmap):
    # Standard min-max as requested for Vision Backbone
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max - h_min > 1e-7:
        return (heatmap - h_min) / (h_max - h_min)
    return np.zeros_like(heatmap)

def create_cell(img_bgr, heatmap_norm, label, h, w, small_text=False):
    heatmap_resized = cv2.resize(heatmap_norm, (w, h))
    img_resized = cv2.resize(img_bgr, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    # Adjust weights: increase img_resized (0.75) and decrease heatmap (0.25)
    overlay = cv2.addWeighted(img_resized, 0.75, heatmap_colored, 0.25, 0)
    # Add white border
    cv2.rectangle(overlay, (0, 0), (w-1, h-1), (255, 255, 255), 1)
    font_scale = 0.4 if small_text else 0.8
    thickness = 1 if small_text else 2
    cv2.putText(overlay, label, (5, int(20 * font_scale / 0.4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return overlay

def visualize_layer_grid(img_bgr, layer_maps):
    num_layers = layer_maps.shape[0]
    grid_cols = int(np.ceil(np.sqrt(num_layers * 1.5)))
    grid_rows = int(np.ceil(num_layers / grid_cols))
    cell_h, cell_w = 320, 320 # Increased size for better visibility
    grid_img = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
    for i in range(num_layers):
        heatmap_norm = apply_robust_norm(layer_maps[i])
        overlay = create_cell(img_bgr, heatmap_norm, f"L{i}", cell_h, cell_w, small_text=True)
        r, c = i // grid_cols, i % grid_cols
        grid_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = overlay
    return grid_img

def visualize_head_grid(img_bgr, head_maps):
    num_heads = head_maps.shape[0]
    grid_cols = 8
    grid_rows = int(np.ceil(num_heads / grid_cols))
    cell_h, cell_w = 320, 320 # Increased size
    grid_img = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
    for i in range(num_heads):
        heatmap_norm = apply_robust_norm(head_maps[i])
        overlay = create_cell(img_bgr, heatmap_norm, f"H{i}", cell_h, cell_w)
        r, c = i // grid_cols, i % grid_cols
        grid_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = overlay
    return grid_img

@parser.wrap()
def main(cfg: VLMVisualConfig):
    init_logging()
    device = get_safe_torch_device(cfg.device, log=True)
    
    dataset = LeRobotDataset(repo_id=cfg.repo_id, video_backend="pyav")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    
    # Apply Monkey Patch for 448x448 support
    monkey_patch_eagle_feature_extraction(policy)
    
    # Support both wrapped and unwrapped model
    real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
    
    # Unwrapping for Grad extraction
    unwrap_no_grad(policy, "select_action")
    unwrap_no_grad(policy, "predict_action_chunk")
    if hasattr(real_model, "action_head"):
        unwrap_no_grad(real_model.action_head, "get_action")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
    )
    
    # Initialize Multi-Layer VLM AGCAM
    # Vision start for Eagle is usually 20 tokens (instructions) + 5 tokens (misc) = ~25
    # Let's use the standard 20 for now.
    viz = VLM_AGCAMVisualizer(real_model, vision_token_start=20, vision_token_len=256)
    
    # Debug Layer Info
    logger.info(f"[VLM] Architecture Type: {type(viz.language_model)}")
    logger.info(f"[VLM] Full Model Config: {viz.language_model.config}")
    
    # Output directory relative to this script
    output_dir = Path(__file__).parents[1] / "Outputs" / "vlm_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    to_idx = dataset.meta.episodes["dataset_to_index"][0]
    
    video_writer = None
    v_name = "VLM_Layer_Summary" if cfg.mode == "summary" else f"VLM_Head_L{cfg.layer_idx}"

    for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
        step = dataset[step_idx]
        observation = extract_observation(step)
        
        # Get image for overlay
        img_key = next((k for k in observation if "images" in k), None)
        if img_key:
            img_raw = observation[img_key]
            # Handle [C, H, W] to [H, W, C] for visualization
            if isinstance(img_raw, torch.Tensor):
                img_rgb = img_raw.permute(1, 2, 0).cpu().numpy()
            else:
                img_rgb = np.transpose(img_raw, (1, 2, 0))
            
            if img_rgb.dtype != np.uint8:
                if img_rgb.max() <= 1.0:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                else:
                    img_rgb = img_rgb.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        action_tensor = predict_action_with_grad(observation, policy, device, preprocessor, postprocessor, cfg.use_amp)
        # VLM_AGCAMVisualizer returns [1, 16, 16] aggregated summary
        res = viz.generate_heatmap(action_tensor.abs().sum())
        
        if res is not None:
            # Create a large summary cell
            heatmap_norm = apply_robust_norm(res[0])
            grid_bgr = create_cell(img_bgr, heatmap_norm, "VLM Brain AG-CAM Summary", 640, 640)
        else:
            grid_bgr = img_bgr 
        
        if res is not None:
            if video_writer is None:
                h, w = grid_bgr.shape[:2]
                v_path = str(output_dir / f"{v_name}.mp4")
                # Use mp4v for standard MP4 support
                video_writer = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
                logger.info(f"Recording VLM video (MP4): {v_path}")
            
            if video_writer:
                video_writer.write(grid_bgr)
            
            if not cfg.headless:
                display_img = cv2.resize(grid_bgr, (0, 0), fx=cfg.display_scale, fy=cfg.display_scale)
                cv2.imshow("VLM Visualizer (Eagle Context)", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
