#!/usr/bin/env python

"""
Eagle Vision Backbone (SigLIP) Attention Visualization script.
Focuses on how the visual brain processes scenes at a foundational level.
NO text tokens, only 1024 patches (32x32 grid).
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
from Attention_Analysis.Attention_Module.agcam_visualizer import AGCAMVisualizer
from unitree_lerobot.eval_robot.utils.utils import extract_observation
import logging_mp

# Setup logging
logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)

@dataclass
class EagleVisionVisualConfig:
    """Config for Vision Backbone analysis."""
    repo_id: str = "ZUO66/handover_drinks"
    policy: PreTrainedConfig | None = None
    
    # Visualization specific parameters
    mode: str = "summary"  # "summary" or "head"
    layer_idx: int = -1    # Focus on the last vision layer by default
    display_scale: float = 1.0
    headless: bool = False
    
    # Common Eval fields for compatibility
    rename_map: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"
    use_amp: bool = False

    def __post_init__(self):
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
    """Ensure interpolate_pos_encoding for high-res handling."""
    logger.info("Applying Monkey Patch to Eagle model...")
    real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
    if not hasattr(real_model, "backbone"):
        return
    
    eagle_model = real_model.backbone.eagle_model
    
    def custom_extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True, interpolate_pos_encoding=True)
            if hasattr(vit_embeds, "last_hidden_state"):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True, interpolate_pos_encoding=True).hidden_states[self.select_layer]

        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    eagle_model.extract_feature = types.MethodType(custom_extract_feature, eagle_model)

def unwrap_no_grad(obj, method_name):
    if not hasattr(obj, method_name): return
    orig_method = getattr(obj, method_name)
    if hasattr(orig_method, "__wrapped__"):
        setattr(obj, method_name, types.MethodType(orig_method.__wrapped__, obj))

def predict_action_with_grad(observation, policy, device, preprocessor, postprocessor, use_amp):
    observation = copy(observation)
    with (
        torch.set_grad_enabled(True),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            if hasattr(observation[name], "unsqueeze"):
                if "images" in name and observation[name].dtype == torch.uint8:
                    observation[name] = observation[name].float() / 255.0
                    if observation[name].ndim == 3:
                        observation[name] = observation[name].permute(2, 0, 1)
                
                # Let the model's preprocessor handle the size to avoid 256/1024 mismatch warnings
                # if we force 224 but preprocessor expects 448.
                
                observation[name] = observation[name].unsqueeze(0).to(device)
        
        observation["task"] = ""
        observation["robot_type"] = ""
        observation = preprocessor(observation)
        
        for name in observation:
            tensor = observation[name]
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                if any(key in name.lower() for key in ["image", "eagle", "pixel"]):
                    tensor.requires_grad_(True)
        
        return policy.select_action(observation)

def apply_robust_norm(heatmap):
    # Standard min-max normalization as requested (no clipping for sinks)
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max - h_min > 1e-7:
        return (heatmap - h_min) / (h_max - h_min)
    return np.zeros_like(heatmap)

def create_cell(img_bgr, heatmap_norm, label, h, w):
    heatmap_resized = cv2.resize(heatmap_norm, (w, h))
    img_resized = cv2.resize(img_bgr, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.75, heatmap_colored, 0.25, 0)
    cv2.rectangle(overlay, (0, 0), (w-1, h-1), (255, 255, 255), 1)
    cv2.putText(overlay, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay

def visualize_layer_grid(img_bgr, layer_maps):
    num_layers = layer_maps.shape[0]
    grid_cols = 6
    grid_rows = int(np.ceil(num_layers / grid_cols))
    cell_h, cell_w = 320, 320
    grid_img = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
    for i in range(num_layers):
        heatmap_norm = apply_robust_norm(layer_maps[i])
        overlay = create_cell(img_bgr, heatmap_norm, f"Vision L{i}", cell_h, cell_w)
        r, c = i // grid_cols, i % grid_cols
        grid_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = overlay
    return grid_img

@parser.wrap()
def main(cfg: EagleVisionVisualConfig):
    init_logging()
    device = get_safe_torch_device(cfg.device, log=True)
    dataset = LeRobotDataset(repo_id=cfg.repo_id, video_backend="pyav")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    
    monkey_patch_eagle_feature_extraction(policy)
    real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
    
    unwrap_no_grad(policy, "select_action")
    unwrap_no_grad(policy, "predict_action_chunk")
    if hasattr(real_model, "action_head"):
        unwrap_no_grad(real_model.action_head, "get_action")

    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy, 
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map)
    )
    
    # Initialize AGCAM Visualizer (Vision Focused)
    from Attention_Analysis.Attention_Module.Attention_Guided_CAM import AttentionGuidedCAM
    v_encoder = real_model.backbone.eagle_model.vision_model.vision_model.encoder
    viz = AttentionGuidedCAM(v_encoder)
    
    logger.info(f"[Vision] Backbone: {type(viz.encoder)}")
    
    # Output directory relative to this script
    output_dir = Path(__file__).parents[1] / "Outputs" / "vision_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from_idx, to_idx = dataset.meta.episodes["dataset_from_index"][0], dataset.meta.episodes["dataset_to_index"][0]
    video_writer = None
    v_name = "Eagle_Vision_Layer_Summary"

    for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
        step = dataset[step_idx]
        observation = extract_observation(step)
        
        img_key = next((k for k in observation if "images" in k), None)
        if img_key:
            img_raw_tensor = observation[img_key]
            # Get original 640x480 image for geometry reference
            img_rgb_raw = img_raw_tensor.permute(1, 2, 0).cpu().numpy() if isinstance(img_raw_tensor, torch.Tensor) else np.transpose(img_raw_tensor, (1, 2, 0))
            if img_rgb_raw.max() <= 1.0: img_rgb_raw = (img_rgb_raw * 255).astype(np.uint8)
            img_bgr_raw = cv2.cvtColor(img_rgb_raw, cv2.COLOR_RGB2BGR)
        else:
            img_bgr_raw = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Forward Pass to capture gradients on the Vision tokens
        # We need to manually trigger the vision forward to isolate its tokens
        policy.train() # Enable gradient tracking
        obs_proc = preprocessor(observation)
        
        # The correct key for Eagle/SigLIP preprocessor output is 'eagle_pixel_values'
        img_key_proc = 'eagle_pixel_values' if 'eagle_pixel_values' in obs_proc else 'pixel_values'
        pixel_values = obs_proc[img_key_proc].to(device).float().requires_grad_(True)
        
        # Extract features directly from the vision backbone (SiglipVisionModel)
        eagle = real_model.backbone.eagle_model
        v_model = eagle.vision_model.vision_model
        # Use interpolate_pos_encoding=True if using high-res (448) inputs on 224-trained model
        vision_outputs = v_model(pixel_values, interpolate_pos_encoding=True)
        vision_tokens = vision_outputs.last_hidden_state # [1, 1024, Dim]
        
        # --- PROJECTOR-AWARE ANALYSIS ---
        # Instead of just SigLIP energy, we see what the TRAINED projector finds important.
        # Audit Result: In this checkpoint, use_pixel_shuffle is False. 
        # MLP1 expects 1152 dimensions (direct SigLIP output). 
        
        # 2. MLP1 Projector (This is the part YOU trained)
        # Directly apply to 1024 tokens of dim 1152
        projected_tokens = eagle.mlp1(vision_tokens)
       #projected_tokens.shapetorch.Size([1, 1024, 512])。
        # 2. Define Loss: Energy of the PROJECTED tokens (the actual input to LLM/Action)
        loss_target = projected_tokens.abs().sum()
        
        # 3. Generate heatmap (vector format [B, 1024])
        saliency_vec = viz.generate_heatmap(loss_target)
        
        if saliency_vec is not None:
            # 4. Rigorous Spatial Reconstruction (Aligning 1024 patches to 640x480 frame)
            heatmap_full = viz.spatial_reconstruction(saliency_vec, img_bgr_raw)
            
            # Create summary cell
            grid_bgr = create_cell(img_bgr_raw, heatmap_full, "AG-CAM Multi-Layer (Vision Energy)", 640, 640)
            
            if video_writer is None:
                h, w = grid_bgr.shape[:2]
                v_path = str(output_dir / f"{v_name}.mp4")
                video_writer = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
                logger.info(f"Recording Eagle Vision video (MP4): {v_path}")
            
            video_writer.write(grid_bgr)
            
            if not cfg.headless:
                display_img = cv2.resize(grid_bgr, (0, 0), fx=cfg.display_scale, fy=cfg.display_scale)
                cv2.imshow("Eagle Vision Visualizer (SigLIP)", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
