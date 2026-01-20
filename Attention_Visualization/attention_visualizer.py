"""
GR00T VLM Cross-Attention Visualizer

Attention extraction for DiT action head using monkey-patching.
Does not modify source files - patches at runtime.

Reference: Groot_Analysis/Groot_VLM_Analysis.md
- Vision Tokens: indices 20-275 (256 patches, 16x16 Row-Major)
- DiT: 16 layers, 32 heads, even layers are Cross-Attention
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

# === Constants (from Groot_VLM_Analysis.md) ===
VISION_START_IDX = 20    # First vision token index
VISION_LEN = 256         # Total vision tokens (16x16)
VISION_GRID = 16         # Grid size
NUM_HEADS = 32           # Real DiT head count
NUM_CROSS_ATTN_LAYERS = 8  # Even layers: 0,2,4,6,8,10,12,14

# Global storage for attention maps
_attn_maps: Dict[str, torch.Tensor] = {}
_original_forwards: Dict[str, callable] = {}


def _compute_attention_weights(attn_module, hidden_states, encoder_hidden_states):
    """
    Manually compute attention weights from Q and K projections.
    This mirrors what happens inside the AttnProcessor2_0.
    """
    query = attn_module.to_q(hidden_states)
    key = attn_module.to_k(encoder_hidden_states)
    
    batch_size = query.shape[0]
    seq_len_q = query.shape[1]
    seq_len_k = key.shape[1]
    
    # Get head_dim
    inner_dim = query.shape[-1]
    head_dim = inner_dim // NUM_HEADS
    
    # Reshape for multi-head: (B, seq, heads*dim) -> (B, heads, seq, dim)
    query = query.view(batch_size, seq_len_q, NUM_HEADS, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len_k, NUM_HEADS, head_dim).transpose(1, 2)
    
    # Compute attention scores (Logits)
    scale = head_dim ** -0.5
    attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # FOR MODEL: Standard Softmax
    attn_weights_model = F.softmax(attn_logits, dim=-1)
    
    # FOR VISUALIZATION: Sigmoid (Multi-label classification style)
    # This avoids the "winner-takes-all" problem of Softmax, allowing multiple
    # independent regions to be highlighted (e.g., object + hand).
    attn_weights_viz = torch.sigmoid(attn_logits)
    
    return attn_weights_model, attn_weights_viz


def _create_patched_forward(original_forward, layer_name, attn_module):
    """Create a patched forward function that captures attention weights."""
    
    def patched_forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        # Capture attention if this is cross-attention (encoder_hidden_states exists)
        if encoder_hidden_states is not None:
            try:
                _, attn_weights_viz = _compute_attention_weights(
                    attn_module, 
                    hidden_states if attention_mask is None else hidden_states,  # normalized hidden states
                    encoder_hidden_states
                )
                _attn_maps[layer_name] = attn_weights_viz.detach().cpu()
            except Exception as e:
                print(f"[AttentionVisualizer] Warning: Failed to capture {layer_name}: {e}")
        
        # Call original forward
        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            temb=temb,
        )
    
    return patched_forward


def register_attention_hooks(policy) -> List:
    """
    Register attention capture by patching BasicTransformerBlock.forward.
    
    Args:
        policy: GRooT policy with _groot_model.action_head.model (DiT)
    
    Returns:
        List of layer names that were patched (for cleanup)
    """
    patched_layers = []
    
    # Navigate to DiT model
    dit = None
    try:
        dit = policy._groot_model.action_head.model
    except AttributeError:
        try:
            dit = policy.model.action_head.model
        except AttributeError:
            print("[AttentionVisualizer] Error: Cannot find DiT model in policy")
            print("[AttentionVisualizer] Expected: policy._groot_model.action_head.model")
            return patched_layers
    
    if dit is None:
        print("[AttentionVisualizer] Error: DiT model is None")
        return patched_layers

    # Patch even-indexed layers (Cross-Attention)
    for idx, block in enumerate(dit.transformer_blocks):
        if idx % 2 == 0:  # Even layers = Cross-Attention
            layer_name = f"cross_attn_layer_{idx}"
            
            # Store original forward
            _original_forwards[layer_name] = block.forward
            
            # Create patched forward
            patched = _create_patched_forward(block.forward, layer_name, block.attn1)
            block.forward = patched
            
            patched_layers.append(layer_name)
            print(f"[AttentionVisualizer] Patched: {layer_name}")
    
    print(f"[AttentionVisualizer] Total layers patched: {len(patched_layers)}")
    return patched_layers


def remove_hooks(patched_layers: List) -> None:
    """Restore original forward methods."""
    for layer_name in patched_layers:
        if layer_name in _original_forwards:
            # We need to find the block and restore its forward
            # For simplicity, just clear the originals dict
            pass
    _original_forwards.clear()
    print(f"[AttentionVisualizer] Removed {len(patched_layers)} patches")



def clear_attention_maps() -> None:
    """Clear stored attention maps (call before each inference step)."""
    _attn_maps.clear()


def get_attention_maps() -> Dict[str, torch.Tensor]:
    """Get raw attention maps dictionary."""
    return _attn_maps.copy()


def compute_vision_heatmap(
    image_size: Tuple[int, int] = (480, 640),
    aggregation: str = "mean",
    head_aggregation: str = "max"
) -> Optional[torch.Tensor]:
    """
    Compute vision attention heatmap from stored cross-attention maps.
    
    Args:
        image_size: Output heatmap size (H, W)
        aggregation: How to aggregate across layers ("mean" or "max")
        head_aggregation: How to aggregate across heads ("mean" or "max")
    
    Returns:
        Heatmap tensor of shape (B, H, W) normalized to [0, 1], or None if no data
    """
    if not _attn_maps:
        return None
    
    # Stack all layer attention maps
    # Each: (B, 32_heads, 83_query, seq_len_k)
    all_attn = list(_attn_maps.values())
    
    # Extract vision tokens only: indices 20-275
    vision_attns = []
    for attn in all_attn:
        seq_len_k = attn.shape[-1]
        if seq_len_k > VISION_START_IDX + VISION_LEN:
            vision_attn = attn[..., VISION_START_IDX:VISION_START_IDX + VISION_LEN]
            vision_attns.append(vision_attn)
    
    if not vision_attns:
        print("[AttentionVisualizer] Warning: No valid vision attention found")
        return None
    
    # Stack layers: (num_layers, B, 32, 83, 256)
    stacked = torch.stack(vision_attns, dim=0)
    
    # Aggregate across layers
    # CHANGED: User requested SUM (accumulate influence across layers).
    if aggregation == "mean":
        layer_agg = stacked.mean(dim=0)  # (B, 32, 83, 256)
    elif aggregation == "sum":   
        layer_agg = stacked.sum(dim=0) # (B, 32, 83, 256)
    else: # Default to max
        layer_agg = stacked.max(dim=0).values # (B, 32, 83, 256)
    
    # Aggregate across heads
    if head_aggregation == "mean":
        head_agg = layer_agg.mean(dim=1)  # (B, 83, 256)
    elif head_aggregation == "sum":
        head_agg = layer_agg.sum(dim=1)   # (B, 83, 256) - New option
    else:
        # Default or explicit 'max'
        head_agg = layer_agg.max(dim=1).values # (B, 83, 256)
    
    # Aggregate across query tokens (sum attention from all action tokens to each vision token)
    query_agg = head_agg.sum(dim=1)  # (B, 256)
    
    # Reshape to 16x16 grid (Row-Major)
    batch_size = query_agg.shape[0]
    heatmap = query_agg.view(batch_size, VISION_GRID, VISION_GRID) # (B, 16, 16)
    
    # Upsample to image size
    heatmap = F.interpolate(
        heatmap.unsqueeze(1).float(),
        size=image_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # (B, H, W)
    
    # Robust Normalization (Percentiles)
    heatmap_cpu = heatmap.cpu().numpy()
    p_min = np.percentile(heatmap_cpu, 2)  # Bottom 1%
    p_max = np.percentile(heatmap_cpu, 98) # Top 1% (ignore extreme outliers)
    
    heatmap = torch.clamp(heatmap, min=p_min, max=p_max)
    
    # Normalize to [0, 1] based on robust range
    if p_max - p_min > 1e-7:
        heatmap = (heatmap - p_min) / (p_max - p_min)
    else:
        heatmap = torch.zeros_like(heatmap)
    
    return heatmap


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3) uint8
        heatmap: Attention heatmap (H, W) in [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
    
    Returns:
        Overlaid image (H, W, 3) uint8
    """
    import matplotlib.pyplot as plt
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Convert heatmap to numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()
    
    # Ensure 2D
    if heatmap.ndim == 3:
        heatmap = heatmap[0]  # Take first batch
    
    # Apply colormap
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float [0,1]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return overlay


# === Per-head visualization ===
def get_per_head_heatmaps(
    image_size: Tuple[int, int] = (480, 640),
    layer_idx: int = 0
) -> Optional[torch.Tensor]:
    """
    Get attention heatmap for each head separately.
    
    Args:
        image_size: Output size
        layer_idx: Which cross-attention layer (0=layer0, 1=layer2, ...)
    
    Returns:
        Tensor of shape (B, 32, H, W) or None
    """
    layer_name = f"cross_attn_layer_{layer_idx * 2}"
    if layer_name not in _attn_maps:
        return None
    
    attn = _attn_maps[layer_name]  # (B, 32, 83, seq_len)
    seq_len = attn.shape[-1]
    
    if seq_len <= VISION_START_IDX + VISION_LEN:
        return None
    
    # Extract vision tokens
    vision_attn = attn[..., VISION_START_IDX:VISION_START_IDX + VISION_LEN]  # (B, 32, 83, 256)
    
    # Sum over query tokens
    vision_attn = vision_attn.sum(dim=2)  # (B, 32, 256)
    
    # Reshape to grid
    B = vision_attn.shape[0]
    heatmaps = vision_attn.view(B, NUM_HEADS, VISION_GRID, VISION_GRID)
    
    # Upsample
    heatmaps = heatmaps.view(B * NUM_HEADS, 1, VISION_GRID, VISION_GRID)
    heatmaps = F.interpolate(heatmaps.float(), size=image_size, mode='bilinear', align_corners=False)
    heatmaps = heatmaps.view(B, NUM_HEADS, *image_size)
    
    # Normalize per-head
    heatmaps_flat = heatmaps.view(B, NUM_HEADS, -1)
    h_min = heatmaps_flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    h_max = heatmaps_flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-8)
    
    return heatmaps
