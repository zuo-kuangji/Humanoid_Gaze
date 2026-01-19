import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from types import MethodType
from collections import defaultdict

class AGCAMVisualizer:
    def __init__(self, policy, vision_token_start=20, vision_token_len=256):
        """
        Dynamic AG-CAM Visualizer for GR00T N1.5.
        Auto-detects architecture config and targets Cross-Attention layers.
        """
        self.vision_start = vision_token_start
        self.vision_len = vision_token_len
        
        # 1. 动态读取配置
        # GR00T N1.5 Action Head is a DiT
        if hasattr(policy, "action_head") and hasattr(policy.action_head, "model"):
            self.dit_model = policy.action_head.model
            self.config = self.dit_model.config
        else:
            raise ValueError("[AG-CAM] Policy does not look like GR00T (missing action_head.model)")

        self.num_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_layers
        self.hidden_size = self.config.attention_head_dim * self.num_heads
        
        # Auto-calculate grid size (e.g., sqrt(256) = 16)
        self.grid_size = int(self.vision_len ** 0.5)
        
        print(f"\n[AG-CAM] Initializing Dynamic Visualizer")
        print(f"[AG-CAM] > Detected DiT Config: {self.num_layers} Layers, {self.num_heads} Heads")
        print(f"[AG-CAM] > Detected Dimensions: Hidden={self.hidden_size}, HeadDim={self.config.attention_head_dim}")
        print(f"[AG-CAM] > Vision Token Mapping: Start={self.vision_start}, Len={self.vision_len} (Grid {self.grid_size}x{self.grid_size})")

        self.attn_scores = {}
        self.layer_indices = []
        self._patched_methods = {} # (module, original_method)
        
        # 2. 自动挂载
        self.attach_hooks()

    def _custom_attention_forward(self, attn_instance, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        Monkey-patched forward for diffusers.models.attention.Attention.
        Captures raw attention scores before softmax.
        """
        # 1. Prepare Q, K, V
        batch_size, sequence_length, _ = hidden_states.shape
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        query = attn_instance.to_q(hidden_states)
        key = attn_instance.to_k(encoder_hidden_states)
        value = attn_instance.to_v(encoder_hidden_states)
        
        query = attn_instance.head_to_batch_dim(query)
        key = attn_instance.head_to_batch_dim(key)
        value = attn_instance.head_to_batch_dim(value)
        
        # 2. Compute raw scores (This is what we want for AGCAM Sigmoid)
        # Handle scaling (usually 1/sqrt(head_dim))
        scale = attn_instance.scale
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask

        # --- AGCAM INTERVENTION ---
        # We need to retain grad for these scores
        if scores.requires_grad:
            scores.retain_grad()
        
        # Save scores to instance storage
        # Use simple object reference or layer_id if we have it mapping
        # But here self is the Visualizer. We need a way to key it.
        # id(attn_instance) is robust.
        self.attn_scores[id(attn_instance)] = scores
        # --------------------------
        
        # 3. Continue standard forward
        # 使用 softmax 之前的 scores 计算 probs
        probs = F.softmax(scores, dim=-1)
        hidden_states = torch.matmul(probs, value)
        
        hidden_states = attn_instance.batch_to_head_dim(hidden_states)
        
        # 4. Final linear projection
        hidden_states = attn_instance.to_out[0](hidden_states)
        hidden_states = attn_instance.to_out[1](hidden_states)
        
        return hidden_states

    def attach_hooks(self):
        """
        Dynamically identifies and monkey-patches Cross-Attention layers (Even indices).
        """
        print(f"[AG-CAM] > Scanning {len(self.dit_model.transformer_blocks)} blocks for Cross-Attention (Even Layers)...")
        
        for idx, block in enumerate(self.dit_model.transformer_blocks):
            # GR00T Architecture: Even = Cross-Attention, Odd = Self-Attention
            if idx % 2 == 0:
                # Target: attn1 module in BasicTransformerBlock
                attn_module = block.attn1
                
                # Save original method
                original_forward = attn_module.forward
                self._patched_methods[id(attn_module)] = (attn_module, original_forward)
                
                # Monkey Patch!
                # We bind the custom method to the Visualizer instance (self) but it needs to act like a bound method on attn_module?
                # Actually, `_custom_attention_forward` signature `(self, attn_instance, ...)`
                # We need to use functools.partial or just a bound method.
                # `MethodType(self._custom_attention_forward, attn_module)` binds it to `attn_module`.
                # So inside `_custom_attention_forward`, `self` will be `attn_module`.
                # BUT wait, `_custom_attention_forward` is defined on `AGCAMVisualizer`.
                # If we bind it to `attn_module`, `attn_instance` (first arg) will be `attn_module`.
                # BUT `self` (Visualizer instance) is needed to access `self.attn_scores`.
                
                # Correct approach: Define a closure or partial that has access to `self` (Visualizer).
                def make_hook(visualizer_instance):
                    def hook(attn_self, *args, **kwargs):
                        return visualizer_instance._custom_attention_forward(attn_self, *args, **kwargs)
                    return hook
                
                # Bind the hook to attn_module
                attn_module.forward = MethodType(make_hook(self), attn_module)
                
                self.layer_indices.append(id(attn_module))
                print(f"[AG-CAM]   - Hooked Layer {idx}: Cross-Attention (Action -> Vision)")
                
        print(f"[AG-CAM] > Successfully hooked {len(self.layer_indices)} layers.")

    def detach(self):
        """ Restore all original methods. """
        for module_id, (module, original) in self._patched_methods.items():
            module.forward = original
        self._patched_methods.clear()
        self.layer_indices = []

    def generate_heatmap(self, loss_target, norm_method='global', return_heads=False, agg_method='mean'):
        """
        Aggregates captured attention scores and gradients into a heatmap.
        loss_target: The scalar value to backprop from (e.g., action.sum().abs())
        norm_method: 'global' or 'layer'
        agg_method: 'mean' (default) or 'max' (use the strongest head).
        """
        # 1. Backprop
        self.dit_model.zero_grad()
        loss_target.backward(retain_graph=True)
        
        all_layer_maps = []
        
        for layer_id in self.layer_indices:
            scores = self.attn_scores.get(layer_id)
            if scores is None or scores.grad is None:
                continue
            
            # scores shape: (B*H, Q, T)
            # Revert to standard Sigmoid (no temperature)
            saliency = torch.sigmoid(scores)
            gradient = F.relu(scores.grad)
            cam = (gradient * saliency).mean(dim=1) # (B*H, T)
            
            # 提取视觉 Patch [20:256]
            vision_cam = cam[:, self.vision_start : self.vision_start + self.vision_len]
            
            # 重塑为 (B, Heads, 256)
            cur_batch_size = vision_cam.shape[0] // self.num_heads
            vision_cam = vision_cam.view(cur_batch_size, self.num_heads, self.grid_size, self.grid_size)
            
            # Head Aggregation
            if not return_heads:
                if agg_method == 'max':
                     # Max over heads (B, 16, 16)
                    layer_cam = vision_cam.max(dim=1)[0]
                else:
                    # Mean over heads (Standard)
                    layer_cam = vision_cam.mean(dim=1)
            else:
                # Keep heads, shape (B, H, 16, 16)
                layer_cam = vision_cam
                
            # If returning heads, we skip standard normalization or apply it per-head?
            # For now, let's keep it raw if return_heads is True to allow flexible post-processing
            if return_heads:
                 layer_map = layer_cam
            else:
                # Normalization Logic (Per Layer) for aggregated maps
                if norm_method == 'layer':
                    layer_map = self._normalize(layer_cam)
                else:
                    # If global, we simply append and normalize later
                    layer_map = layer_cam
                
            all_layer_maps.append(layer_map)

        if not all_layer_maps:
            return None

        # Stack into (B, Layers, 16, 16)
        all_layer_stack = torch.stack(all_layer_maps, dim=1)
        
        if norm_method == 'global':
            # Max-Min globally (Intensity Analysis)
            # This normalizes the entire stack (all layers) together
            all_layer_stack = self._normalize(all_layer_stack)
            
        return all_layer_stack.float().detach().cpu().numpy()

    def _normalize(self, tensor):
        """Min-Max normalization to [0, 1]"""
        # Usually per-sample normalization.
        # tensor shape (B, ...)
        B = tensor.shape[0]
        flat = tensor.view(B, -1)
        
        f_min = flat.min(dim=1, keepdim=True)[0]
        f_max = flat.max(dim=1, keepdim=True)[0]
        
        # Reshape min/max to broadcast against original tensor
        # We need to unsqueeze enough times
        view_shape = [B] + [1] * (tensor.ndim - 1)
        
        f_min = f_min.view(*view_shape)
        f_max = f_max.view(*view_shape)
        
        return (tensor - f_min) / (f_max - f_min + 1e-8)

    @staticmethod
    def create_head_grid_visualization(img_orig, head_heatmaps, cols=4):
        """
        创建一个 2x4 的网格，展示每个头的注意力。
        img_orig: 原图 (H, W, 3)
        head_heatmaps: (Heads, grid_H, grid_W)
        """
        heads = head_heatmaps.shape[0]
        if heads == 0: return img_orig
        
        # Limit to e.g. 8 heads if there are too many, or just show first few
        # For GR00T layer evolution, we are passing Layers as Heads dim effectively logic wise?
        # The user code passes (Layers, H, W) to this function.
        # So "Heads" here actually means "Layers" visually.
        
        rows = (heads + cols - 1) // cols
        
        h, w = img_orig.shape[:2]
        # 每个单元格缩小一点以便排版
        cell_h, cell_w = h // 2, w // 2
        
        grid_img = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
        
        for i in range(heads):
            # i corresponds to layer index in the stack
            # Map index i (0..7) to actual layer num (0, 2, ..., 14)
            layer_actual_num = i * 2 
            
            small_img = cv2.resize(img_orig, (cell_w, cell_h))
            overlay = AGCAMVisualizer.overlay_on_image(small_img, head_heatmaps[i], alpha=0.6)
            
            # Add label
            cv2.putText(overlay, f"Lay {layer_actual_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            r = i // cols
            c = i % cols
            grid_img[r*cell_h : (r+1)*cell_h, c*cell_w : (c+1)*cell_w] = overlay
            
        return grid_img

    @staticmethod
    def overlay_on_image(img_orig, heatmap, alpha=0.5, is_bgr=False):
        """
        img_orig: np.uint8 (H, W, 3)
        heatmap: np.float32 (16, 16)
        is_bgr: if False, we assume img_orig is RGB and convert it for OpenCV.
        """
        if not is_bgr:
            img_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_orig.copy()

        # Upsample heatmap
        heatmap_resize = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resize), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

class VLMVisualizer:
    def __init__(self, policy, vision_token_start=20, vision_token_len=256):
        """
        Specialized Visualizer for VLM (LLM Brain) Attention.
        Targets LLM Self-Attention and extracts the Reasoning-to-Vision sub-matrix.
        """
        self.vision_start = vision_token_start
        self.vision_len = vision_token_len
        self.grid_size = int(vision_token_len ** 0.5)

        # Access LLM backbone - handle both wrapped and unwrapped model instances
        real_model = policy
        if hasattr(policy, "_groot_model"):
            real_model = policy._groot_model
        
        if hasattr(real_model, "backbone"):
            self.backbone = real_model.backbone
            potential_eagle = self.backbone.eagle_model if hasattr(self.backbone, "eagle_model") else self.backbone
            self.language_model = potential_eagle.language_model
            
            # FORCE Eager Attention Implementation for Visualization
            self.language_model.config._attn_implementation = "eager"
            print(f"[VLM-Vis] > Forcing 'eager' attention implementation for matrix extraction.")
        else:
            # Fallback for direct language model access if debugging
            if hasattr(real_model, "language_model"):
                self.language_model = real_model.language_model
                self.language_model.config._attn_implementation = "eager"
            else:
                raise ValueError("[VLM-Vis] Policy does not have the expected GR00T backbone structure.")

        self.num_layers = len(self.language_model.model.layers)
        self.num_heads = self.language_model.config.num_attention_heads
        
        self.attn_scores = {}
        self._patched_methods = {}
        self.layer_indices = []
        
        print(f"\n[VLM-Vis] Initializing VLM Visualizer")
        print(f"[VLM-Vis] > Detected LLM: {self.num_layers} Layers, {self.num_heads} Heads")
        print(f"[VLM-Vis] > Vision Mapping: Start={self.vision_start}, Len={self.vision_len} (Grid {self.grid_size}x{self.grid_size})")

        self.attach_hooks()

    def _custom_llm_attention_forward(self, attn_instance, hidden_states, *args, **kwargs):
        """
        Capture attention scores from Qwen2/Llama Attention module.
        Since we need the matrix, we'll use output_attentions=True if possible, 
        or perform a manual calculation of QK^T for visualization.
        """
        # 1. We must ensure we don't break the original call
        # We'll use the original forward to get the result
        original_forward = self._patched_methods[id(attn_instance)][1]
        
        # Enable output_attentions for this call
        kwargs["output_attentions"] = True
        outputs = original_forward(hidden_states, *args, **kwargs)
        
        # 2. Extract Attention Map
        # outputs is typically (attn_output, attn_weights, past_key_value)
        if len(outputs) > 1 and outputs[1] is not None:
            scores = outputs[1] # Shape: (B, Heads, SeqLen, SeqLen)
            
            # --- AG-CAM INTERVENTION ---
            if scores.requires_grad:
                scores.retain_grad()
            self.attn_scores[id(attn_instance)] = scores
            # --------------------------
            
        return outputs

    def attach_hooks(self):
        """ Hook all LLM layers. """
        for idx, layer in enumerate(self.language_model.model.layers):
            attn_module = layer.self_attn
            
            # Save original method
            original_forward = attn_module.forward
            self._patched_methods[id(attn_module)] = (attn_module, original_forward)
            
            def make_hook(v_inst):
                def hook(attn_self, *args, **kwargs):
                    return v_inst._custom_llm_attention_forward(attn_self, *args, **kwargs)
                return hook
            
            attn_module.forward = MethodType(make_hook(self), attn_module)
            self.layer_indices.append(id(attn_module))
            
        print(f"[VLM-Vis] > Successfully hooked {len(self.layer_indices)} LLM layers.")

    def detach(self):
        for m_id, (module, original) in self._patched_methods.items():
            module.forward = original

    def generate_heatmap(self, loss_target, layer_idx=None, return_heads=False):
        """
        Aggregate LLM attention.
        Reasoning Tokens (e.g., from <img> onwards or last token) -> Vision Tokens (20:20+256)
        """
        if loss_target.requires_grad:
             loss_target.backward(retain_graph=True)

        all_layer_maps = []
        
        # Decide which layers to process
        if layer_idx is None:
            target_ids = self.layer_indices
        elif isinstance(layer_idx, int):
            target_ids = [self.layer_indices[layer_idx]]
        else: # list
            target_ids = [self.layer_indices[i] for i in layer_idx]

        for m_id in target_ids:
            scores = self.attn_scores.get(m_id)
            if scores is None or scores.grad is None:
                continue
                
            # AG-CAM: Sigmoid(A) * ReLU(grad)
            saliency = torch.sigmoid(scores) # (B, Heads, N, N)
            gradient = F.relu(scores.grad)   # (B, Heads, N, N)
            
            # Calculate CAM
            cam = (saliency * gradient) # (B, Heads, Seq, Seq)
            
            # Extract Reasoning -> Vision
            # Vision indices: [self.vision_start : self.vision_start + self.vision_len]
            v_start = self.vision_start
            v_end = v_start + self.vision_len
            
            # Reasoning tokens: usually everything AFTER the vision tokens
            r_start = v_end
            
            # Extract sub-matrix: How reasoning tokens look at vision patches
            # Shape: (B, Heads, ReasoningLen, VisionLen)
            cam_rv = cam[:, :, r_start:, v_start:v_end]
            
            # Average over reasoning tokens to get a single spatial map per head
            # Shape: (B, Heads, VisionLen)
            head_maps = cam_rv.mean(dim=2) 
            
            # Reshape VisionLen -> (Grid, Grid)
            # Shape: (B, Heads, 16, 16)
            head_maps = head_maps.view(-1, self.num_heads, self.grid_size, self.grid_size)
            
            if return_heads:
                all_layer_maps.append(head_maps.detach().float().cpu().numpy())
            else:
                # Merge heads
                layer_map = head_maps.mean(dim=1).detach().float().cpu().numpy()
                all_layer_maps.append(layer_map)

        if len(all_layer_maps) == 0:
            return None
            
        return np.stack(all_layer_maps, axis=1) # (B, Layers, Heads, 16, 16)

class EagleVisionVisualizer:
    """
    Specialized Visualizer for the Eagle Vision Backbone (SigLIP/InternVision).
    Focuses on Self-Attention within the visual encoder layers.
    Shows the 'visual brain' process without text interference.
    """
    def __init__(self, policy, grid_size=16):
        """
        policy: The GrootPolicy or GR00T model instance.
        grid_size: Number of patches per side (e.g., 16 for 224x224 SigLIP).
        """
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # Access Vision Backbone
        real_model = policy
        if hasattr(policy, "_groot_model"):
            real_model = policy._groot_model
        
        if hasattr(real_model, "backbone"):
            self.vision_model = real_model.backbone.eagle_model.vision_model
            # Force eager if applicable, though SigLIP often uses SDPA which is hookable
            if hasattr(self.vision_model.config, "_attn_implementation"):
                self.vision_model.config._attn_implementation = "eager"
        else:
            raise ValueError("[Vision-Vis] Policy does not have the expected Eagle backbone structure.")

        # Identify layers: vision_model.vision_model.encoder.layers
        self.encoder = self.vision_model.vision_model.encoder
        self.num_layers = len(self.encoder.layers)
        self.num_heads = self.vision_model.config.num_attention_heads
        
        self.attn_scores = {}
        self._patched_methods = {}
        
        print(f"\n[Vision-Vis] Initializing Eagle Vision Visualizer")
        print(f"[Vision-Vis] > Detected SigLIP: {self.num_layers} Layers, {self.num_heads} Heads")
        print(f"[Vision-Vis] > Resolution Grid: {grid_size}x{grid_size}")

        self.attach_hooks()

    def _custom_vision_attention_forward(self, attn_instance, hidden_states, *args, **kwargs):
        """
        Monkey-patched forward for SiglipAttention.
        Captures Self-Attention scores.
        """
        # SigLIP usually calculates Q, K, V
        # Since we use 'eager' mode, it will follow the base transformer logic
        # We'll just extract the weights if they are computed or re-compute them for visualization
        # However, to avoid heavy re-computation, we rely on the gradient-based approach (Grad-CAM) 
        # but we also want the raw scores for the 'heads' view.
        
        # We use a similar trick to capture 'attn_weights' if the original model produces them.
        # But SiglipAttention forward usually looks like:
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Let's perform a lightweight capture by wrapping the forward.
        orig_forward = self._patched_methods[id(attn_instance)]
        
        # Force the model to return attentions if possible, or we manually intervene
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        
        # SigLIP Attention returns (attn_output, attn_weights, present_key_value)
        if len(outputs) > 1 and outputs[1] is not None:
            self.attn_scores[id(attn_instance)] = outputs[1]
            if outputs[1].requires_grad:
                outputs[1].retain_grad()
            
        return outputs

    def attach_hooks(self):
        """Hook every encoder layer's self-attention."""
        for i, layer in enumerate(self.encoder.layers):
            attn_module = layer.self_attn
            if id(attn_module) not in self._patched_methods:
                self._patched_methods[id(attn_module)] = attn_module.forward
                
                def make_hook(v_inst):
                    def hook(attn_self, *args, **kwargs):
                        return v_inst._custom_vision_attention_forward(attn_self, *args, **kwargs)
                    return hook
                
                attn_module.forward = MethodType(make_hook(self), attn_module)

    def detach(self):
        for i, layer in enumerate(self.encoder.layers):
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]
        self._patched_methods.clear()
        self.attn_scores.clear()

    def generate_heatmap(self, loss_target, layer_idx=None, return_heads=False):
        """
        Aggregate Vision attention.
        Self-Attention is (B, Heads, Tokens, Tokens).
        We focus on the average attention each token receives (Salience).
        """
        loss_target.backward(retain_graph=True)
        
        all_layer_maps = []
        target_layers = [layer_idx] if layer_idx is not None else range(self.num_layers)
        
        for i in target_layers:
            attn_instance = self.encoder.layers[i].self_attn
            if id(attn_instance) not in self.attn_scores:
                continue
            
            # (Batch, Heads, Q_Len, K_Len)
            scores = self.attn_scores[id(attn_instance)]
            grads = scores.grad
            
            if grads is not None:
                # Grad-CAM style: Weight heads by their gradients
                weights = grads.mean(dim=(2, 3), keepdim=True)
                head_maps = (weights * scores).sum(dim=1) # [B, Q_Len, K_Len]
                # Further reduce to see 'where' the model is looking overall
                head_maps = head_maps.mean(dim=1) # [B, K_Len]
            else:
                # Fallback to plain attention if no gradients
                # Mean across heads and queries
                head_maps = scores.mean(dim=(1, 2)) # [B, K_Len]
                
            # Reshape to grid [B, Grid, Grid]
            # Dynamically detect grid size based on token count
            num_tokens = head_maps.shape[-1]
            grid_size = int(np.sqrt(num_tokens))
            layer_map = head_maps.view(-1, grid_size, grid_size)
            all_layer_maps.append(layer_map.detach().float().cpu().numpy())

        if len(all_layer_maps) == 0:
            return None
            
        # Stack to [Layers, B, Grid, Grid] and then take first batch -> [Layers, Grid, Grid]
        res = np.stack(all_layer_maps)
        return res[:, 0]

class AGCAMVisualizer:
    """
    Implementation of 'Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention' (AAAI 2024).
    Paper: https://arxiv.org/abs/2303.14616
    Code Logic: Multi-layer aggregation with Sigmoid-Attention and ReLU-Gradients.
    """
    def __init__(self, policy, grid_size=32):
        self.grid_size = grid_size
        
        # Access Vision Backbone (SigLIP)
        real_model = policy
        if hasattr(policy, "_groot_model"):
            real_model = policy._groot_model
        
        if hasattr(real_model, "backbone"):
            self.vision_model = real_model.backbone.eagle_model.vision_model
            if hasattr(self.vision_model.config, "_attn_implementation"):
                self.vision_model.config._attn_implementation = "eager"
        else:
            raise ValueError("[AG-CAM] Policy does not have the expected Eagle backbone structure.")

        self.encoder = self.vision_model.vision_model.encoder
        self.num_layers = len(self.encoder.layers)
        
        self.attn_scores = {}
        self._patched_methods = {}
        
        print(f"\n[AG-CAM] Initializing Official AG-CAM Logic")
        print(f"[AG-CAM] > Target: SigLIP Vision Backbone ({self.num_layers} Layers)")
        self.attach_hooks()

    def _custom_forward(self, attn_instance, hidden_states, *args, **kwargs):
        orig_forward = self._patched_methods[id(attn_instance)]
        # We need the raw attention weights (pre-softmax/normalized)
        # Note: AGCAM authors use Sigmoid on the weights.
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        
        if len(outputs) > 1 and outputs[1] is not None:
            self.attn_scores[id(attn_instance)] = outputs[1]
            if outputs[1].requires_grad:
                outputs[1].retain_grad()
        return outputs

    def attach_hooks(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) not in self._patched_methods:
                self._patched_methods[id(attn_module)] = attn_module.forward
                attn_module.forward = MethodType(lambda inst, *a, **k: self._custom_forward(inst, *a, **k), attn_module)

    def generate_heatmap(self, loss_target):
        """
        AG-CAM logic:
        1. Gradient = ReLU(Actual Grad)
        2. Attention = Sigmoid(Actual Attention)
        3. Mask = Gradient * Attention
        4. Sum across heads and layers
        """
        loss_target.backward(retain_graph=True)
        
        mask_accumulator = None
        
        for i in range(self.num_layers):
            attn_module = self.encoder.layers[i].self_attn
            if id(attn_module) not in self.attn_scores:
                continue
            
            # scores: (Batch, Heads, Q_Len, K_Len)
            scores = self.attn_scores[id(attn_module)]
            grads = scores.grad
            
            if grads is None:
                continue
                
            # Equation 7: grad = ReLU(grads)
            # Equation 2: attn = Sigmoid(scores)
            grad_filtered = torch.nn.functional.relu(grads)
            attn_norm = torch.sigmoid(scores)
            
            # mask_layer: (Batch, Heads, Q_Len, K_Len)
            mask_layer = grad_filtered * attn_norm
            
            # Aggregate heads: sum (as per AGCAM implementation)
            mask_reduced = mask_layer.sum(dim=1) # [B, Q_Len, K_Len]
            # Average across queries to see global patch importance
            mask_reduced = mask_reduced.mean(dim=1) # [B, K_Len]
            
            if mask_accumulator is None:
                mask_accumulator = mask_reduced
            else:
                mask_accumulator += mask_reduced
                
        if mask_accumulator is None:
            return None
            
        # Reshape to grid
        num_tokens = mask_accumulator.shape[-1]
        grid_dim = int(np.sqrt(num_tokens))
        res = mask_accumulator.view(grid_dim, grid_dim).detach().float().cpu().numpy()
        
        return res[np.newaxis, ...]

    def detach(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]
        self._patched_methods.clear()
        self.attn_scores.clear()

class VLM_AGCAMVisualizer:
    """
    AG-CAM implementation for the VLM (LLM Brain).
    Adapts the AAAI 2024 logic to a causal multimodal LLM by focusing on Reasoning-to-Vision attention.
    """
    def __init__(self, policy, vision_token_start=20, vision_token_len=256):
        self.vision_start = vision_token_start
        self.vision_len = vision_token_len
        self.grid_size = int(vision_token_len ** 0.5)

        # Access LLM backbone
        real_model = policy
        if hasattr(policy, "_groot_model"):
            real_model = policy._groot_model
        
        if hasattr(real_model, "backbone"):
            potential_eagle = real_model.backbone.eagle_model if hasattr(real_model.backbone, "eagle_model") else real_model.backbone
            self.language_model = potential_eagle.language_model
            self.language_model.config._attn_implementation = "eager"
        else:
            raise ValueError("[VLM-AGCAM] Policy does not have the expected GR00T backbone structure.")

        self.num_layers = len(self.language_model.model.layers)
        
        self.attn_scores = {}
        self._patched_methods = {}
        
        print(f"\n[VLM-AGCAM] Initializing Multi-Layer AG-CAM for LLM Brain")
        print(f"[VLM-AGCAM] > Detected LLM: {self.num_layers} Layers")
        print(f"[VLM-AGCAM] > Vision Mapping: Start={self.vision_start}, Len={self.vision_len} (Grid {self.grid_size}x{self.grid_size})")
        self.attach_hooks()

    def _custom_forward(self, attn_instance, hidden_states, *args, **kwargs):
        orig_forward = self._patched_methods[id(attn_instance)]
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        
        if len(outputs) > 1 and outputs[1] is not None:
            self.attn_scores[id(attn_instance)] = outputs[1]
            if outputs[1].requires_grad:
                outputs[1].retain_grad()
        return outputs

    def attach_hooks(self):
        for layer in self.language_model.model.layers:
            attn_module = layer.self_attn
            if id(attn_module) not in self._patched_methods:
                self._patched_methods[id(attn_module)] = attn_module.forward
                attn_module.forward = MethodType(lambda inst, *a, **k: self._custom_forward(inst, *a, **k), attn_module)

    def generate_heatmap(self, loss_target):
        """
        LLM AG-CAM logic:
        1. Gradient = ReLU(Actual Grad)
        2. Attention = Sigmoid(Actual Attention)
        3. Slicing: Extract [ReasoningTokens, VisionTokens] sub-matrix.
        4. Accumulate across all layers.
        """
        loss_target.backward(retain_graph=True)
        
        mask_accumulator = None
        
        for i in range(self.num_layers):
            attn_module = self.language_model.model.layers[i].self_attn
            if id(attn_module) not in self.attn_scores:
                continue
            
            # scores: (Batch, Heads, SeqLen, SeqLen)
            scores = self.attn_scores[id(attn_module)]
            grads = scores.grad
            
            if grads is None:
                continue
                
            # AG-CAM official math
            grad_filtered = torch.nn.functional.relu(grads)
            attn_norm = torch.sigmoid(scores)
            
            # full_mask: (Batch, Heads, SeqLen, SeqLen)
            full_mask = grad_filtered * attn_norm
            
            # Extract Reasoning-to-Vision part
            # Vision: [start : start+len]
            # Reasoning: [start+len : ] (The policy tokens at the end)
            v_start = self.vision_start
            v_end = v_start + self.vision_len
            r_start = v_end
            
            # cam_rv: (Batch, Heads, ReasoningLen, VisionLen)
            if full_mask.shape[-2] > r_start:
                cam_rv = full_mask[:, :, r_start:, v_start:v_end]
                
                # Aggregate Heads: sum
                # Aggregate Reasoning Tokens: mean (since each reasoning step should look at the vision patches)
                mask_reduced = cam_rv.sum(dim=1).mean(dim=1) # [B, VisionLen]
                
                if mask_accumulator is None:
                    mask_accumulator = mask_reduced
                else:
                    mask_accumulator += mask_reduced
                
        if mask_accumulator is None:
            return None
            
        # Reshape to grid
        res = mask_accumulator.view(self.grid_size, self.grid_size).detach().float().cpu().numpy()
        
        return res[np.newaxis, ...]

    def detach(self):
        for layer in self.language_model.model.layers:
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]
        self._patched_methods.clear()
        self.attn_scores.clear()
