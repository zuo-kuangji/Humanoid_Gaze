import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from types import MethodType

# =========================================================================================
# VERSION 1: LEGACY AG-CAM (CURRENT STABLE VERSION)
# =========================================================================================
# This is the version we are currently using, which uses global head summation.

class AGCAMVisualizer_Legacy:
    """
    Legacy Implementation (Current Stable).
    Logic: Sum(Sigmoid(Attn) * ReLU(Grad)) across all heads and layers.
    """
    def __init__(self, policy, grid_size=32):
        self.grid_size = grid_size
        real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
        self.vision_model = real_model.backbone.eagle_model.vision_model
        self.encoder = self.vision_model.vision_model.encoder
        self.num_layers = len(self.encoder.layers)
        self.attn_scores = {}
        self._patched_methods = {}
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
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) not in self._patched_methods:
                self._patched_methods[id(attn_module)] = attn_module.forward
                attn_module.forward = MethodType(lambda inst, *a, **k: self._custom_forward(inst, *a, **k), attn_module)

    def generate_heatmap(self, loss_target):
        loss_target.backward(retain_graph=True)
        mask_accumulator = None
        for i in range(self.num_layers):
            attn_module = self.encoder.layers[i].self_attn
            if id(attn_module) not in self.attn_scores: continue
            scores = self.attn_scores[id(attn_module)]
            grads = scores.grad
            if grads is None: continue
            
            # Core Logic: Sigmoid + ReLU
            mask_layer = torch.nn.functional.relu(grads) * torch.sigmoid(scores)
            mask_reduced = mask_layer.sum(dim=1).mean(dim=1) # Sum Heads, Mean Queries
            
            if mask_accumulator is None: mask_accumulator = mask_reduced
            else: mask_accumulator += mask_reduced
                
        if mask_accumulator is None: return None
        res = mask_accumulator.view(self.grid_size, self.grid_size).detach().float().cpu().numpy()
        return res[np.newaxis, ...]

    def detach(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]

# =========================================================================================
# VERSION 2: OFFICIAL AG-CAM (AAAI 2024 REFINED)
# =========================================================================================
# This version follows the specific findings in the 'Technical Implementation Plan':
# 1. Skip-connection gradient tracking (handled by full-model backward).
# 2. Specific emphasis on Sigmoid over Softmax.
# 3. Head-level aggregation refinement.

class AGCAMVisualizer_Official:
    """
    Official Reference Implementation (AAAI 2024 Math-Aligned).
    Refinement: 
    1. Specifically targets the 'Query-to-Key' relationship for decision tokens.
    2. Strictly applies ReLU(Grad) and Sigmoid(Attn) fusion.
    3. Handles Spatial Patch reconstruction by discarding non-spatial tokens.
    """
    def __init__(self, policy, grid_size=32, target_query_idx=0):
        """
        target_query_idx: In ViT, 0 is usually CLS. In pooled backbones, this might vary.
        For robot backbone, we'll try to focus on the 'Action-relevant' query logic.
        """
        self.grid_size = grid_size
        self.target_query_idx = target_query_idx
        real_model = policy._groot_model if hasattr(policy, "_groot_model") else policy
        self.vision_model = real_model.backbone.eagle_model.vision_model
        self.encoder = self.vision_model.vision_model.encoder
        self.num_layers = len(self.encoder.layers)
        
        self.attn_scores = {}
        self._patched_methods = {}
        print(f"[AGCAM-Official] DEEP Implementation Active: Targeting Query {target_query_idx}")
        self.attach_hooks()

    def _custom_forward(self, attn_instance, hidden_states, *args, **kwargs):
        orig_forward = self._patched_methods[id(attn_instance)]
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        
        if len(outputs) > 1 and outputs[1] is not None:
            scores = outputs[1] # [B, H, Q, T]
            self.attn_scores[id(attn_instance)] = scores
            if scores.requires_grad:
                scores.retain_grad()
        return outputs

    def attach_hooks(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) not in self._patched_methods:
                self._patched_methods[id(attn_module)] = attn_module.forward
                attn_module.forward = MethodType(lambda inst, *a, **k: self._custom_forward(inst, *a, **k), attn_module)

    def generate_heatmap(self, loss_target):
        """
        Official Aggregation Logic (Mathematical Proof Aligned):
        L^c = Sum_k Sum_h ( Sigmoid(A_h_k[target_q]) * ReLU(Grad_h_k[target_q]) )

        Note: SigLIP (InternVision-6B) usually does NOT have a CLS token.
        If we detect a perfect square token count (e.g. 1024), we switch to 
        Global Saliency (Mean Query) across all patches, which approximates 
        the global pooling logic used in these models.
        """
        loss_target.backward(retain_graph=True)
        
        cam_accumulator = None
        
        for i in range(self.num_layers):
            attn_instance = self.encoder.layers[i].self_attn
            if id(attn_instance) not in self.attn_scores: continue
            
            # A: [Batch, Heads, Q_Len, T_Len]
            A = self.attn_scores[id(attn_instance)]
            G = A.grad
            
            if G is None: continue
            
            # --- ARCHITECTURE ADAPTATION ---
            num_tokens = A.shape[-1]
            is_perfect_square = int(np.sqrt(num_tokens))**2 == num_tokens
            
            # If No CLS Token (like SigLIP), we use Mean of all queries to get global saliency
            if is_perfect_square:
                a_q = A.mean(dim=2, keepdim=True)
                g_q = G.mean(dim=2, keepdim=True)
            elif A.shape[2] > self.target_query_idx:
                # Official ViT logic: Target only the CLS query row
                a_q = A[:, :, self.target_query_idx:self.target_query_idx+1, :]
                g_q = G[:, :, self.target_query_idx:self.target_query_idx+1, :]
            else:
                a_q = A.mean(dim=2, keepdim=True)
                g_q = G.mean(dim=2, keepdim=True)

            # 2. Filtering (Eq 7 & Eq 2)
            f_maps = torch.sigmoid(a_q)
            alpha_weights = torch.nn.functional.relu(g_q)
            
            # 3. Hadamard Fusion
            mask_layer = f_maps * alpha_weights
            
            # 4. Aggregation (Sum over Heads)
            mask_reduced = mask_layer.sum(dim=1).squeeze(1) # [B, T_Len]
            
            # 5. Token removal logic
            if not is_perfect_square:
                # Discard CLS token at index 0 if it exists
                num_spatial = self.grid_size * self.grid_size
                if mask_reduced.shape[1] > num_spatial:
                     # Usually CLS is at 0, or spatial patches are at the end
                     mask_reduced = mask_reduced[:, -num_spatial:]
            
            if cam_accumulator is None:
                cam_accumulator = mask_reduced
            else:
                cam_accumulator += mask_reduced

        if cam_accumulator is None: return None

        # 6. Global Spatial Reconstruction
        heatmap = cam_accumulator.view(self.grid_size, self.grid_size)
        
        # 7. Official Robust Normalization
        h_min, h_max = heatmap.min(), heatmap.max()
        heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
        
        return heatmap.detach().cpu().numpy()[np.newaxis, ...]

    def detach(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]

# =========================================================================================
# VLM RE Reasoning-to-Vision (OFFICIAL REF)
# =========================================================================================

class VLM_AGCAM_Official:
    """
    VLM variant following Official Ref logic.
    Optimized for reasoning steps tracking.
    """
    def __init__(self, policy, v_start=20, v_len=256):
        self.v_start, self.v_len = v_start, v_len
        self.grid_size = int(v_len ** 0.5)
        # Access language model
        real_policy = policy._groot_model if hasattr(policy, "_groot_model") else policy
        self.lm = real_policy.backbone.eagle_model.language_model
        self.lm.config._attn_implementation = "eager"
        self.attn_scores = {}
        self._patched_methods = {}
        self.attach_hooks()

    def _custom_forward(self, attn_instance, hidden_states, *args, **kwargs):
        orig_forward = self._patched_methods[id(attn_instance)]
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        if len(outputs) > 1 and outputs[1] is not None:
            self.attn_scores[id(attn_instance)] = outputs[1]
            if outputs[1].requires_grad: outputs[1].retain_grad()
        return outputs

    def attach_hooks(self):
        for layer in self.lm.model.layers:
            attn_module = layer.self_attn
            self._patched_methods[id(attn_module)] = attn_module.forward
            attn_module.forward = MethodType(lambda inst, *a, **k: self._custom_forward(inst, *a, **k), attn_module)

    def generate_heatmap(self, loss_target):
        loss_target.backward(retain_graph=True)
        acc = None
        for layer in self.lm.model.layers:
            attn_id = id(layer.self_attn)
            if attn_id not in self.attn_scores: continue
            scores = self.attn_scores[attn_id]
            grads = scores.grad
            if grads is None: continue
            
            # Following Sigmoid + ReLU
            mask = torch.sigmoid(scores) * torch.nn.functional.relu(grads)
            
            # Slice Reasoning -> Vision
            v_end = self.v_start + self.v_len
            cam_rv = mask[:, :, v_end:, self.v_start:v_end]
            
            # Aggregate: Sum Heads, Mean Reasoning Steps
            layer_map = cam_rv.sum(dim=1).mean(dim=1) # [B, VisionLen]
            if acc is None: acc = layer_map
            else: acc += layer_map
            
        if acc is None: return None
        res = acc.view(self.grid_size, self.grid_size).detach().float().cpu().numpy()
        # Min-max norm
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        return res[np.newaxis, ...]

    def detach(self):
        for layer in self.lm.model.layers:
            layer.self_attn.forward = self._patched_methods[id(layer.self_attn)]
