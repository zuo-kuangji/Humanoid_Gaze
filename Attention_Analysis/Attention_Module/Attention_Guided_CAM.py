import torch
import torch.nn.functional as F
import numpy as np
import cv2
from types import MethodType

class AttentionGuidedCAM:
    """
    Refined AG-CAM for SigLIP (InternVision) Perception Analysis.
    Logic:
    1. Backpropagate from raw 1024 Vision Token Energy.
    2. Apply Sigmoid(Attn) * ReLU(Grad) Hadamard Product.
    3. Dynamically reconstruct spatial topology based on input scale.
    """
    def __init__(self, vision_encoder):
        """
        vision_encoder: The internal encoder of SigLIP (e.g., policy.backbone.eagle_model.vision_model.vision_model.encoder)
        """
        self.encoder = vision_encoder
        self.num_layers = len(self.encoder.layers)
        self.attn_scores = {}
        self._patched_methods = {}
        self.attach_hooks()

    def _custom_forward(self, attn_instance, hidden_states, *args, **kwargs):
        orig_forward = self._patched_methods[id(attn_instance)]
        kwargs['output_attentions'] = True
        outputs = orig_forward(hidden_states, *args, **kwargs)
        if len(outputs) > 1 and outputs[1] is not None:
            # Capture raw attention score [B, H, Q, K]
            scores = outputs[1]
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
        Calculates AG-CAM using the 1024 Token Backpropagation.
        """
        loss_target.backward(retain_graph=True)
        
        cam_accumulator = None
        
        for i in range(self.num_layers):
            attn_instance = self.encoder.layers[i].self_attn
            if id(attn_instance) not in self.attn_scores: continue
            
            A = self.attn_scores[id(attn_instance)]
            G = A.grad
            if G is None: continue
            
            # --- Hadamard Product Reconstruction (Layer Level) ---
            # Sigmoid(Attn) * ReLU(Grad)
            mask_layer = torch.sigmoid(A) * F.relu(G)
            
            # Reduce: Sum Heads, Mean Queries (Global Saliency for No-CLS)
            # Result: [Batch, Tokens]
            layer_saliency = mask_layer.sum(dim=1).mean(dim=1)
            
            if cam_accumulator is None:
                cam_accumulator = layer_saliency
            else:
                cam_accumulator += layer_saliency
                
        if cam_accumulator is None: return None
        
        # Final result [B, 1024]
        return cam_accumulator.detach().float()

    def spatial_reconstruction(self, saliency_vector, original_image):
        """
        Rigorously reconstructs the 1024 vision tokens back to the original image resolution.
        
        SOURCE CODE AUDIT FINDING: 
        For Eagle-2.5-VL (GRooT), the preprocessor (Eagle25VLImageProcessorFast) performs a 
        direct F.resize(image, (448, 448)) when tiling results in a (1,1) grid (which is the observed case).
        This means the 640x480 image is SQUASHED into a square, NOT cropped.
        """
        B, T = saliency_vector.shape
        grid_size = int(np.sqrt(T)) # Dynamic grid (32 if 1024)
        
        # 1. Reshape to grid
        raw_cam = saliency_vector.view(B, grid_size, grid_size).cpu().numpy()[0]
        
        # 2. Robust Normalization
        raw_cam = (raw_cam - raw_cam.min()) / (raw_cam.max() - raw_cam.min() + 1e-8)
        
        # 3. DIRECT RESIZE (SQUASH RECOVERY)
        # Since the preprocessor squashed 640x480 -> 448x448, 
        # we simply stretch 32x32 saliency back to the original (img_w, img_h).
        img_h, img_w = original_image.shape[:2]
        full_cam = cv2.resize(raw_cam, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        
        return full_cam

    def detach(self):
        for layer in self.encoder.layers:
            attn_module = layer.self_attn
            if id(attn_module) in self._patched_methods:
                attn_module.forward = self._patched_methods[id(attn_module)]
