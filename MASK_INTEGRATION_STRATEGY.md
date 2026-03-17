# 使用 Pick/Place Mask 增强 GROOT Cross-Attention 视觉 Token

## 现有架构流程

```
RGB Image (224x224)
    ↓
SigLIP Vision Encoder (patch=14)
    ↓ 
256 Patches → Pixel Shuffle (2x2 merge) → 64 Tokens (8x8 grid)
    ↓
MLP Projector → backbone_features (B, 64, hidden_dim=1536)
    ↓
DiT Cross-Attention (encoder_hidden_states)
    ↓
预测 Action Sequence
```
