# Groot (Eagle) 图像数据处理流程分析 / Groot (Eagle) Image Processing Loop Analysis

根据代码库 (`lerobot/policies/groot`) 中的 `groot_n1.py`, `modeling_eagle2_5_vl.py` 和 configuration 文件，以下是 Eagle 策略处理图像数据的详细分析。

## 1. 核心流程概览

Eagle 的图像处理不是简单的 ViT 流程，它包含了一个关键的 **Pixel Shuffle (像素重组)** 步骤，用于压缩视觉 Token 序列长度并对齐 LLM 维度。

**路径**: `Input Image` -> `Siglip Vision Tower` -> `Pixel Shuffle` -> `MLP Projector` -> `Visual Tokens`

## 2. 逐步深度解析

### 步骤 1: 初始 Patch Embedding (Siglip)
**代码位置**: `modeling_eagle2_5_vl.py` -> `extract_feature` -> `self.vision_model`

Eagle 使用 **Siglip (Sigmoid Loss for Language Image Pre-training)** 作为视觉骨干（Backbone）。
*   **输入**: 原始 RGB 图像 (如 `224x224` 或更高分辨率)。
*   **处理**: 标准的 Vision Transformer Patch Embedding。通常使用一个 `Conv2d` 层：
    *   **Kernel Size / Stride**: `patch_size` (通常为 **14** 或 16)。
    *   **作用**: 将图像切分为 `patch_size * patch_size` 的小块。
*   **输出**: 视觉特征序列 `(Batch, Num_Patches, Hidden_Dim)`。
    *   例如，如果图像是 224x224，patch=14，则有 (224/14)*(224/14) = 16*16 = **256 个 patches**。

### 步骤 2: 视觉特征提取
特征经过 Siglip 的 Transformer 层处理。Eagle 默认提取 **倒数某一层** (默认 `select_layer=-1` 或配置中的 `-4`) 的 Hidden States。

### 步骤 3: Pixel Shuffle (关键步骤)
**代码位置**: `modeling_eagle2_5_vl.py` -> `pixel_shuffle`

这是 Eagle 处理 `patch*patch` 的独特之处。它不仅仅是线性映射，还进行了降采样聚合。

*   **配置**: `use_pixel_shuffle=True`, `downsample_ratio=0.5` (默认配置)。
*   **逻辑**:
    1.  将线性序列还原为 2D 网格 `(H, W, C)`。
    2.  应用 **Scale Factor 0.5** 的 Pixel Shuffle（实际上是逆 Pixel Shuffle 或 Space-to-Depth 操作）。
    3.  **效果**: 它将 **2x2** 的相邻 Patch 特征块合并为一个 Token。
        *   **空间维度**: 变为原来的 1/2 (例如 16x16 -> 8x8)。
        *   **通道维度**: 变为原来的 4倍 (例如 Hidden_Dim -> 4 * Hidden_Dim)。
*   **目的**: 减少 LLM 需要处理的 Token 数量 (减少4倍)，同时保留局部细节信息（通过特征拼接）。

### 步骤 4: 特征对齐 (MLP Projector)
**代码位置**: `modeling_eagle2_5_vl.py` -> `self.mlp1`

*   **输入**: 经过 Pixel Shuffle 合并后的特征，维度为 `4 * Siglip_Dim`。
*   **处理**: 一个多层感知机 (MLP)。
    *   `LayerNorm`
    *   `Linear(4 * Siglip_Dim -> LLM_Dim)`
    *   `GELU`
    *   `Linear(LLM_Dim -> LLM_Dim)`
*   **输出**: 最终的 **Visual Token**。

## 3. 最终得到的 Token

最后得到的 Token 是 **与语言模型 (Qwen2/Llama) 嵌入空间对齐的视觉表征向量**。

*   **物理意义**: 每一个 Token 代表了原始图像中 **2x2 个 Patch 区域** (即 `2*patch_size` x `2*patch_size` 的像素区域) 的语义和纹理信息。
*   **维度**: `LLM Hidden Size` (例如 Qwen2-7B 的 4096 维，具体取决于 LLM 配置)。
*   **用途**: 这些 Token 会替换掉文本输入中的 `<image>` 占位符，直接作为 LLM 的输入，让 LLM "看见" 图像。

---

### 总结
Eagle 对 `patch*patch` 的处理不仅仅是 Embedding，还包含了一个 **2x2 Patch Merge (通过 Pixel Shuffle)** 的过程。这使得最终的 Token 包含了更高密度的视觉信息，同时显著降低了序列长度，提高了推理效率。
