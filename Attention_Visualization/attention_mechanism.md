# GR00T 注意力可视化技术文档 (Technical Documentation)

本文档详细说明了 GR00T (VLM-Action Policy) 的交叉注意力 (Cross-Attention) 可视化系统的实现原理。

## 1. 系统架构概览

可视化系统主要由两部分组成：
1.  **核心可视化器 (`attention_visualizer.py`)**：负责拦截模型内部信号，计算注意力权重，生成热力图。
2.  **评估流水线 (`eval_attention.py` / `eval_attention_visualization.py`)**：负责推理循环、视频帧捕获以及将热力图与相机图像同步。

---

## 2. 核心机制详解 (`attention_visualizer.py`)

### 2.1 Monkey Patching 策略 (拦截机制)
为了在不修改原始库代码的情况下捕获 Cross-Attention 权重，我们采用了 **Monkey Patching** (运行时动态替换) 技术拦截 `diffusers.models.attention.BasicTransformerBlock.forward`。

### 2.2 Token 选择与 QK 计算 (Softmax 策略)

我们复现了 `AttnProcessor2_0` 内部的注意力计算逻辑，并根据 Explainable AI (XAI) 的最佳实践进行了关键改进。

#### 2.2.1 输入张量与 Token 来源
-   **Query ($Q$)**: 来自 Action Head (DiT) 的 Action Tokens (序列长度 49)。
-   **Key ($K$)**: 来自 VLM (Eagle) 的视觉特征 (序列长度 294)。
-   **Vision Token 切片**: 我们提取索引 **`[20 : 276]`**，对应 $16 \times 16$ 的图像 Patch。

#### 2.2.2 计算公式：Softmax
为了获得清晰、高信噪比的注意力热力图，我们采用 **Softmax** 策略，而非之前的 Sigmoid。

$$ \text{AttnViz} = \text{Softmax}(\frac{Q \cdot K^T}{\sqrt{d}}) $$

**选择 Softmax 的原因**：
*   **稀疏性 (Sparsity)**：Softmax 具有极强的竞争机制（归一化为1），能够强制压制无关背景（Background Suppression），使非关注区域的权重接近于 0。
*   **抗噪能力**：真实的 Cross-Attention 包含 **12,800 条路径**（8层 × 32头 × 50 Token）。如果是 Sigmoid（底噪约 0.5），叠加后的背景噪音高达 6000+，完全淹没信号。而 Softmax 将底噪压至几乎为零，使得 **SUM 聚合** 成为可能。

### 2.3 聚合策略 (Aggregation)

我们采用 **SUM** 聚合策略，以捕捉所有层和所有 Token 的综合影响：
1.  **Layer Aggregation**: `SUM` (累加所有 Cross-Attention 层的贡献)
2.  **Head Aggregation**: `SUM` (累加所有注意力头的贡献)
3.  **Query Aggregation**: `SUM` (累加所有 Action Token 对该像素的关注)

### 2.4 后处理

1.  **重塑与上采样**: $256 \rightarrow 16 \times 16 \rightarrow 480 \times 640$ (双线性插值)。
2.  **鲁棒归一化**: 
    -   保留了 2%-98% 的百分位截断逻辑，以进一步增强对比度，滤除极端的噪点。

---

## 3. 结果分析

通过使用 **Softmax + SUM Aggregation**：
-   **信噪比高**: 背景噪音几乎为零（蓝色），目标物体显著高亮（红色）。
-   **累积效应**: 能够捕捉到多层、多头对同一区域的共同关注。
-   **数值合理**: 聚合后的数值通常在几十到几百之间（取决于关注强度），避免了 Sigmoid 的数千级别底噪。
