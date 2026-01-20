# GR00T 注意力可视化技术文档 (Technical Documentation)

本文档详细说明了 GR00T (VLM-Action Policy) 的交叉注意力 (Cross-Attention) 可视化系统的实现原理。

## 1. 系统架构概览

可视化系统主要由两部分组成：
1.  **核心可视化器 (`attention_visualizer.py`)**：负责拦截模型内部信号，计算注意力权重，生成热力图。
2.  **评估流水线 (`eval_attention.py`)**：负责推理循环、视频帧捕获以及将热力图与相机图像同步。

---

## 2. 核心机制详解 (`attention_visualizer.py`)

### 2.1 Monkey Patching 策略 (拦截机制)
为了在不修改原始库代码的情况下捕获 Cross-Attention 权重，我们采用了 **Monkey Patching** (运行时动态替换) 技术拦截 `diffusers.models.attention.BasicTransformerBlock.forward`。

### 2.2 Token 选择与 QK 计算 (Sigmoid 改进版)

我们复现了 `AttnProcessor2_0` 内部的注意力计算逻辑，并根据 Explainable AI (XAI) 的最佳实践进行了关键改进。

#### 2.2.1 输入张量与 Token 来源
-   **Query ($Q$)**: 来自 Action Head (DiT) 的 Action Tokens (序列长度 49)。
-   **Key ($K$)**: 来自 VLM (Eagle) 的视觉特征 (序列长度 294)。
-   **Vision Token 切片**: 我们提取索引 **`[20 : 276]`**，对应 $16 \times 16$ 的图像 Patch。

#### 2.2.2 计算公式：Sigmoid vs Softmax
为了解决 Softmax 导致的"光斑效应"（Spotlight Issue），我们采用了**双轨制计算**：

1.  **模型推理 (For Model)**:
    $$ W_{model} = \text{Softmax}\left( \frac{Q \cdot K^T}{\sqrt{d}} \right) $$
    *   **作用**: 保证模型推理逻辑不变，维持机器人的正常动作。
    *   **特性**: 竞争性（归一化为1），会导致极值抑制非极值。

2.  **可视化 (For Viz)**: 
    $$ W_{viz} = \text{Sigmoid}\left( \frac{Q \cdot K^T}{\sqrt{d}} \right) $$
    *   **作用**: 生成用于渲染的热力图。
    *   **特性**: **独立性**。每个 Patch 独立打分，互不抑制。
    *   **优势**: 能够同时点亮多个物体（例如同时关注这个杯子和那个盒子），而不是只显示最亮的那一个点。这与 *Attention Guided CAM* 的思想一致。

### 2.3 聚合策略 (Aggregation)

我们对 Sigmoid 输出的 $W_{viz}$ 进行聚合：

#### Step 1: 层聚合 (Layer Aggregation) -> SUM
-   **操作**: 对 8 个 Cross-Attention 层求**和 (Sum)**。
-   **含义**: 累积不同层级 (Layer 0, 2, ..., 14) 对视觉特征的关注总证据量。

#### Step 2: 头聚合 (Head Aggregation) -> SUM
-   **操作**: 对 32 个 Attention Head 求**和 (Sum)**。
-   **含义**: 汇总所有注意力头发现的特征。

#### Step 3: Query 聚合 (Query Aggregation) -> SUM
-   **操作**: 对 49 个 Action Token 求**和 (Sum)**。
-   **含义**: 动作序列整体对图像的总关注度。

### 2.4 后处理

1.  **重塑与上采样**: $256 \rightarrow 16 \times 16 \rightarrow 480 \times 640$ (双线性插值)。
2.  **鲁棒归一化**: 
    -   保留了 5%-95% 的百分位截断逻辑，以进一步增强对比度，滤除极端的噪点。

---

## 3. 结果分析

通过切换到 **Sigmoid**：
-   **光斑消失**: 不再只是几个孤立的高亮像素。
-   **轮廓显现**: 物体的完整形状（如杯身、手掌）能被完整覆盖。
-   **多点关注**: 如果画面中有多个相关物体，它们能同时被点亮，符合人类直觉。
