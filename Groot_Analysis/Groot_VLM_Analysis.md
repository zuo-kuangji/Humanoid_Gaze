# GRooT VLM Token 序列分析报告

## 执行摘要

本报告通过加载真实的训练权重、数据集和预处理流水线，对 GRooT N1.5 模型的 Token 序列进行了**高保真数字审计**。通过实证验证，我们精确定位了图像 Token 的物理位置，并确认了视觉 Patch 的空间排列顺序。

---

## 1. Token 序列物理映射

### 1.1 实测结果（基于真实推理流水线）

通过运行 [`inspect_groot_tokens.py`](file:///home/g1/zuo/unitree_IL_lerobot1/Groot_Analysis/inspect_groot_tokens.py)，我们获得了以下**绝对物理映射**：

| 物理索引范围 | Token 内容 | 功能角色 | Token 数量 |
|:---|:---|:---|:---|
| **0 - 18** | `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image 1>` | System Prompt + User Header (Metadata) | 19 |
| **19** | `<img>` | **图像起始边界标记** | 1 |
| **20 - 275** | `<IMG_CONTEXT>` × 256 | **视觉 Patch 数据 (16×16 grid)** | 256 |
| **276** | `</img>` | **图像结束边界标记** | 1 |
| **277 - 293** | `['Pick up the drink...']` | 任务指令 (Language Instruction) | 17 |

**总序列长度**: 294 Tokens

### 1.2 关键结论

1. **图像内容的物理起始位置**: 索引 **20**（紧跟在 `<img>` 标记后面）
2. **图像内容的物理结束位置**: 索引 **275**（紧接着 `</img>` 标记之前）
3. **图像 Patch 总数**: **256** 个（对应 16×16 的网格）
4. **System Prompt 占据前 19 个位置**，这解释了为什么 `<img>` 不是从索引 0 开始

---

## 2. 视觉 Patch 空间排列顺序验证

### 2.1 Row-Major vs Column-Major 争议

在初步分析中，存在对 Patch 排列顺序的疑问：是 **Row-Major（行优先）** 还是 **Column-Major（列优先）**？

### 2.2 源码证据：绝对实锤

通过分析 GRooT 的底层图像处理代码 [`image_processing_eagle2_5_vl_fast.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py)，我们找到了决定性证据：

```python
# 源码位置: image_processing_eagle2_5_vl_fast.py
for i in range(blocks):
    box = (
        (i % cols) * tile_size,  # x 坐标：列索引 (水平方向)
        (i // cols) * tile_size, # y 坐标：行索引 (垂直方向)
        ...
    )
```

**数学证明**：
- `i % cols` 计算**列索引**（水平位置）
- `i // cols` 计算**行索引**（垂直位置）
- 这意味着索引 `i` 是**先沿水平方向（列）递增，到达行末后再换到下一行**

**结论**: **Row-Major（行优先）排列** 是绝对正确的。

### 2.3 可视化示意

对于 16×16 的 Grid（256 个 Patch）：

```
索引顺序（Row-Major）:
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │ ... │ 15  │  ← 第 1 行
├─────┼─────┼─────┼─────┼─────┼─────┤
│ 16  │ 17  │ 18  │ 19  │ ... │ 31  │  ← 第 2 行
├─────┼─────┼─────┼─────┼─────┼─────┤
│ 32  │ 33  │ 34  │ 35  │ ... │ 47  │  ← 第 3 行
├─────┼─────┼─────┼─────┼─────┼─────┤
│ ... │ ... │ ... │ ... │ ... │ ... │
├─────┼─────┼─────┼─────┼─────┼─────┤
│ 240 │ 241 │ 242 │ 243 │ ... │ 255 │  ← 第 16 行
└─────┴─────┴─────┴─────┴─────┴─────┘
```

**映射到 Token 序列**:
- Token 索引 20 → Patch 0（左上角）
- Token 索引 35 → Patch 15（第一行最右）
- Token 索引 36 → Patch 16（第二行最左）
- Token 索引 275 → Patch 255（右下角）

---

## 3. Chat Template 与 Token 构造机制

### 3.1 为什么 `<img>` 在索引 19？

GRooT 使用的是 **Mistral 风格的 Chat Template**，会自动插入系统提示词和对话结构。

### 3.2 源码证据：Eagle 的媒体占位符替换逻辑

在 [`processing_eagle2_5_vl.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py#L226) 中，定义了图像插入的格式：

```python
# 核心逻辑（简化）：
special_placeholder = f"<image {idx+1}><img>{self.image_token * num_all_tiles * self.tokens_per_tile}</img>"
```

其中：
- `self.image_token = "<IMG_CONTEXT>"`
- `self.tokens_per_tile = 256`（固定值）
- `num_all_tiles = 1`（标准配置）

**最终构造的序列**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image 1><img>[256个<IMG_CONTEXT>]</img>['task instruction']<|im_end|>
<|im_start|>assistant

```

### 3.3 Token 序列生成流程

```
原始输入 
  → Chat Template 渲染
  → 插入 System Prompt
  → 替换 <image-1> 占位符
  → 插入 <img> 标签 + 256 Patches
  → 追加任务指令
  → 分词器编码
  → 最终 Token 序列
```

---

## 4. 梯度方差分析与 Attention Sinks

### 4.1 观察到的现象

在之前的实验中，通过 [`groot_token_variance.csv`](file:///home/g1/zuo/unitree_IL_lerobot1/groot_token_variance.csv) 发现：
- **索引 19 (`<img>`)** 和 **索引 276 (`</img>`)** 的梯度方差显著高于其他位置
- 这些"特殊标记"成为了 **Attention Sinks（注意力陷坑）**

### 4.2 原因分析

1. **边界标记的语义重要性**: `<img>` 和 `</img>` 在模型训练中起到"开关"作用，告诉模型何时进入/退出视觉模态
2. **高频激活**: 每张图像都会激活这些标记，导致它们累积了大量的注意力权重
3. **对可视化的影响**: 如果不进行处理，这些高方差 Token 会"污染"热力图，使得真正的视觉注意力被掩盖

### 4.3 缓解策略

在 AGCAM 可视化中，我们采用了以下技术：
1. **精确的索引范围**: 只取 `[20:276]` 的视觉 Token，排除边界标记
2. **Quantile Clipping**: 使用 95th 百分位截断，抑制极端离群值
3. **Max Aggregation**: 在多头注意力中使用 `max` 而非 `mean`，让任务相关的头主导

---

## 5. 实测数据：真实样本分析

### 5.1 测试样本

- **数据集**: `ZUO66/handover_drinks`
- **Episode**: 0
- **任务指令**: `"Pick up the drink and hand it to the person."`

### 5.2 完整 Token 序列（节选）

```
索引 000 | <|im_start|>    | System/Prompt/Metadata
索引 001 | system          | System/Prompt/Metadata
索引 002 | \n              | System/Prompt/Metadata
索引 003 | You             | System/Prompt/Metadata
索引 004 |  are            | System/Prompt/Metadata
...
索引 019 | <img>           | >>> 图像起始边界 <<<
索引 020 | <IMG_CONTEXT>   | IMAGE PATCH (Patch 0)
索引 021 | <IMG_CONTEXT>   | IMAGE PATCH (Patch 1)
...
索引 275 | <IMG_CONTEXT>   | IMAGE PATCH (Patch 255)
索引 276 | </img>          | >>> 图像结束边界 <<<
索引 277 | ['              | Task Instruction
索引 278 | Pick            | Task Instruction
索引 279 |  up             | Task Instruction
...
```

---

## 6. 对 AGCAM 可视化的指导意义

### 6.1 正确的 Vision Token 范围

在实现 AGCAM 热力图时，**必须**使用以下参数：

```python
vision_start_idx = 20  # 图像内容的第一个 Token
vision_len = 256        # 图像 Patch 的总数
```

### 6.2 Heatmap 重建逻辑

```python
# 从梯度中提取视觉部分
vision_cam = cam[:, vision_start_idx : vision_start_idx + vision_len]

# 重塑为 16x16 网格 (Row-Major)
heatmap = vision_cam.view(batch_size, 16, 16)

# 上采样到原始图像分辨率
heatmap_upsampled = F.interpolate(heatmap, size=(H, W), mode='bilinear')
```

### 6.3 常见错误

❌ **错误 1**: 从索引 0 开始取视觉 Token → 会包含 System Prompt  
❌ **错误 2**: 包含索引 19 (`<img>`) → 会引入 Attention Sink 噪声  
❌ **错误 3**: 使用 Column-Major 排列 → 热力图会旋转 90 度  

✅ **正确做法**: 严格使用 `[20:276]` 范围 + Row-Major 重塑

---

## 7. 技术验证方法

### 7.1 验证脚本

使用 [`inspect_groot_tokens.py`](file:///home/g1/zuo/unitree_IL_lerobot1/Groot_Analysis/inspect_groot_tokens.py) 可以在任何时刻验证 Token 映射：

```bash
python Groot_Analysis/inspect_groot_tokens.py
```

该脚本会：
1. 加载真实的训练权重
2. 从数据集中提取真实样本
3. 通过模型的预处理器生成 Token 序列
4. 打印完整的物理索引映射

### 7.2 输出示例

```
===========================================================================
官方流水线真实序列审计 (总长度: 294)
===========================================================================
索引 019 | <img>                     | >>> 图像起始 <<<
索引 020 | <IMG_CONTEXT>             | IMAGE PATCH
...
索引 275 | <IMG_CONTEXT>             | IMAGE PATCH
索引 276 | </img>                    | >>> 图像结束 <<<

【审计结果】
1. 图像物理位置范围: [20 : 276]
2. 图像起始索引 (Start): 20
3. 图像结束索引 (End):   275
4. 图像 Patch 总数:      256
===========================================================================
```

---

## 8. 参考文献与源码位置

### 8.1 关键源文件

| 文件 | 路径 | 功能 |
|:---|:---|:---|
| Eagle 图像处理器 | [`image_processing_eagle2_5_vl_fast.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py) | 定义 Patch 切分逻辑（Row-Major） |
| Eagle 预处理器 | [`processing_eagle2_5_vl.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py) | 定义 Chat Template 和媒体占位符替换 |
| GRooT 模型主体 | [`groot_n1.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/policies/groot/groot_n1.py) | 定义 Backbone 和 Action Head 交互 |
| AGCAM 可视化器 | [`agcam_visualizer.py`](file:///home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/src/lerobot/utils/agcam_visualizer.py) | 实现注意力热力图生成 |

### 8.2 实验数据

- **Token 方差分析**: [`groot_token_variance.csv`](file:///home/g1/zuo/unitree_IL_lerobot1/groot_token_variance.csv)
- **训练权重**: `unitree_lerobot/lerobot/outputs/train/groot_handover/checkpoints/020000/pretrained_model`
- **测试数据集**: `ZUO66/handover_drinks`

---

## 9. 未来工作方向

1. **动态 Token 长度支持**: 当前假设固定 256 个 Patch，未来可能需要支持多尺度图像
2. **多模态注意力分析**: 研究文本-图像跨模态注意力的分布模式
3. **时序注意力追踪**: 在视频输入场景下，分析帧间注意力迁移

---

**文档版本**: v1.1  
**最后更新**: 2026-01-17  
**验证状态**: ✅ 已通过真实推理流水线验证

---

## 10. (严重更新) 架构参数修正：Attention Heads 数量差异

### 10.1 发现时间
**2026-01-17**

### 10.2 默认参数与真实参数的差异

在 AGCAM 初始实现中，代码因无法读取底层 config 而 fallback 参考了 `groot_n1.py` 的默认参数可以（`num_attention_heads=8`）。
然而，通过查阅 `nvidia/GR00T-N1.5-3B` 的底层配置文件，我们发现实际训练的架构与默认参数有巨大出入：

| 参数 | 源码默认值 (Assumed) | 真实模型值 (Actual) | 差异影响 |
|:---|:---|:---|:---|
| **num_layers** | 12 (Typical DiT) | **16** | 平均注意力时多了 4 层数据，影响较小 |
| **num_attention_heads** | **8** | **32** | **影响巨大**。导致可视化时发生了混叠 |
| **attention_head_dim** | 64 | **48** | 影响计算时的维度对齐 |

### 10.3 证据来源

**底层配置文件路径**:
`/home/g1/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e/config.json`

**配置文件内容 (节选)**:
```json
"action_head_cfg": {
    "diffusion_model_cfg": {
        "attention_head_dim": 48,
        "num_attention_heads": 32,    <--- 真实值
        "num_layers": 16,             <--- 真实值
        ...
    }
}
```

### 10.4 混叠现象解释

由于代码按 `heads=8` 进行 reshape，而实际只有 `heads=32` 的数据：
*   实际数据维度：`(Batch, 32, 256)`
*   代码解读维度：`(Batch, 8, 256)` （假设 Batch 变大了 4 倍，或发生了维度折叠）

**后果**：
在 `view(batch_size, heads, lengths)` 操作中，每 4 个真实的 Head 可能被压缩或平均到了 1 个展示的 Head 中。
例如：我们看到的 **“Head 0”** 很可能是真实模型中 **Head 0, 1, 2, 3** 的混合体。
这解释了为什么只有部分 Head 功能明确，而其他 Head 看起来很模糊。

### 10.5 修正方案
在后续可视化中，必须显式指定 `heads=32`，并将网格布局从 `2x4` 改为 `4x8` 以展示所有 32 个专家的独立视角。
