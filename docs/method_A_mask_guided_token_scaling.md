# 方案 A：Mask-Guided Token Scaling — 完整技术总结

## 1. 核心公式

### 单 mask（仅 pick）

```
X'_i = s_i · X_i
s_i = 1 + a · M_i
```

### 双 mask（pick + place）

```
X'_i = s_i · X_i
s_i = 1 + a_p · M_pick_i + a_l · M_place_i
```

**变量定义：**

| 符号 | 含义 | 形状 | 说明 |
|------|------|------|------|
| X_i | 第 i 个 backbone token | (1536,) | 经过 VLM + LayerNorm + SelfAttention 后的特征 |
| M_i | 第 i 个 vision token 的 mask 覆盖率 | 标量 ∈ [0,1] | 从像素级 mask 通过 avg pooling 得到 |
| a (或 a_p, a_l) | learnable 标量 | 标量 | 初始化为 0（保证初始行为等价 baseline） |
| X'_i | 修改后的 backbone token | (1536,) | 替换原始 token 后送入 DiT |

**关键约束：只作用于 vision tokens，text tokens 不修改。**

**工程稳定性建议：**

```
s_i = clamp(s_i, min=0.1, max=4.0)
```

这样可以避免 scale 异常（例如负值或过大）导致训练不稳定。

---

## 2. 注入位置

在 `FlowmatchingActionHead.process_backbone_output` 之后，DiT forward 之前。

```
pipeline 顺序：

backbone_features (B, S, 1536)        ← 来自 Eagle2 VLM
        ↓
vlln (LayerNorm)                      ← 已有
        ↓
vl_self_attention (SelfAttention)     ← 已有
        ↓
★ Mask Scaling ← 在这里插入         ← 新增
        ↓
DiT cross-attention (12 layers)       ← 已有
```

**建议直接改现有实现（更贴合你当前仓库）：**

1. 在 `FlowmatchingActionHeadConfig` 增加开关（例如 `use_mask_token_scaling`）。
2. 在 `FlowmatchingActionHead.__init__` 增加 `a_p / a_l`。
3. 在 `process_backbone_output` 中执行 scaling。

这样最少改动、最容易和你已有 baseline 对齐。

---

## 3. Mask 从像素到 token 的映射

### 流程

```
像素级 mask (H, W) binary
        ↓
按 tile 切分（Eagle2 支持 1~6 tiles）
        ↓
对每个 tile: adaptive_avg_pool2d → (16, 16)
        ↓
flatten → 每个 tile 产生 256 个值
        ↓
拼接所有 tile → M ∈ [0,1]^{N_vis}
        ↓
填入 backbone sequence 的 vision token 位置
（text token 位置填 0）
```

### 为什么是 16×16

在你当前这套 GR00T/Eagle2 管线里，每个 tile 对应 `tokens_per_tile=256`，
即每个 tile 的视觉 token 网格等价为 `16×16`。
所以 mask 应降采样到 `16×16` 才能对齐到每个 vision token。

### M_i 的值是连续的

- M_i = 1.0：这个 patch 完全被 mask 覆盖
- M_i = 0.0：这个 patch 完全不在 mask 内
- M_i = 0.3：这个 patch 有 30% 面积在 mask 内（通常是物体边缘）

---

## 4. 方案 A 对 DiT Cross-Attention 的影响机制

### DiT 结构回顾

```
DiT 输入：
  Q 端：sa_embs = concat(state_features, future_tokens, action_features)
  KV 端：backbone_features ← 就是你修改的 X'

每个 DiT cross-attention block：
  attention_weight(j,i) = softmax( Q_j · K_i / sqrt(d) )
  output_j = Σ_i attention_weight(j,i) · V_i
```

### 两条影响路径

**路径 1：K 方向 — 改变 attention 分布**

```
K_i = W_K · X'_i = (1 + a·M_i) · W_K · X_i = (1 + a·M_i) · K_i_orig

score(j,i) = Q_j · K_i = (1 + a·M_i) · Q_j · K_i_orig
```

mask token (M_i>0) 的 score 被等比放大 → softmax 后获得更高 weight

注意：
- 原始 score 为正 → 放大后更正 → attention 增强 ✅
- 原始 score 为负 → 放大后更负 → attention 反而降低 ⚠️
- 原始 score 为零 → 放大后仍为零 → 无效 ⚠️

**路径 2：V 方向 — 改变输出值强度**

```
V_i = W_V · X'_i = (1 + a·M_i) · V_i_orig

output_j = Σ_i weight(j,i) · (1 + a·M_i) · V_i_orig
```

即使 attention weight 不变，mask token 贡献的 value 也被放大了。

### Phase-dependent 效果分析

| 阶段 | mask 状态 | 增益 (1+a·M) | 效果 |
|------|----------|-------------|------|
| Approaching | M≈1，完整覆盖 | 最大，≈1+a | 物体 token 强增强 |
| 接近 | M≈1，但手开始进入画面 | 仍然较大 | 物体增强，手不受影响 |
| 即将抓取 | M 局部下降（手遮挡物体） | 逐渐降低 | 增强自然减弱 |
| 抓取中 | 大部分 M→0 | ≈1，基本无增强 | 约等于原始 GR00T |
| 放置 | place mask M≈1 | 增强转移到 place 目标 | place token 被增强 |

**核心特性：mask 的增强会随着画面自然消退，不需要额外设计 phase-aware 逻辑。**

---

## 5. 与已有 Outline + 语言指令的协同

```
你已有的系统：
  ┌─ 图像层面：RGB 上画 outline（不填充不变暗）
  │    → SigLIP 天然对边缘敏感，encoder 就会对 outline 区域产生不同的 token
  │
  ├─ 语言层面："Pick up the outlined cup"
  │    → LLM 把 "outlined" 语义绑定到 vision token
  │
  └─ 新增 token 层面：方案 A 的 mask scaling
       → 在 VLM 产出的 token 上进一步增强 mask 区域

三层叠加，各管不同层次：
  SigLIP：让 encoder 看到 outline
  LLM：让语义理解 "outlined" 指什么
  方案 A：让 DiT 的 cross-attention 更关注这些 token
```

---

## 6. 参数与初始化

| 参数 | 数量 | 初始化 | 说明 |
|------|------|--------|------|
| a（单 mask） | 1 | 0 | 训练开始时模型行为 = 原始 GR00T |
| a_p + a_l（双 mask） | 2 | 都是 0 | pick 和 place 各自学增益 |

**训练开始时 a=0 → (1+0·M)=1 → X'=X → 完全等价原始 GR00T。** 不可能比原来更差。

---

## 7. 局限性（诚实评估）

1. **所有维度统一缩放**：1536 个维度缩放同一个倍数，所有 attention head 的反应一样
2. **对负 score 有反效果**：如果某个 phase 的 Q 本来就不关注物体 token，缩放会让它更不关注
3. **a 是全阶段共享的**：approaching 和 grasping 用同一个 a，学到的是折衷值
4. **只能加强已有偏好，不能创造新偏好**：不改变 token 在 K 空间的方向

这些局限是方案 B 可以解决的 → 如果 A 效果不够好，B 是升级方向。

---

## 8. 伪代码实现

```python
class FlowmatchingActionHead(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.use_mask_token_scaling = getattr(config, "use_mask_token_scaling", False)
        self.mask_scale_pick = nn.Parameter(torch.zeros(1))   # a_p
        self.mask_scale_place = nn.Parameter(torch.zeros(1))  # a_l
    
    def process_backbone_output(self, backbone_output, action_input=None):
        # 先跑原始的 LayerNorm + SelfAttention
        features = backbone_output["backbone_features"]
        features = self.vlln(features)
        features = self.vl_self_attention(features)
        
        if self.use_mask_token_scaling and action_input is not None:
            # 约定由 processor 提供，形状 (B, S)，text token 位置必须为 0
            mask_pick = action_input.get("mask_pick_tokens", None)
            mask_place = action_input.get("mask_place_tokens", None)

            if mask_pick is not None:
                scale = 1.0 + self.mask_scale_pick * mask_pick.unsqueeze(-1)
                if mask_place is not None:
                    scale = scale + self.mask_scale_place * mask_place.unsqueeze(-1)
                scale = torch.clamp(scale, min=0.1, max=4.0)
                features = features * scale
        
        backbone_output["backbone_features"] = features  # (B, S, 1536)
        return backbone_output
```

---

## 9. 实验预期

| 指标 | 预期表现 |
|------|---------|
| 训练收敛速度 | 可能加快（mask 减少了 attention 的搜索空间） |
| Seen 物体成功率 | 持平或小幅提升 |
| Unseen 物体成功率 | 可能提升（outline + mask 组合的泛化能力） |
| 抓取精度 | 不应退化（mask 自然消退 + a 是端到端学的） |
| 训练后 a 的值 | 预计 0.1~2.0 之间，取决于 mask 信号的有用程度 |

---

## 10. A0 / A1 说明（你当前怎么做）

- A0：不开启 mask scaling（纯 baseline）。
- A1：开启 mask scaling（本文件方案 A）。

你已经有 baseline，可直接实现 A1，不需要再额外做 A0 实现。

---

## 11. 代码核对结论（Pick-only，先阅读不改实现）

本节是针对你当前问题的“代码事实版”结论，只回答：
1. mask 应该怎么 patch（16x16 的行列方向）  
2. VLM token 序列到底什么结构  
3. 2048 -> 1536 后应该注入在哪里  

### 11.1 输入分辨率与 resize 路径

你当前训练配置里，视觉输入特征仍是 `3x480x640`（见训练 checkpoint 的 `config.json`）。
但进入 Eagle 处理器后，会走到 `224x224` 的 tile：

- `processor_groot.py` 固定了 `images_kwargs={"min_dynamic_tiles":1,"max_dynamic_tiles":1,"use_thumbnail":False}`，即单 tile（`unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py:515`）。
- Eagle 处理器配置中 `size.height/width=224`、`tokens_per_tile=256`（`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/preprocessor_config.json:35`）。
- 图像在 `_get_image_patches` 里被 resize 到目标 tile 大小（`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py:285`）。

结论：你说的“RGB/mask 原始是 480x640，后面到 224x224”在这条链路下是成立的。

### 11.2 mask patch 应该怎么做（行列顺序）

在当前配置下：

- `force_image_size=224`，`patch_size=14`（`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json:13`, `:73`）
- `use_pixel_shuffle=false`（同文件 `:59`）

所以每张图像视觉 patch 数就是：

`(224 / 14) * (224 / 14) = 16 * 16 = 256`

也就是说，pick mask 直接对齐到 `16x16` 即可，不需要 13x13。

推荐 patch 映射规则（pick-only）：

1. 把 mask 按图像同样几何流程映射到 224x224（和 RGB 保持同一坐标系）。
2. 对 224x224 mask 做 `16x16` 平均池化，得到每个 patch 的覆盖率 `M ∈ [0,1]^{16x16}`。
3. 按 row-major flatten（`index = row * 16 + col`）得到 `M_flat ∈ [0,1]^{256}`。

行列方向依据：

- 你自己的 token 审计文档和可视化脚本都按 16x16 row-major 使用（`Groot_Analysis/Groot_VLM_Analysis.md`，`Attention_Visualization/attention_visualizer.py:231`）。
- Eagle 的 tile 切分索引也是先列后行推进，整体等价 row-major 组织（`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py:289`）。

### 11.3 VLM token 序列结构（294 不是固定常数）

你说的“system prompt + <img> + vision token + </img> + text token”方向是对的。
代码事实：

- chat template 默认会自动补 system prompt（`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/chat_template.json:2`）。
- 图像占位符在 processor 中被替换成  
  `<image n><img><IMG_CONTEXT> * (num_tiles * tokens_per_tile)</img>`  
  （`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py:226`）。
- 当前 `tokens_per_tile=256`，单 tile 即 256 个 `<IMG_CONTEXT>`。

`inspect_groot_tokens.py` 的样例确实得到总长度约 294，但这会随文本长度变化。  
所以实现时不要硬编码 `[20:276]`，应动态用 `<img>` / `</img>` 或 `image_token_index` 定位。

### 11.4 2048 -> 1536 路径与注入点

当前真实路径：

1. Eagle/Qwen hidden states: `[..., 2048]`（`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json:36`）
2. `eagle_linear: 2048 -> 1536`（`unitree_lerobot/lerobot/src/lerobot/policies/groot/groot_n1.py:88`，前向在 `:144`）
3. Action head 中 `vlln + vl_self_attention`（`unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py:259`）
4. 再进入 DiT cross-attention K/V（`unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py:326`）

你问“到底注入在哪里”：  
**对 A1（pick-only token scaling），最稳的注入点是第 3 步后、第 4 步前**，也就是 `process_backbone_output` 后的 `backbone_features (B,S,1536)`。

原因：

- 正好是“VLM 输出到 DiT 前”；
- 不改 Eagle 主干，不破坏 2048 语言空间；
- 直接作用于 DiT 的 K/V 输入，目标最明确。

### 11.5 一条很关键的工程约束

`GrootPolicy` 在 forward/predict 时会过滤输入键，只保留：

- 基础键：`state/state_mask/action/action_mask/embodiment_id`
- 或前缀 `eagle_` 的键  
（`unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py:101`, `:134`）

所以如果后续传 pick mask token，请使用 `eagle_` 前缀（例如 `eagle_mask_pick_tokens`），
否则会在 policy wrapper 被直接丢掉。

---

## 12. 离线实测审计（你当前可用 policy + dataset）

这一节是你要求的“先跑出来再说”，使用你刚确认的本地资源：

- policy: `/home/g1/humanoid_gaze/unitree_lerobot/lerobot/outputs/train/groot_pick_crumpled_paper_ball_outlined20000`
- dataset: `ZUO66/pick_crumpled_paper_ball_outline1`
- dataset root: `/home/g1/.cache/huggingface/lerobot/ZUO66/pick_crumpled_paper_ball_outline1`
- tokenizer local path: `/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5`

说明：

- 这次是离线运行（`HF_HUB_OFFLINE=1`，`TRANSFORMERS_OFFLINE=1`）。
- 只做了运行时 override（`device_processor.device=cpu`），没有修改任何 checkpoint 的 json 文件。

### 12.1 实测 token id

实测得到：

- `<IMG_CONTEXT>` id = `151669`
- `<img>` id = `151670`
- `</img>` id = `151671`

对应关系与模板一致：`<img>` 后面紧跟一段 `<IMG_CONTEXT>`，最后由 `</img>` 结束。

### 12.2 实测索引与长度（真实样本）

对样本 `0/1/2`（task 相同）：

- `len(tokens) = 293`
- `idx_start(<img>) = 19`
- `idx_end(</img>) = 276`
- `inside_img = idx_end - idx_start - 1 = 256`
- `<IMG_CONTEXT>` 计数 = `256`

关键结论：视觉 token 数确实是 256（单图单 tile 16x16）。

### 12.3 文本长度变化实验（同一帧图像）

在同一图像/状态下，只改 task 文本：

- base: `len=293`
- `+ quickly`: `len=295`
- `+ and keep the wrist stable ...`: `len=305`

同时观察到：

- `idx_start(<img>)` 始终 `19`
- `idx_end(</img>)` 始终 `276`
- `<IMG_CONTEXT>` 始终 `256`

解释：

- 这套 chat template 是“先 image，再 task 文本”，所以文本变长会改总长度，但不会改当前这版模板下的图像段位置。
- 但是工程实现仍然必须动态定位，不能写死 `[20:276]`。因为模板、相机数量、tile 数、特殊 token 规则一变，硬编码就会错。

### 12.4 给 A1 的最终实现约束（基于实测）

1. 绝不硬编码视觉区间常数切片。  
2. 运行时每个 batch 动态找 `<img>` 与 `</img>`，或用 `image_token_index` 统计 `<IMG_CONTEXT>` 位置。  
3. 只对视觉 token 对应位置注入 mask scaling；文本 token 必须保持不变。  
4. 如果未来改成多图/多 tile，视觉 token 数要按实际计数，不再默认 256。

### 12.5 最小可用定位逻辑（伪代码）

```python
tokens = tokenizer.convert_ids_to_tokens(input_ids)
idx_start = tokens.index("<img>")
idx_end = tokens.index("</img>")
vision_slice = slice(idx_start + 1, idx_end)  # 当前样本真实视觉 token 段

# 或者更稳：按 image token id 定位（不依赖字符串反解）
img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
vision_pos = (input_ids == img_ctx_id)  # bool mask
```

这两种都满足“动态定位”；第二种在工程上更稳（避免 tokenizer 字符串细节差异）。

---

## 13. A1 实现落地点（当前代码库）

下面是已经落到代码里的 A1（pick-only）实现路径，便于你直接对照检查。

### 13.1 Processor 侧（mask -> token mask）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py`

已实现：

1. `GrootPackInputsStep` 支持可选输入键 `observation.mask_pick`。  
2. mask 支持形状 `(H,W)` / `(B,H,W)` / `(B,1,H,W)`，统一转为 `(B,H,W)` 并归一化到 `[0,1]`。  
3. `GrootEagleCollateStep` 在拿到 `eagle_input_ids` 后，动态按 `<IMG_CONTEXT>` 位置构建 `eagle_mask_pick_tokens`（`B,S`）。  
4. text token 位置固定为 0，只在 vision token 位置填入 mask 覆盖率。  

实现细节：

- resize 到 `224x224`（与 Eagle 图像尺寸一致）；
- 再做 `16x16` 平均池化；
- flatten（row-major）得到 256 个 patch 值；
- 按视觉 token 数量重复/截断填充到全部 vision token 位置（兼容未来多图/多 tile）。

### 13.2 Action Head 侧（DiT 前缩放）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py`

已实现：

1. 新增配置项：
   - `use_mask_token_scaling`（默认 `False`）
   - `mask_pick_token_key`（默认 `eagle_mask_pick_tokens`）
   - `mask_pick_scale_init`（默认 `0.0`）
   - `mask_scale_min` / `mask_scale_max`（默认 `0.1 / 4.0`）
2. 新增可学习标量参数：`mask_pick_scale`。  
3. 在 `process_backbone_output` 中，`vlln + vl_self_attention` 后执行：

`X' = X * clamp(1 + a * M, min=0.1, max=4.0)`

其中 `M` 来自 `eagle_mask_pick_tokens`。

4. 训练 `forward` 和推理 `get_action` 都走同一注入路径。

### 13.3 Eval SAM 对接（可选）

文件：`unitree_lerobot/eval_robot/eval_g1_sam.py`

已实现：

- 当 SAM 返回 `mask` 时，写入 `observation["observation.mask_pick"]`，可直接被 preprocessor 消费。

### 13.4 默认行为安全性

默认 `use_mask_token_scaling=true`，但只有当输入里存在 `observation.mask_pick`
并且 processor 成功生成 `eagle_mask_pick_tokens` 时，A1 才会实际生效。  
如果没有 mask 输入，行为与旧流程一致。  
如需强制关闭，可把 `use_mask_token_scaling` 设为 `false`。

CASE B: sample0 longer task
task: 'Put the outlined paper ball into the outlined bin. and keep the wrist stable while approaching from the right side'
total_len=305, img_start=19, img_end=276, omitted_img_ctx=256
000 | <|im_start|>         | TEXT
001 | system               | TEXT
002 | Ċ                    | TEXT
003 | You                  | TEXT
004 | Ġare                 | TEXT
005 | Ġa                   | TEXT
006 | Ġhelpful             | TEXT
007 | Ġassistant           | TEXT
008 | .                    | TEXT
009 | <|im_end|>           | TEXT
010 | Ċ                    | TEXT
011 | <|im_start|>         | TEXT
012 | user                 | TEXT
013 | Ċ                    | TEXT
014 | <                    | TEXT
015 | image                | TEXT
016 | Ġ                    | TEXT
017 | 1                    | TEXT
018 | >                    | TEXT
019 | <img>                | IMG_START
020 | ... [OMITTED 256 VISION TOKENS: <IMG_CONTEXT> x 256] ...
276 | </img>               | IMG_END
277 | ['                   | TEXT
278 | Put                  | TEXT
279 | Ġthe                 | TEXT
280 | Ġoutlined            | TEXT
281 | Ġpaper               | TEXT
282 | Ġball                | TEXT
283 | Ġinto                | TEXT
284 | Ġthe                 | TEXT
285 | Ġoutlined            | TEXT
286 | Ġbin                 | TEXT
287 | .                    | TEXT
288 | Ġand                 | TEXT
289 | Ġkeep                | TEXT
290 | Ġthe                 | TEXT
291 | Ġwrist               | TEXT
292 | Ġstable              | TEXT
293 | Ġwhile               | TEXT
294 | Ġapproaching         | TEXT
295 | Ġfrom                | TEXT
296 | Ġthe                 | TEXT
297 | Ġright               | TEXT
298 | Ġside                | TEXT
299 | ']                   | TEXT
300 | <|im_end|>           | TEXT
301 | Ċ                    | TEXT
302 | <|im_start|>         | TEXT
303 | assistant            | TEXT
304 | Ċ                    | TEXT
