# GR00T N1.5 Mask-Derived Position Token 注入实施指南

本文档面向你当前的目标：在 GR00T N1.5 的 VLM 序列中增加 mask-derived 位置 token（目标物、放置物），并比较两种注入位置：

1. `vision` 末尾（即 `</img>` 前）
2. `</img>` 后（推荐优先）

同时给出代码库中真实的序列构建逻辑、稳定性风险点、最小改动实现方案、验证与回滚策略。

---

## 1. 先说结论

1. 当前 `groot` 主干里，视觉序列不是通过 `torch.cat(vision_feats, text_feats)` 拼出来的，而是先在文本序列中放 `<IMG_CONTEXT>` 占位 token，再在前向里把这些位置的 embedding 替换成视觉特征。
2. 你现有的 `70% RGB + 30% overlay mask` 方案和这条链路兼容，不冲突。
3. 新增两个位置 token（目标物、放置物）时，建议先做 `</img>` 后注入，原因是对预训练视觉块结构干扰最小。
4. 当前 action head 存在一个稳定性隐患：`encoder_attention_mask` 实际没有在 DiT cross-attention 里生效（代码中被置为 `None`）。这个问题建议一并修复。

---

## 2. 你现在这套代码的真实序列构建方式

### 2.1 文本侧占位构建（Processor）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py`

关键逻辑在 `replace_media_placeholder()`：

- `tokens_per_tile=256`（初始化参数，见该文件约 138 行）
- 图像占位被替换为：
  - `<image N><img>{<IMG_CONTEXT> * num_all_tiles * 256}</img>`
  - 关键代码约在 `:226`

这一步只是构造文本 token 序列，不做视觉 embedding 拼接。

### 2.2 训练流程里通常固定为单 tile（256）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py`

- `collate()` 里调用 processor 时固定：
  - `images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False}`
  - 约在 `:518`

所以训练常见是 1 tile，对应 256 个 `<IMG_CONTEXT>` 占位。

### 2.3 前向替换（Model）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py`

关键步骤：

1. 先取文本 embedding：
   - `input_embeds = self.language_model.get_input_embeddings()(input_ids)`（约 `:225`）
2. 视觉分支提取 `vit_embeds`（约 `:227`）
3. 用 `input_ids == image_token_index` 找到 `<IMG_CONTEXT>` 的所有位置（约 `:237`）
4. 用视觉特征覆盖这些位置 embedding（约 `:239`）

这就是你需要记住的核心事实：**视觉 token 是“替换”进入序列，不是拼接进入序列**。

---

## 3. Special tokens 与 token_id（本地缓存实测）

文件：`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/added_tokens.json`

- `<IMG_CONTEXT>`: `151669`
- `<img>`: `151670`
- `</img>`: `151671`

文件：`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json`

- `image_token_index = 151669`

这与模型代码中的 `self.image_token_index` 使用方式一致。

---

## 4. 为什么你会看到 20/276，以及为什么不要硬编码它

你在分析文档和可视化脚本中看到的 `20~275`、`[20:276]` 是某个模板+某段指令下的“物理位置结果”，不是模型硬编码约束。

- 分析文档中有该假设：`Groot_Analysis/Groot_VLM_Analysis.md`
- 可视化脚本也写死了该范围：`Attention_Visualization/attention_visualizer.py`

在主干 `groot` 模型代码里，没有发现 `vision_start_idx=20` / `vision_end_idx=276` 这类硬编码切片。

因此新增 token 后，固定索引策略会失效。后续你自己的分析脚本应改为“按 token_id 动态定位”。

---

## 5. 注入位置对比：`</img>` 后 vs `vision` 末尾

### 5.1 `</img>` 后（推荐先做）

序列形态：

`<img> [256*IMG_CONTEXT] </img> <OBJ_POS> <PLACE_POS> [task text]`

优点：

1. 不破坏 `<img> ... </img>` 的视觉块边界结构
2. 对已有视觉预训练分布偏移较小
3. 你原有 overlay mask 方案可直接共存

代价：

1. 位置 token 不在视觉块内部，归纳偏置偏“语言条件”

### 5.2 `vision` 末尾（`</img>` 前）

序列形态：

`<img> [254*IMG_CONTEXT] <OBJ_POS> <PLACE_POS> </img> [task text]`

优点：

1. token 更像视觉块内元素，可能增强局部空间建模

风险：

1. 直接改动视觉块分布，OOD 风险更大
2. 容易扰动你已验证有效的 overlay 输入收益

建议策略：

1. 第一阶段只做 `</img>` 后注入
2. 第二阶段再做 `</img>` 前对照实验

---

## 6. 你要改哪些文件（最小可行改动集）

## 6.1 Processor 层：插入占位 token 文本

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py`

目标：

1. 在图像占位串后追加你的两个位置占位符（可配置注入位置）
2. 例如增加配置项：
   - `position_token_mode`: `none | after_img_end | before_img_end`
   - `obj_pos_token`: `"<OBJ_POS>"`
   - `place_pos_token`: `"<PLACE_POS>"`

实现点：

1. 目前 `special_placeholder` 在约 `:226` 直接拼接 `<img>... </img>`
2. 在这里根据 mode 改写模板字符串

建议伪代码：

```python
base_visual = f"{self.image_start_token}{self.image_token * n}{self.image_end_token}"
pos_pair = f"{self.obj_pos_token}{self.place_pos_token}"

if self.position_token_mode == "after_img_end":
    visual_block = base_visual + pos_pair
elif self.position_token_mode == "before_img_end":
    visual_block = (
        f"{self.image_start_token}"
        f"{self.image_token * (n - 2)}"
        f"{self.obj_pos_token}{self.place_pos_token}"
        f"{self.image_end_token}"
    )
else:
    visual_block = base_visual
```

备注：

1. `before_img_end` 方案要保证长度管理正确，避免和视觉 token 数不对齐。
2. 建议先不动 `IMG_CONTEXT` 数量，先做 `after_img_end`。

## 6.2 Tokenizer 资产：新增 special tokens

相关资产目录：

`/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/`

需要保证：

1. `added_tokens.json` 增加 `<OBJ_POS>`、`<PLACE_POS>`
2. `special_tokens_map.json` / `tokenizer_config.json` 的 `additional_special_tokens` 包含这两个 token
3. 模型词表大小和 embedding resize 一致

注意：

1. 如果不更新 tokenizer 并 resize embedding，新 token 会落到未知路径，训练不稳定。

## 6.3 VLM 前向层：把位置 token 的 embedding 替换为 mask-derived 向量

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py`

你需要做的事情和 `<IMG_CONTEXT>` 替换是同构的：

1. 在 `forward()` 增加两个可选输入：
   - `obj_pos_embeds: [B, D]`
   - `place_pos_embeds: [B, D]`
2. 找到 `input_ids == obj_pos_token_index` 和 `input_ids == place_pos_token_index`
3. 用对应 embedding 覆盖这些位置

备注：

1. `D` 必须等于语言模型输入 embedding 维度（当前 Eagle/Qwen3 配置通常是 2048）。
2. 你可以在模型中增加一个小 MLP，把 mask 几何特征映射到 `D` 维。

## 6.4 数据管线：把 mask 位置特征传到模型输入

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py`

建议新增：

1. 从 mask 计算两个几何向量（例如中心点、宽高、面积、角度等）
2. 打包到 batch，例如：
   - `eagle_obj_pos_feat`
   - `eagle_place_pos_feat`
3. 在 backbone 输入透传给 `Eagle25VLForConditionalGeneration.forward()`

---

## 7. mask 到 token 的建议编码方式

你有两个 mask：

1. 目标物 mask
2. 放置物 mask

建议先从稳定且可解释的低维几何特征做起：

1. `cx, cy`（归一化中心）
2. `w, h`（归一化包围框尺寸）
3. `area_ratio`
4. `depth_hint`（如果有深度）

然后用一个小 MLP 投影到 LLM 维度 `D`：

```text
[B, F] --MLP--> [B, D]
```

再写入 `<OBJ_POS>` / `<PLACE_POS>` 位置。

这样做的好处：

1. 训练稳定
2. 特征可控
3. 与你 overlay 方案互补而不是替代

---

## 8. 关键稳定性修复（建议和注入一起做）

文件：`unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/cross_attention_dit.py`

当前问题：

1. `BasicTransformerBlock.forward()` 调用 attention 时，`encoder_attention_mask` 被注释（约 `:168`）
2. `DiT.forward()` 传入 block 时把 `encoder_attention_mask=None`（约 `:289`）

影响：

1. cross-attention 不会利用 backbone 的有效 token mask
2. 当序列长度变化或 padding 增加时，鲁棒性下降

建议修复：

1. 在 `attn1` 调用中传入 `encoder_attention_mask`
2. 在 `DiT.forward()` 把上层传下来的 mask 原样传给 block

---

## 9. 实验设计建议（和你现有结果对齐）

你的已验证基线：

1. `70% RGB + 30% overlay mask` 在复杂场景有效，且有一定泛化

建议实验矩阵：

1. Baseline：overlay only
2. Overlay + `after_img_end` 两位置 token
3. Overlay + `before_img_end` 两位置 token
4. Overlay + 两位置 token + cross-attn mask 修复

关键指标：

1. 指定目标抓取成功率
2. 非训练物体泛化成功率
3. 错抓率（被干扰物抓取）
4. 训练稳定性（loss 波动、NaN、收敛速度）

---

## 10. 验证清单（上线前必须通过）

1. token 验证
   - 能在 `input_ids` 中看到 `<OBJ_POS>`、`<PLACE_POS>`
2. embedding 替换验证
   - 对应位置 embedding 被 mask-derived 向量覆盖
3. 维度验证
   - 替换向量维度与 LLM 输入维度一致
4. 序列位置验证
   - `after_img_end` 模式下，两个 token 物理位置在 `</img>` 之后
5. action head mask 验证
   - `encoder_attention_mask` 确实进入 attention 计算
6. 回归验证
   - 不开启位置 token 时，性能不劣化

---

## 11. 常见坑

1. 仍然使用固定 `[20:276]` 解析视觉区间，新增 token 后会错位。
2. 新增 token 但未更新 tokenizer 资产和 embedding 大小。
3. mask 特征未经归一化直接输入 MLP，训练初期震荡大。
4. 只改 processor 文本，不改 forward embedding 替换，导致新 token 只是普通文本 token。
5. 忽略 action head 的 `encoder_attention_mask` 问题，导致长序列退化。

---

## 12. 推荐实施顺序

1. 先做 `after_img_end` 注入（最小风险）
2. 同步修复 DiT cross-attention mask 透传
3. 做小规模验证确认收益
4. 再做 `before_img_end` 作为对照

---

## 13. 和你当前方案的关系

你当前验证成功的核心是：

1. SAM2/分割得到 mask
2. overlay 到原图（70/30）
3. GR00T 能在复杂场景更稳地抓指定目标

新位置 token 不应替代这条路径，而应做“增益通道”：

1. overlay 提供像素级显著性引导
2. position token 提供结构化空间约束

两者叠加更符合你当前任务（HRI 场景中的目标/放置关系建模）。

---

## 14. 相关代码定位索引

1. 序列占位替换：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py:226`
2. VLM embedding 替换：`unitree_lerobot/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py:237`
3. 固定单 tile 设置：`unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py:518`
4. action head cross-attn mask 问题：
   - `unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/cross_attention_dit.py:168`
   - `unitree_lerobot/lerobot/src/lerobot/policies/groot/action_head/cross_attention_dit.py:289`
5. special tokens（本地缓存）：
   - `/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/added_tokens.json`
   - `/home/g1/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/config.json`

