# GR00T N1.5 架构深度分析

## 目录
1. [整体架构概览](#1-整体架构概览)
2. [核心模块详解](#2-核心模块详解)
3. [数据流与API详解](#3-数据流与api详解)
4. [VLM与DiT的交互机制](#4-vlm与dit的交互机制)
5. [各脚本功能总结](#5-各脚本功能总结)

---

## 1. 整体架构概览

GR00T N1.5 采用 **双脑架构 (Dual-Brain Architecture)**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GR00T N1.5 整体架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │   Input Data      │    │   Eagle2.5 VLM    │    │   DiT Action Head │   │
│  │  (Video/State/    │───▶│    (Backbone)     │───▶│  (Flow Matching)  │   │
│  │   Language)       │    │                   │    │                   │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────┘   │
│           │                        │                        │              │
│           │                        ▼                        ▼              │
│           │               backbone_features          action_pred           │
│           │               [B, SeqLen, 1536]         [B, T, ActionDim]      │
│           │                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件
1. **EagleBackbone (VLM)**: 视觉-语言-机器人状态的统一表征
2. **FlowmatchingActionHead (DiT)**: 基于Flow Matching的动作生成器
3. **Processor Pipeline**: 数据预处理与后处理流水线

---

## 2. 核心模块详解

### 2.1 EagleBackbone (VLM 骨干网络)

**文件**: `groot_n1.py` (L52-144)

#### 架构组成

```python
class EagleBackbone(nn.Module):
    """
    Eagle2.5 VLM 作为 GR00T 的视觉-语言骨干网络
    
    组件:
    1. eagle_model: Eagle2.5-VL 完整模型
       - vision_model: SiglipVisionModel (视觉编码器)
       - language_model: Qwen2/Qwen3 (语言模型)
       - mlp1: Vision-to-LLM 投影层
    2. eagle_linear: 2048 -> 1536 的投影层
    """
```

#### 关键API

| 方法 | 输入 | 输出 | 功能 |
|------|------|------|------|
| `__init__()` | 配置参数 | - | 初始化Eagle模型，移除多余LLM层 |
| `forward_eagle()` | BatchFeature | (features, mask) | 提取VLM特征 |
| `forward()` | BatchFeature | BatchFeature | 完整前向传播 |
| `set_trainable_parameters()` | tune_llm, tune_visual | - | 设置可训练参数 |

#### VLM特征提取流程

```python
def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
    # 1. 提取eagle_*前缀的输入
    eagle_input = {k.removeprefix("eagle_"): v for k, v in vl_input.items() 
                   if k.startswith("eagle_")}
    
    # 2. Eagle模型前向传播，获取hidden states
    eagle_output = self.eagle_model(**eagle_input, 
                                     output_hidden_states=True, 
                                     return_dict=True)
    
    # 3. 选择指定层的特征 (默认 select_layer=-1, 即倒数第1层)
    eagle_features = eagle_output.hidden_states[self.select_layer]
    
    # 4. 线性投影 2048 -> 1536
    eagle_features = self.eagle_linear(eagle_features)
    
    return eagle_features, eagle_input["attention_mask"]
    # 输出: [B, SeqLen, 1536], [B, SeqLen]
```

**关键参数**:
- `select_layer`: 选择LLM哪一层的hidden states (-1表示最后一层)
- `tune_llm`: 是否微调语言模型 (默认False)
- `tune_visual`: 是否微调视觉编码器 (默认False)
- `project_to_dim`: 投影维度 (默认1536)

---

### 2.2 Eagle2.5-VL 模型详解

**文件**: `eagle2_hg_model/modeling_eagle2_5_vl.py`

#### 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      Eagle2.5-VL 模型                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ SiglipVision│    │    MLP1     │    │   Qwen2/3 LLM       │  │
│  │   Model     │───▶│  Connector  │───▶│  (Language Model)   │  │
│  │(448×448 img)│    │(4096->2048) │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        ▲                                         │              │
│        │                                         ▼              │
│   pixel_values                            hidden_states         │
│   [B, C, H, W]                           [B, SeqLen, 2048]      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### 关键方法

```python
class Eagle25VLForConditionalGeneration(PreTrainedModel, GenerationMixin):
    
    def extract_feature(self, pixel_values):
        """
        从图像提取视觉特征
        
        流程:
        1. ViT编码: pixel_values -> vit_embeds [B, 1024, 1024]
        2. Pixel Shuffle下采样: [B, 1024, 1024] -> [B, 256, 4096]
        3. MLP投影: [B, 256, 4096] -> [B, 256, 2048]
        """
        vit_embeds = self.vision_model(pixel_values, 
                                        output_hidden_states=True)
        vit_embeds = vit_embeds.hidden_states[self.select_layer]
        
        # Pixel Shuffle: 空间下采样，通道增加
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=0.5)
        
        # MLP投影到LLM维度
        vit_embeds = self.mlp1(vit_embeds)
        
        return vit_embeds  # [B, 256, 2048]
    
    def forward(self, pixel_values, input_ids, attention_mask, ...):
        """
        完整前向传播
        
        1. 获取文本embedding
        2. 提取视觉特征
        3. 将视觉特征插入到<IMG_CONTEXT>位置
        4. 送入LLM生成输出
        """
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        vit_embeds = self.extract_feature(pixel_values)
        
        # 替换 image_token 位置的 embedding
        selected = (input_ids == self.image_token_index)
        input_embeds[selected] = vit_embeds.reshape(-1, c)
        
        # LLM前向传播
        outputs = self.language_model(inputs_embeds=input_embeds, ...)
        
        return outputs
```

#### 配置参数 (configuration_eagle2_5_vl.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vision_config` | SiglipVisionConfig | 视觉编码器配置 |
| `text_config` | Qwen2Config | 语言模型配置 |
| `select_layer` | -4 | 选择ViT的哪一层特征 |
| `downsample_ratio` | 0.5 | Pixel Shuffle下采样率 |
| `use_pixel_shuffle` | True | 是否使用Pixel Shuffle |
| `mlp_connector_layers` | 2 | MLP连接器层数 |
| `min_dynamic_tiles` | 1 | 动态分块最小数量 |
| `max_dynamic_tiles` | 6 | 动态分块最大数量 |

---

### 2.3 FlowmatchingActionHead (DiT动作头)

**文件**: `action_head/flow_matching_action_head.py`

#### 架构概览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FlowmatchingActionHead 架构                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VLM Features ──┬─▶ [vlln] ──▶ [vl_self_attention] ──▶ encoder_hidden_states │
│  [B,S,1536]     │                                              │             │
│                 │                                              ▼             │
│                 │   ┌──────────────────────────────────────────────────────┐ │
│                 │   │              DiT Transformer                          │ │
│                 │   │  ┌─────────────────────────────────────────────────┐ │ │
│                 │   │  │ Cross-Attn Block (Layer 0, 2, 4, ...)           │ │ │
│                 │   │  │  Q: hidden_states (state+future+action)         │ │ │
│                 │   │  │  K,V: encoder_hidden_states (VLM features)      │ │ │
│                 │   │  └─────────────────────────────────────────────────┘ │ │
│                 │   │  ┌─────────────────────────────────────────────────┐ │ │
│                 │   │  │ Self-Attn Block (Layer 1, 3, 5, ...) [optional] │ │ │
│                 │   │  │  Q,K,V: all from hidden_states                  │ │ │
│                 │   │  └─────────────────────────────────────────────────┘ │ │
│                 │   └──────────────────────────────────────────────────────┘ │
│                 │                                              │             │
│ State ──────────┼─▶ [state_encoder] ──────┐                   │             │
│ [B,1,64]        │                          │                   │             │
│                 │                          ▼                   │             │
│ Future Tokens ──┼─▶ nn.Embedding ─────────┬──▶ hidden_states  │             │
│ (learnable)     │                          │   [B,T,1536]      │             │
│                 │                          │                   │             │
│ Noisy Action ───┴─▶ [action_encoder] ─────┘                   │             │
│ [B,T,32]                                                       │             │
│                                                                ▼             │
│                                                    [action_decoder]          │
│                                                          │                   │
│                                                          ▼                   │
│                                                    Predicted Velocity        │
│                                                      [B,T,32]                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 关键组件

##### 1. State Encoder (状态编码器)

```python
class CategorySpecificMLP(nn.Module):
    """
    多embodiment状态编码器
    每个embodiment有独立的MLP参数
    
    输入: state [B, 1, max_state_dim=64]
    输出: state_features [B, 1, 1536]
    """
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)
    
    def forward(self, x, cat_ids):
        # cat_ids: embodiment ID, 用于选择对应的权重
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)
```

##### 2. Action Encoder (动作编码器)

```python
class MultiEmbodimentActionEncoder(nn.Module):
    """
    带时间步编码的动作编码器
    
    输入: 
        actions [B, T, action_dim]
        timesteps [B,] - diffusion时间步
        cat_ids [B,] - embodiment ID
    输出: 
        action_features [B, T, 1536]
    """
    def forward(self, actions, timesteps, cat_ids):
        # 1. 动作线性投影
        a_emb = self.W1(actions, cat_ids)  # [B, T, hidden_size]
        
        # 2. 时间步正弦编码
        tau_emb = self.pos_encoding(timesteps)  # [B, T, hidden_size]
        
        # 3. 拼接并融合
        x = torch.cat([a_emb, tau_emb], dim=-1)  # [B, T, 2*hidden_size]
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        
        return x  # [B, T, 1536]
```

##### 3. Future Tokens (可学习的未来token)

```python
# 可学习的嵌入，用于预测未来动作
self.future_tokens = nn.Embedding(num_target_vision_tokens, 1536)
# 默认 num_target_vision_tokens = 32
```

##### 4. VL Self-Attention (可选)

```python
# 对VLM特征进行额外的self-attention处理
self.vlln = nn.LayerNorm(backbone_embedding_dim)
self.vl_self_attention = SelfAttentionTransformer(**config)
```

#### 训练流程 (forward)

```python
def forward(self, backbone_output, action_input):
    """
    训练时的前向传播 - Flow Matching
    
    核心思想: 学习从噪声到动作的速度场
    """
    # 1. 处理VLM特征
    vl_embs = self.process_backbone_output(backbone_output).backbone_features
    
    # 2. 获取embodiment ID
    embodiment_id = action_input.embodiment_id  # [B,]
    
    # 3. 编码状态
    state_features = self.state_encoder(action_input.state, embodiment_id)
    # [B, 1, 1536]
    
    # 4. Flow Matching: 添加噪声
    actions = action_input.action  # [B, T, action_dim]
    noise = torch.randn_like(actions)
    t = self.sample_time(batch_size)  # Beta分布采样时间
    t = t[:, None, None]  # [B, 1, 1]
    
    # 线性插值: noisy_trajectory = (1-t)*noise + t*actions
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise  # 目标速度
    
    # 5. 编码噪声动作
    t_discretized = (t[:, 0, 0] * 1000).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
    # [B, T, 1536]
    
    # 6. 添加位置编码
    action_features = action_features + self.position_embedding(pos_ids)
    
    # 7. 拼接所有token: [state, future_tokens, action]
    future_tokens = self.future_tokens.weight.expand(B, -1, -1)  # [B, 32, 1536]
    sa_embs = torch.cat([state_features, future_tokens, action_features], dim=1)
    # [B, 1+32+T, 1536]
    
    # 8. DiT前向传播 - Cross Attention with VLM features
    model_output = self.model(
        hidden_states=sa_embs,           # Q来源
        encoder_hidden_states=vl_embs,   # K,V来源 (VLM特征)
        timestep=t_discretized,
    )
    
    # 9. 解码动作
    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1]:]  # 取最后T个
    
    # 10. 计算MSE损失
    loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = loss.sum() / action_mask.sum()
    
    return {"loss": loss}
```

#### 推理流程 (get_action)

```python
@torch.no_grad()
def get_action(self, backbone_output, action_input):
    """
    推理时的动作生成 - 欧拉积分去噪
    """
    vl_embs = self.process_backbone_output(backbone_output).backbone_features
    state_features = self.state_encoder(action_input.state, embodiment_id)
    
    # 从纯噪声开始
    actions = torch.randn(batch_size, action_horizon, action_dim)
    
    num_steps = self.num_inference_timesteps  # 默认10步
    dt = 1.0 / num_steps
    
    # 迭代去噪
    for t in range(num_steps):
        t_cont = t / num_steps
        t_discretized = int(t_cont * 1000)
        
        # 编码当前动作
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        
        # 拼接token
        sa_embs = torch.cat([state_features, future_tokens, action_features], dim=1)
        
        # DiT预测速度
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred_velocity = self.action_decoder(model_output, embodiment_id)[:, -T:]
        
        # 欧拉积分更新
        actions = actions + dt * pred_velocity
    
    return {"action_pred": actions}
```

---

### 2.4 DiT Transformer (Diffusion Transformer)

**文件**: `action_head/cross_attention_dit.py`

#### 架构

```python
class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer
    
    特点:
    1. 交替的Cross-Attention和Self-Attention块
    2. AdaLayerNorm用于时间步条件化
    3. 正弦位置编码
    """
    
    def __init__(self, ...):
        # 时间步编码器
        self.timestep_encoder = TimestepEncoder(embedding_dim=inner_dim)
        
        # Transformer块
        for idx in range(num_layers):
            use_self_attn = (idx % 2 == 1) and interleave_self_attention
            
            if use_self_attn:
                # Self-Attention: Q,K,V都来自hidden_states
                cross_attention_dim = None
            else:
                # Cross-Attention: Q来自hidden_states, K,V来自VLM
                cross_attention_dim = cross_attention_dim
            
            blocks.append(BasicTransformerBlock(
                dim=inner_dim,
                cross_attention_dim=curr_cross_attention_dim,
                norm_type="ada_norm",  # 时间步条件化
                ...
            ))
```

#### 核心组件

##### 1. TimestepEncoder (时间步编码器)

```python
class TimestepEncoder(nn.Module):
    """
    将离散时间步转换为连续嵌入
    
    流程: timestep -> Timesteps(正弦) -> TimestepEmbedding(MLP)
    """
    def __init__(self, embedding_dim):
        self.time_proj = Timesteps(num_channels=256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
    
    def forward(self, timesteps):
        timesteps_proj = self.time_proj(timesteps)  # 正弦编码
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # MLP
        return timesteps_emb  # [B, embedding_dim]
```

##### 2. AdaLayerNorm (自适应层归一化)

```python
class AdaLayerNorm(nn.Module):
    """
    时间步条件化的LayerNorm
    
    x_out = LayerNorm(x) * (1 + scale) + shift
    其中 scale, shift 由时间步嵌入生成
    """
    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x
```

##### 3. BasicTransformerBlock

```python
class BasicTransformerBlock(nn.Module):
    """
    基础Transformer块
    
    结构:
    1. AdaLayerNorm + Attention (Cross或Self)
    2. LayerNorm + FeedForward
    """
    def forward(self, hidden_states, encoder_hidden_states=None, temb=None):
        # 1. AdaLN + Attention
        norm_hidden_states = self.norm1(hidden_states, temb)  # 时间步条件化
        
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,  # Cross-Attn时使用
        )
        hidden_states = attn_output + hidden_states
        
        # 2. LN + FFN
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        
        return hidden_states
```

#### DiT前向传播

```python
def forward(self, hidden_states, encoder_hidden_states, timestep, ...):
    """
    DiT完整前向传播
    
    Args:
        hidden_states: [B, T, D] - 状态+动作token
        encoder_hidden_states: [B, S, D] - VLM特征
        timestep: [B,] - 时间步
    """
    # 1. 编码时间步
    temb = self.timestep_encoder(timestep)  # [B, D]
    
    # 2. 通过Transformer块
    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1 and self.config.interleave_self_attention:
            # Self-Attention块
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None,  # Q,K,V都来自hidden_states
                temb=temb,
            )
        else:
            # Cross-Attention块
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,  # K,V来自VLM
                temb=temb,
            )
    
    # 3. 输出处理 (AdaLN + 投影)
    shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    output = self.proj_out_2(hidden_states)
    
    return output  # [B, T, output_dim]
```

---

## 4. VLM与DiT的交互机制

### 4.1 Cross-Attention 详解

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    Cross-Attention 机制                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  VLM Features (encoder_hidden_states)                                     │
│  [B, S, 1536] ──────────────────────────────┐                            │
│  S = 图像tokens + 文本tokens                  │                            │
│                                              │                            │
│                                              ▼                            │
│                                    ┌─────────────────┐                   │
│                                    │   Linear_K      │                   │
│                                    │   Linear_V      │                   │
│                                    └────────┬────────┘                   │
│                                             │                            │
│                                             ▼                            │
│                                      K: [B, S, D]                        │
│                                      V: [B, S, D]                        │
│                                             │                            │
│  State+Action (hidden_states)               │                            │
│  [B, T, 1536] ──────────────┐              │                            │
│  T = 1(state) + 32(future) + 16(action)    │                            │
│                              │              │                            │
│                              ▼              │                            │
│                    ┌─────────────────┐      │                            │
│                    │   Linear_Q      │      │                            │
│                    └────────┬────────┘      │                            │
│                             │               │                            │
│                             ▼               ▼                            │
│                      Q: [B, T, D]    ┌──────────────┐                   │
│                             │        │   Attention   │                   │
│                             └───────▶│   Softmax     │                   │
│                                      │   (Q @ K^T)   │                   │
│                                      └──────┬───────┘                   │
│                                             │                            │
│                                             ▼                            │
│                                    Attention Output                      │
│                                    [B, T, D]                             │
│                                                                           │
│  语义理解:                                                                 │
│  - Q (Query): 机器人状态和动作 "问": 在当前状态下应该怎么动?              │
│  - K (Key): VLM特征 "索引": 这是图像和语言的语义信息                      │
│  - V (Value): VLM特征 "答案": 根据attention权重加权的视觉语言信息          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Self-Attention 详解 (可选交替层)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    Self-Attention 机制 (交替层)                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  State+Action (hidden_states)                                             │
│  [B, T, 1536] ──────────────────────────────────┐                        │
│                                                  │                        │
│                              ┌───────────────────┼───────────────────┐   │
│                              │                   │                   │   │
│                              ▼                   ▼                   ▼   │
│                    ┌─────────────────┐ ┌─────────────────┐ ┌────────────┐│
│                    │   Linear_Q      │ │   Linear_K      │ │  Linear_V  ││
│                    └────────┬────────┘ └────────┬────────┘ └─────┬──────┘│
│                             │                   │                │       │
│                             ▼                   ▼                ▼       │
│                      Q: [B, T, D]         K: [B, T, D]      V: [B, T, D] │
│                             │                   │                │       │
│                             └───────────┬───────┘                │       │
│                                         ▼                        │       │
│                               ┌──────────────────┐               │       │
│                               │    Attention     │               │       │
│                               │  Softmax(Q@K^T)  │◀──────────────┘       │
│                               └────────┬─────────┘                       │
│                                        │                                 │
│                                        ▼                                 │
│                               Attention Output                           │
│                                  [B, T, D]                               │
│                                                                           │
│  语义理解:                                                                 │
│  - 状态和动作tokens之间相互关注                                            │
│  - 建模动作序列的时间依赖关系                                              │
│  - 让future tokens能够聚合信息                                            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 4.3 VLM输出位置与数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VLM输出位置与数据流                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Eagle2.5-VL 处理流程:                                                    │
│     ┌──────────────────────────────────────────────────────────────────────┐│
│     │ Images [B,V,C,H,W]  ──▶ SiglipVision ──▶ pixel_shuffle ──▶ MLP1     ││
│     │                         [B,1024,1024]    [B,256,4096]     [B,256,2048]│
│     │                                                                      ││
│     │ Text (input_ids) ──▶ Tokenizer ──▶ Embedding                        ││
│     │                                     [B,S_text,2048]                  ││
│     │                                                                      ││
│     │ 合并: [CLS, IMG_tokens×256, TEXT_tokens] ──▶ Qwen2 LLM              ││
│     │                                                                      ││
│     │ 输出: hidden_states[select_layer] ──▶ eagle_linear                  ││
│     │       [B, SeqLen, 2048]              [B, SeqLen, 1536]               ││
│     └──────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  2. VLM输出格式:                                                             │
│     backbone_features: [B, SeqLen, 1536]                                     │
│     backbone_attention_mask: [B, SeqLen]                                     │
│                                                                              │
│     SeqLen = num_vision_tokens + num_text_tokens                             │
│     典型值: 256 (vision) + ~100 (text) ≈ 356                                 │
│                                                                              │
│  3. DiT接收位置:                                                             │
│     FlowmatchingActionHead.forward() / get_action()                          │
│     └── process_backbone_output(backbone_output)                             │
│         └── vl_embs = backbone_output.backbone_features  [B, S, 1536]       │
│     └── model(encoder_hidden_states=vl_embs)                                 │
│         └── DiT的Cross-Attention层使用vl_embs作为K和V                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 各脚本功能总结

### 5.1 主目录文件

| 文件 | 功能 | 主要API |
|------|------|---------|
| `groot_n1.py` | GR00T N1.5核心模型定义 | `EagleBackbone`, `GR00TN15`, `GR00TN15Config` |
| `modeling_groot.py` | LeRobot集成的Policy包装器 | `GrootPolicy` |
| `configuration_groot.py` | Policy配置类 | `GrootConfig` |
| `processor_groot.py` | 数据预处理/后处理流水线 | `make_groot_pre_post_processors()`, `GrootPackInputsStep`, `GrootEagleEncodeStep`, `GrootEagleCollateStep` |
| `utils.py` | 工具函数 | `ensure_eagle_cache_ready()` |

### 5.2 action_head/ 目录

| 文件 | 功能 | 主要API |
|------|------|---------|
| `flow_matching_action_head.py` | Flow Matching动作头 | `FlowmatchingActionHead`, `FlowmatchingActionHeadConfig`, `CategorySpecificMLP`, `MultiEmbodimentActionEncoder` |
| `cross_attention_dit.py` | DiT Transformer | `DiT`, `SelfAttentionTransformer`, `BasicTransformerBlock`, `AdaLayerNorm`, `TimestepEncoder` |
| `action_encoder.py` | 动作编码器组件 | `SinusoidalPositionalEncoding`, `swish()` |

### 5.3 eagle2_hg_model/ 目录

| 文件 | 功能 | 主要API |
|------|------|---------|
| `modeling_eagle2_5_vl.py` | Eagle2.5-VL模型 | `Eagle25VLForConditionalGeneration`, `Eagle25VLPreTrainedModel` |
| `configuration_eagle2_5_vl.py` | Eagle配置 | `Eagle25VLConfig` |
| `processing_eagle2_5_vl.py` | Eagle处理器 | `Eagle25VLProcessor` |
| `image_processing_eagle2_5_vl_fast.py` | 快速图像处理 | `Eagle25VLImageProcessorFast` |

---

### 5.4 详细脚本功能

#### 5.4.1 groot_n1.py

```python
# 主要类和功能

class EagleBackbone(nn.Module):
    """
    Eagle2.5 VLM骨干网络
    
    功能:
    - 加载Eagle2.5-VL模型
    - 提取视觉-语言特征
    - 支持选择性微调 (tune_llm, tune_visual)
    
    关键方法:
    - forward_eagle(): 提取VLM hidden states
    - forward(): 完整前向传播
    - set_trainable_parameters(): 设置可训练参数
    """

class GR00TN15Config(PretrainedConfig):
    """
    模型配置类
    
    关键配置:
    - backbone_cfg: EagleBackbone配置
    - action_head_cfg: FlowmatchingActionHead配置
    - action_horizon: 动作预测步数 (默认16)
    - action_dim: 动作维度 (默认32)
    """

class GR00TN15(PreTrainedModel):
    """
    GR00T N1.5完整模型
    
    关键方法:
    - forward(): 训练时前向传播
    - get_action(): 推理时动作生成
    - from_pretrained(): 加载预训练模型
    - prepare_input(): 准备输入数据
    """
```

#### 5.4.2 modeling_groot.py

```python
class GrootPolicy(PreTrainedPolicy):
    """
    LeRobot集成的Policy包装器
    
    功能:
    - 包装GR00TN15模型供LeRobot使用
    - 处理动作队列和temporal ensembling
    
    关键方法:
    - forward(): 训练forward
    - predict_action_chunk(): 预测动作chunk
    - select_action(): 选择单个动作
    - reset(): 重置状态
    """
```

#### 5.4.3 processor_groot.py

```python
# 预处理流水线步骤

class GrootPackInputsStep(ProcessorStep):
    """
    打包输入数据
    
    功能:
    - 将video转换为numpy格式
    - 归一化state和action (min-max)
    - 填充到max_state_dim/max_action_dim
    - 添加embodiment_id
    """

class GrootEagleEncodeStep(ProcessorStep):
    """
    Eagle编码步骤
    
    功能:
    - 使用Eagle处理器编码图像和文本
    - 生成eagle_content中间表示
    """

class GrootEagleCollateStep(ProcessorStep):
    """
    Eagle collate步骤
    
    功能:
    - 批处理eagle_content
    - 生成eagle_*张量 (pixel_values, input_ids, attention_mask等)
    """

class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    """
    动作后处理步骤
    
    功能:
    - 裁剪到环境动作维度
    - 反归一化 (inverse min-max)
    """
```

#### 5.4.4 flow_matching_action_head.py

```python
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """
    配置参数:
    - hidden_size: 1024 (内部维度)
    - input_embedding_dim: 1536 (输入嵌入维度)
    - action_dim: 32 (动作维度)
    - action_horizon: 16 (动作步数)
    - num_inference_timesteps: 10 (推理步数)
    - noise_beta_alpha/beta: Beta分布参数
    - max_num_embodiments: 32 (最大embodiment数)
    - num_target_vision_tokens: 32 (future tokens数量)
    """

class FlowmatchingActionHead(nn.Module):
    """
    Flow Matching动作头
    
    组件:
    - state_encoder: CategorySpecificMLP
    - action_encoder: MultiEmbodimentActionEncoder
    - action_decoder: CategorySpecificMLP
    - model: DiT
    - future_tokens: nn.Embedding
    - vlln: LayerNorm
    - vl_self_attention: SelfAttentionTransformer
    
    关键方法:
    - forward(): 训练时计算loss
    - get_action(): 推理时生成动作
    - process_backbone_output(): 处理VLM特征
    """
```

#### 5.4.5 cross_attention_dit.py

```python
class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer
    
    配置:
    - num_attention_heads: 8
    - attention_head_dim: 64
    - num_layers: 12
    - dropout: 0.1
    - interleave_self_attention: 是否交替self-attn
    
    组件:
    - timestep_encoder: 时间步编码
    - transformer_blocks: Cross/Self Attention块列表
    - norm_out: 输出LayerNorm
    - proj_out_1, proj_out_2: 输出投影
    """

class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    """
    纯Self-Attention Transformer
    
    用于对VLM特征进行额外处理
    """

class BasicTransformerBlock(nn.Module):
    """
    基础Transformer块
    
    结构: AdaLN -> Attention -> Residual -> LN -> FFN -> Residual
    支持Cross-Attention和Self-Attention
    """

class AdaLayerNorm(nn.Module):
    """
    自适应LayerNorm
    
    通过时间步嵌入调制scale和shift
    """
```

---

## 6. 完整数据流示意图

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           GR00T N1.5 完整数据流                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  输入数据                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ video: [B,1,V,C,H,W] uint8    state: [B,D_state]    action: [B,T,D_action] ││
│  │ language: str                  embodiment_tag: str                          ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                       │
│                                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        Preprocessor Pipeline                                 ││
│  │  GrootPackInputsStep: 归一化 + 填充                                          ││
│  │  GrootEagleEncodeStep: 图像+文本编码                                          ││
│  │  GrootEagleCollateStep: 批处理                                               ││
│  │  DeviceProcessorStep: 移至GPU                                                ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                       │
│                                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         EagleBackbone (VLM)                                  ││
│  │                                                                              ││
│  │  eagle_pixel_values ──▶ SiglipVision ──▶ pixel_shuffle ──▶ MLP1            ││
│  │  eagle_input_ids ──▶ Qwen2 Embedding                                        ││
│  │  合并 ──▶ Qwen2 LLM (截断到select_layer=-1)                                  ││
│  │       ──▶ eagle_linear (2048->1536)                                         ││
│  │                                                                              ││
│  │  输出: backbone_features [B, SeqLen, 1536]                                   ││
│  │        backbone_attention_mask [B, SeqLen]                                   ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                       │
│                                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                      FlowmatchingActionHead (DiT)                           ││
│  │                                                                              ││
│  │  1. VLM特征处理:                                                             ││
│  │     backbone_features ──▶ vlln ──▶ vl_self_attention                        ││
│  │                                                                              ││
│  │  2. 状态编码:                                                                ││
│  │     state [B,1,64] ──▶ state_encoder ──▶ [B,1,1536]                         ││
│  │                                                                              ││
│  │  3. 动作编码 (训练时):                                                        ││
│  │     noise ──▶ noisy_action = (1-t)*noise + t*action                         ││
│  │     noisy_action ──▶ action_encoder ──▶ [B,T,1536]                          ││
│  │                                                                              ││
│  │  4. Token拼接:                                                               ││
│  │     [state_feat, future_tokens, action_feat] ──▶ [B, 1+32+T, 1536]          ││
│  │                                                                              ││
│  │  5. DiT Transformer:                                                         ││
│  │     ┌──────────────────────────────────────────────────────────────────┐   ││
│  │     │ for layer in transformer_blocks:                                  │   ││
│  │     │   if Cross-Attention层:                                           │   ││
│  │     │     Q = hidden_states, K,V = VLM_features                         │   ││
│  │     │   elif Self-Attention层:                                          │   ││
│  │     │     Q,K,V = hidden_states                                         │   ││
│  │     │   output = AdaLN(output, timestep_emb) + FFN                      │   ││
│  │     └──────────────────────────────────────────────────────────────────┘   ││
│  │                                                                              ││
│  │  6. 动作解码:                                                                ││
│  │     output[:, -T:] ──▶ action_decoder ──▶ pred_velocity [B,T,D_action]      ││
│  │                                                                              ││
│  │  训练: loss = MSE(pred_velocity, target_velocity) * action_mask             ││
│  │  推理: action = action + dt * pred_velocity (欧拉积分)                       ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                       │
│                                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        Postprocessor Pipeline                                ││
│  │  GrootActionUnpackUnnormalizeStep: 裁剪+反归一化                              ││
│  │  DeviceProcessorStep: 移至CPU                                                ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                       │
│                                          ▼                                       │
│  输出: action_pred [B, T, D_action_env]                                          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键配置参数汇总

### 7.1 EagleBackbone 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tune_llm` | False | 是否微调LLM |
| `tune_visual` | False | 是否微调视觉编码器 |
| `select_layer` | -1 | 选择LLM第几层的hidden states |
| `project_to_dim` | 1536 | 输出投影维度 |

### 7.2 FlowmatchingActionHead 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 1024 | DiT内部维度 |
| `input_embedding_dim` | 1536 | 输入嵌入维度 |
| `action_dim` | 32 | 动作维度 |
| `action_horizon` | 16 | 动作预测步数 |
| `num_inference_timesteps` | 10 | 推理去噪步数 |
| `num_timestep_buckets` | 1000 | 时间步离散化桶数 |
| `noise_beta_alpha` | 1.5 | Beta分布α参数 |
| `noise_beta_beta` | 1.0 | Beta分布β参数 |
| `max_num_embodiments` | 32 | 最大embodiment数 |
| `num_target_vision_tokens` | 32 | future tokens数量 |

### 7.3 DiT 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_attention_heads` | 8 | 注意力头数 |
| `attention_head_dim` | 64 | 每个头的维度 |
| `num_layers` | 12 | Transformer层数 |
| `dropout` | 0.1 | Dropout率 |
| `interleave_self_attention` | False | 是否交替Self-Attn |
| `norm_type` | "ada_norm" | 归一化类型 |

---

## 8. 总结

GR00T N1.5 的核心创新点：

1. **双脑架构**: VLM负责理解，DiT负责动作生成
2. **Cross-Attention桥接**: DiT通过Cross-Attention从VLM获取视觉语言信息
3. **Flow Matching**: 使用连续归一化流替代离散扩散，提高采样效率
4. **多Embodiment支持**: 通过CategorySpecific层支持不同机器人形态
5. **Future Tokens**: 可学习的tokens帮助预测未来动作序列

这种架构使得模型能够：
- 理解复杂的视觉场景和语言指令
- 生成平滑连贯的动作序列
- 适应不同的机器人形态
- 实现端到端的视觉-语言-动作学习
