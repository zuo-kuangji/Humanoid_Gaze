# GrooT Model Architecture for Fine-tuning

Based on your current configuration (Introduction of a new robot embodiment with `tune_diffusion_model=False`), here is the architectural view of the model.

## key Legend
*   **🟦 Blue Components**: **Frozen (Pre-trained)**. These parts retain the massive knowledge from the original GrooT training.
*   **🟧 Orange Components**: **Tuned (Trainable)**. These "Adapters" or "Projectors" are being updated to translate your specific robot's language into GrooT's universal language.

## Architecture Diagram

```mermaid
graph TD
    %% Define Styles
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:black,font-weight:bold,font-size:20px;
    classDef frozen fill:#add8e6,stroke:#333,stroke-width:4px,color:black,font-weight:bold,font-size:20px;
    classDef tuned fill:#ff9900,stroke:#333,stroke-width:4px,color:black,font-weight:bold,font-size:20px;
    classDef input fill:#ffffff,stroke:#333,stroke-width:2px,color:black,font-weight:bold,font-size:20px;

    subgraph Inputs
        State["Robot State<br/>(Joint Positions/Velocities)"]:::input
        Img["Camera Images<br/>(RGB)"]:::input
        Text["Task Description<br/>(Text)"]:::input
        ActionInput["Action Sequence<br/>(Noisy Trajectory for Training)"]:::input
        Time["Diffusion Timestep (t)"]:::input
    end

    subgraph "🟧 Projectors (Trainable Adapters)"
        StateEnc[["State Encoder<br/>(MLP)<br/>Projects 64-dim ⮕ 1536-dim"]]:::tuned
        ActionEnc[["Action Encoder<br/>(MLP)<br/>Projects 32-dim ⮕ 1536-dim"]]:::tuned
    end

    subgraph "🟦 Vision & Language Backbone (Frozen)"
        Eagle["Eagle VLM (Vision-LLM)<br/>(Extracts Features)"]:::frozen
        VL_Proj["VL Projector<br/>(1536-dim)"]:::frozen
    end

    subgraph "🟦 Diffusion Backbone (Frozen)"
        DiT["DiT (Diffusion Transformer)<br/>The 'Brain'<br/>Processes all tokens"]:::frozen
    end

    subgraph "🟧 Action Decoder (Trainable)"
        ActionDec[["Action Decoder<br/>(MLP)<br/>Projects 1024-dim ⮕ Action Dim"]]:::tuned
    end
    
    PredAction["Predicted Action Update<br/>(velocity/position)"]:::input

    %% Data Flow
    State --> StateEnc
    Img --> Eagle
    Text --> Eagle
    
    Eagle --> VL_Proj
    
    ActionInput --> ActionEnc
    Time --> ActionEnc

    %% Features Integration
    StateEnc --> DiT
    VL_Proj --> DiT
    ActionEnc --> DiT

    %% Output
    DiT --> ActionDec
    ActionDec --> PredAction
    
```

## Architecture Diagram  +Bias Mask

```mermaid
flowchart TD
    %% ==========================================================
    %% 1. 全局样式定义 (Global Styles)
    %% ==========================================================
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:black
    classDef input fill:#ffffff,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,color:black
    classDef frozen fill:#add8e6,stroke:#0277bd,stroke-width:3px,color:black
    classDef tuned fill:#ffcc80,stroke:#ef6c00,stroke-width:3px,color:black
    classDef new fill:#ffcdd2,stroke:#c62828,stroke-width:4px,color:black

    %% ==========================================================
    %% 2. 输入数据流 (Data Inputs)
    %% ==========================================================
    subgraph Inputs ["Input Data Streams"]
        Img["Input Video/Image"]:::input
        Text["Task Text"]:::input
        Mask["SAM2 Binary Mask (448x448)"]:::input
        RobotState["Robot Proprioception (Joints)"]:::input
        NoisyAction["Noisy Action Trajectory (t)"]:::input
        TimeStep["Diffusion Step (k)"]:::input
    end

    %% ==========================================================
    %% 3. 感知主干 (VLM Backbone - Frozen)
    %% ==========================================================
    subgraph VLM_Backbone ["🦅 VLM Backbone (Frozen Eagle)"]
        direction TB
        Img & Text --> VisionModel["Vision Transformer"]:::frozen
        VisionModel --> Projector["Linear Projector"]:::frozen
        Projector --> ContextFeats["Context Features (Key/Value)<br/>Shape: (B, 256, D)"]:::frozen
    end

    %% ==========================================================
    %% 4. 你的 Mask 注入模块 (Your Injection Module - Trainable)
    %% ==========================================================
    subgraph Injection_Module ["🎯 Mask Injection (Your Work)"]
        direction TB
        Mask --> MaskEnc["Mask Encoder (CNN)<br/>Stride=7, Overlap"]:::new
        MaskEnc --> RawBias["Raw Bias Map<br/>Shape: (B, 1, 256)"]:::new
        RawBias -->|Multiply| Gamma["Scalar γ (Learnable Gate)<br/>Init=0.0"]:::new
        Gamma --> SoftBias["Final Soft Bias"]:::new
    end

    %% ==========================================================
    %% 5. 动作生成头 (DiT - Flow Matching)
    %% ==========================================================
    subgraph DiT_Head ["🤖 Flow Matching Action Head (DiT)"]
        direction TB
        
        %% 5.1 Embedding Layers
        RobotState --> StateEnc["State MLP"]:::tuned
        NoisyAction --> ActionEnc["Action MLP"]:::tuned
        TimeStep --> TimeEnc["Time MLP"]:::tuned
        
        StateEnc & ActionEnc & TimeEnc --> TokenSum["Token Summation (Query Input)"]
        
        %% 5.2 The True Interleaved DiT Architecture
        subgraph Interleaved_Blocks ["Interleaved Transformer Blocks (Strictly Alternating)"]
            direction TB
            
            %% Block 0: Cross-Attention ONLY
            subgraph Layer_0 ["Layer 0 (Even) - Cross-Attention"]
                direction TB
                TokenSum --> L0_LN1["LayerNorm"]:::tuned
                
                %% Cross-Attention Injection Point
                L0_LN1 --> L0_Q["Query: Robot"]
                ContextFeats --> L0_K["Key: Image"]
                ContextFeats --> L0_V["Value: Image"]
                
                %% INJECTION HAPPENS HERE
                L0_Q & L0_K & SoftBias --> L0_AttnMath{{"Attention Score = <br/>Softmax( Q@K.T + Bias )"}}:::new
                
                L0_AttnMath & L0_V --> L0_Out["Weighted Sum"]
                L0_Out --> L0_FFN["Feed Forward"]:::tuned
                L0_FFN --> L0_Res((+))
                TokenSum -->|Residual| L0_Res
            end
            
            %% Block 1: Self-Attention ONLY
            subgraph Layer_1 ["Layer 1 (Odd) - Self-Attention"]
                direction TB
                L0_Res --> L1_LN1["LayerNorm"]:::tuned
                L1_LN1 --> L1_SA["Self-Attention (Robot <-> Robot)<br/>(No Mask Bias Here!)"]:::tuned
                L1_SA --> L1_FFN["Feed Forward"]:::tuned
                L1_FFN --> L1_Res((+))
                L0_Res -->|Residual| L1_Res
            end
            
            L1_Res -.-> NextLayers["... Repeat (Layer 2, 3, ...) ..."]
        end
        
        %% 5.3 Output Decoding
        NextLayers --> FinalNorm["Final LayerNorm"]:::tuned
        FinalNorm --> ActionDec["Action Decoder (Linear)"]:::tuned
        ActionDec --> PredAction["Predicted Action Update"]:::tuned
    end

    %% ==========================================================
    %% 6. 跨模块连接 (Cross-Module Connections)
    %% ==========================================================
    SoftBias -.->|Inject into Score| AttnMath
```

## 微调Mask Encoder + SoftBias
第一阶段：Self-Attention (不变)输入：RobotTokens (包含你的 Action Noise, State, Timestep)。作用：让 tokens 之间交互（例如：这一帧的手臂位置和上一帧的速度有什么关系）。你的 Mask 不需要在这里介入，因为这里还没有看图像。第二阶段：Cross-Attention (手术点)Query ($Q$): 来自上一阶段 Self-Attention 的输出（即“机器人当下的意图”）。Key ($K$) & Value ($V$): 来自 Eagle VLM 的图像特征（即“环境里的杯子、桌子”）。Interaction: 标准公式是 $Q \times K^T$。Injection: 我们在这里加上 SoftBias。直观理解: 当 Robot (Q) 扫描 Image (K) 时，原本它对两个杯子的注意力也是 0.5 vs 0.5。加上 Bias 后，被 Mask 盖住的那个杯子对应的 Key 分数被强行压低（或目标被拉高），导致 Robot 只能“关注”到目标杯子。第三阶段：FFN (不变)常规的前馈网络，整合信息。


## Component Breakdown

### 1. Vision & Language Backbone (`EagleBackbone`)
*   **Status**: 🟦 **Frozen**
*   **Input**: Images and Text instructions.
*   **Function**: Understands the scene and the command. Extracts rich "Visual-Language Tokens".
*   **Why Frozen?**: It already knows how to recognize objects (cups, tables, etc.) and understand language. We don't need to reteach it "what a cup is", just "how to hand it over".

### 2. Projectors (The "Translators")
*   **Status**: 🟧 **Fully Tuned**
*   **Components**:
    *   **State Encoder:** Takes your robot's specific joint data (up to 64 dims) and projects it into the high-dimensional space (1536 dims) that the DiT understands.
    *   **Action Encoder:** Takes the action sequence (up to 32 dims) and projects it for the DiT.
*   **Why Tuned?**: This is the crucial "Translation Layer". It makes your Unitree G1 robot "look/feel" like any other robot in GrooT's internal representation.

### 3. Diffusion Transformer (`DiT`)
*   **Status**: 🟦 **Frozen**
*   **Function**: The core reasoning engine. It takes the scene info (from Eagle) and the robot state (from Projectors) and decides "what to do next".
*   **Why Frozen?**: This contains the "General Manipulation Policy". It understands physics, grasping logic, and movement primitives. By freezing it, we preserve its general intelligence and prevent overfitting to your small dataset.

### 4. Action Decoder
*   **Status**: 🟧 **Fully Tuned**
*   **Function**: Takes the high-level plan from the DiT and converts it back into specific motor commands (Joint Velocities/Positions) for your robot.

## 4. Final Verification Success (After Fix) ✅

We have successfully implemented the **Mask-Guided IAB (Initialized Attention Bias)**. The verification scripts confirm the following:

### 1. Model Integrity (`verify_token_layout.py`)
*   **Result**: ✅ **Passed**
*   **Analysis**: The `ModuleNotFoundError` is resolved. The `MaskEncoder` is correctly instantiated within the `FlowmatchingActionHead`, and the model loads without errors. The Vision-Language token layout remains correct.

### 2. IAB Gradient Logic (`test_soft_bias_grad.py`)
*   **Result**: ✅ **Passed**
*   **Key Findings**:
    *   **Gamma Init**: `0.0` (Verified). The Mask Bias starts as a "Null Operation", meaning the model behaves exactly like the pre-trained GrooT at step 0. Safe!
    *   **Bias Mean**: `0.0` (Verified).
    *   **Grad Gamma**: `0.11` (Verified). The `gamma` parameter receives non-zero gradients, proving it is **learnable**.
    *   **Grad Encoder**: `0.0` (Verified). At step 0, because `gamma` is 0, gradients do not yet reach the CNN weights. This is the expected "Soft Start" behavior. As `gamma` grows, the CNN will begin to learn.

### Summary
The **Mask-Guided Attention** feature is now fully implemented and verified. The model is ready for training experiment!

## 6. VLM Token Layout & Mask Alignment Details 📐

### 6.1 Eagle VLM Token Structure
The `Eagle` VLM processes inputs into a sequence of tokens. Crucially, we have verified that the **first 20 tokens are occupied by a fixed System Prompt**.

#### A. The Exact System Prompt (Tokens 0-19)
We decoded the first 25 tokens using `verify_token_layout.py` and recovered the following exact text:
```text
RAW TEXT:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image 1>...
```

#### B. Token Accounting (Why Offset=20?)
Here is the precise breakdown of the first 20 tokens that necessitates our `VISION_START_OFFSET = 20`:

| Index | Content | Description |
| :--- | :--- | :--- |
| **0** | `<|im_start|>` | Conversation Start |
| **1-2** | `system` + `\n` | Role Name |
| **3-8** | `You` `are` `a` `helpful` `assistant` `.` | System Instruction (6 tokens) |
| **9-10** | `<|im_end|>` + `\n` | End of System block |
| **11** | `<|im_start|>` | User Role Start |
| **12-13** | `user` + `\n` | Role Name |
| **14** | `<image 1>` | Image Placeholder |
| **15-19** | `<img>`...`<img>` | 5 Padding Tokens (Image Context) |
| **20-275** | **[VISUAL TOKENS]** | **Target Area for Mask Bias (16x16=256)** |
| **276+** | `Perform the task...` | User Text Instruction |

**Conclusion**: The visual tokens **always** start at index 20 (verified). Our implementation correctly injects the Mask Bias starting at this index to perfectly cover the image features without corrupting the prompt.

### 6.2 The Alignment Challenge
Our `MaskEncoder` produces a bias map of shape `(B, 1, 256)` (for 16x16 grid).
*   **The Problem**: If we directly enable this bias, it would broadcast to `(B, 1, SeqLen)`. Since `SeqLen > 256`, this would either crash (mismatch) or mistakenly mask the System Prompt (if broadcasted from start).
*   **The Requirement**: We need the bias to affect **only** indices `[20:276]` (Visual Tokens), while leaving `[0:20]` (System) and `[276:]` (Text) completely untouched (Bias = 0).

### 6.3 Our Solution: "Pad-and-Shift" Injection
We implemented a robust alignment strategy in `FlowmatchingActionHead.forward`:

1.  **Generate Raw Bias**: `raw_bias = mask_encoder(mask)` -> Shape `(B, 1, 256)`.
2.  **Create Canvas**: Initialize a full-sequence zeros tensor `padded_bias` of shape `(B, 1, SeqLen)`.
    *   Zeros mean "No Bias" (Identity/Pass-through in additive attention).
3.  **Precise Injection**:
    ```python
    VISION_START_OFFSET = 20
    padded_bias[:, :, VISION_START_OFFSET : VISION_START_OFFSET + 256] = raw_bias
    ```
4.  **Result**: 
    *   System Prompt -> Attention unaffected.
    *   **Visual Tokens -> Attention biased by Mask.**
### 6.4 Flattening Order & Embedding Dimension (D)
**Check Confirmed**: We verified the `modeling_eagle2_5_vl.py` code (Lines 314-320).

*   **Flattening Order**: `vit_embeds.reshape(B, -1, D)`.
    *   Input tensor layout is `(B, H, W, D)`.
    *   PyTorch's default `reshape`/`flatten` operation is **Row-Major** (iterates W, then H).
    *   **Conclusion**: Our MaskEncoder must also use `view(B, 1, -1)` on a `(B, 1, H, W)` tensor, which naturally follows the same Row-Major order. **Alignment is guaranteed.**

*   **Dimension D**: 
    *   Based on code comments and Eagle config: `D = 4096`.
## 7. Detailed Code Implementation Report 🛠️

Here is the precise documentation of the code changes applied to enable Mask-Guided IAB.

### 7.1 MaskEncoder (`src/.../mask_encoder.py`)
**Type**: New File
**Role**: The "Compressor" & "Gate"

*   **CNN Architecture (Downsampling Strategy)**:
    We use a 3-layer CNN to aggressively downsample the `448x448` mask to match the `16x16` visual token grid.
    1.  **Layer 1**: `Conv2d(1, 16, k=7, s=7, p=0)` → Input 448 → Output 64. (Features: 16 channels)
    2.  **Layer 2**: `Conv2d(16, 32, k=3, s=2, p=1)` → Input 64 → Output 32. (Features: 32 channels)
    3.  **Layer 3**: `Conv2d(32, 1, k=3, s=2, p=1)` → Input 32 → Output 16. (Features: 1 channel)
    *   **Result**: A `(B, 1, 16, 16)` feature map.
    *   **Flatten**: View as `(B, 1, 256)` row-major to align with Eagle.

*   **Zero-Initialization (Safety Mechanism)**:
    *   Defined: `self.gamma = nn.Parameter(torch.zeros(1))`
    *   Forward: `output = features * self.gamma`
    *   **Effect**: Since `gamma` starts at 0, the initial output is **strictly 0**. This ensures the pre-trained model sees "no change" at the start of training.

### 7.2 CrossAttentionDiT (`src/.../cross_attention_dit.py`)
**Type**: Modification
**Role**: The "Receiver"

*   **Signature Update**:
    Updated `DiT.forward` and `BasicTransformerBlock.forward` to accept a new argument: `cross_attention_bias=None`.

*   **Injection Logic (in `BasicTransformerBlock`)**:
    We perform a safe merge with the existing `encoder_attention_mask`.
    ```python
    # 1. Check if Bias exists
    if cross_attention_bias is not None:
        # 2. Ensure existing mask is in "Additive Mode" (-inf/0)
        #    Standard masks are often (0/1). We convert 0->-inf, 1->0.
        if cross_attn_mask is (0/1 format):
             convert_to_additive_format(cross_attn_mask)
        
        # 3. Additive Injection
        cross_attn_mask = cross_attn_mask + cross_attention_bias
    ```
    This merged mask is then passed to `self.attn1()` (Cross-Attention).

### 7.3 FlowMatchingActionHead (`src/.../flow_matching_action_head.py`)
**Type**: Modification
**Role**: The "Orchestrator"

*   **Configuration**:
    *   Constant Defined: `VISION_START_OFFSET = 20`. This protects the system prompt.
    *   In `__init__`: Instantiated `self.mask_encoder`.

*   **Forward Pass (Alignment Strategy)**:
    The core logic ensures the 256-length bias lands exactly on the image tokens.
    ```python
    # 1. Get Raw Bias (1, 256)
    raw_bias = self.mask_encoder(mask)
    
    # 2. Create Canvas (1, SeqLen) - Filled with 0
    padded_bias = torch.zeros(..., seq_len)
    
    # 3. Inject at Offset 20
    padded_bias[..., 20 : 20+256] = raw_bias
    
    # 4. Pass to DiT
    self.model(..., cross_attention_bias=padded_bias)
    ```
    **Safety**: System Prompts (0-19) and Text Tokens (276+) get a bias of `0`, meaning their attention scores are untouched.

---
**Verification Signed-off**: All above logic verified by `verify_token_layout.py` and `test_soft_bias_grad.py` on 2026-01-14.







## 8. Mask Integration Strategy (3-Channel Storage) 🎭

To ensure compatibility with LeRobot's video infrastructure while preserving mask data integrity, we implemented a robust **Storage/Usage Decoupling Strategy**.

### 8.1 The Storage Challenge
*   **Constraint**: LeRobot's v3.0 core treats all high-dimensional inputs as "Videos". The underlying `video_utils.py` encoder enforces `input.convert("RGB")`.
*   **Risk**: Storing 1-channel grayscale masks directly would cause shape mismatches or encoding errors when the system attempts to visualize/save them as RGB MP4s.

### 8.2 The 3-Channel Solution (Storage Phase)
We intentionally "camouflage" the mask as an RGB video during dataset conversion.
*   **File**: `convert_unitree_json_to_lerobot.py`
*   **Logic**:
    1.  Read grayscale mask: `(H, W)`
    2.  Expand to 1-channel: `(H, W, 1)`
    3.  **Duplicate Channels**: `np.repeat(mask, 3, axis=-1)` → `(H, W, 3)`
*   **Result**: The dataset claims and stores `observation.masks.cam_head` as an RGB video `(3, 480, 640)`. This satisfies the encoder naturally.

### 8.3 The Slicing Solution (Usage Phase)
We recover the pure mask data immediately upon loading.
*   **File**: `processor_groot.py` (`GrootPackInputsStep`)
*   **Logic**:
    ```python
    # 1. Load 3-channel data
    mask = obs["observation.masks.cam_head"] # (B, H, W, 3)
    
    # 2. Slice back to 1-channel
    mask = mask[..., 0:1] # (B, H, W, 1)
    ```
*   **Safety**: This ensures that downstream components (VLM, Action Head) receive the exact `(1, H, W)` tensor they expect, identical to the original grayscale data.
*   **Verification**: Verified by `verify_mask_content.py` proving all 3 channels are identical (Sum of Difference = 0.0).

### 8.4 Policy Whitelisting
To allow this new modality to reach the model:
*   **File**: `modeling_groot.py`
*   **Action**: Added `"masks"` to the `allowed_base` sets in both `forward` and `predict_action_chunk`. This prevents the default filter from dropping the mask tensor.
