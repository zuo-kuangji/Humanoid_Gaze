# SAM2 实时物体追踪系统文档

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              整体架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐          ZMQ          ┌─────────────────────────┐    │
│   │  eval_g1_sam.py │  ◄─────────────────►  │   run_gaze_server.py    │    │
│   │   (客户端)       │    TCP:5556           │      (SAM2 服务器)       │    │
│   │                 │                       │                         │    │
│   │  groot1.5 环境   │                       │     glasses 环境        │    │
│   └─────────────────┘                       └─────────────────────────┘    │
│           │                                           │                    │
│           ▼                                           ▼                    │
│   ┌─────────────────┐                       ┌─────────────────────────┐    │
│   │  RealSense      │                       │      SAM2 模型          │    │
│   │  相机 → RGB 图像 │                       │  sam2.1_hiera_small.pt  │    │
│   └─────────────────┘                       └─────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. run_gaze_server.py (SAM2 服务器)

### 1.1 SAM2 处理流程

```
初始化阶段 (init 命令):
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 接收第一帧 RGB 图像                                                    │
│ 2. 弹出 matplotlib GUI 窗口                                               │
│ 3. 用户点击选择物体:                                                       │
│    - 左键: 前景点 (Foreground, label=1)                                   │
│    - 右键: 背景点 (Background, label=0)                                   │
│    - 空格: 预览分割效果                                                    │
│    - 关闭窗口: 确认选择                                                    │
│ 4. 使用 SAM2ImagePredictor 生成初始 mask                                  │
│ 5. 初始化 SAM2 VideoPredictor 的 inference_state                          │
│ 6. 将用户点击的 points + labels 添加到 frame_idx=0                        │
└──────────────────────────────────────────────────────────────────────────┘

追踪阶段 (track 命令):
┌──────────────────────────────────────────────────────────────────────────┐
│ 对于每一帧:                                                               │
│                                                                          │
│ 1. 预处理图像:                                                            │
│    - resize 到 SAM2 要求的尺寸 (self.img_size)                           │
│    - 转换为 tensor: (H,W,3) → (3,H,W)                                    │
│    - 归一化: (img - mean) / std                                          │
│                                                                          │
│ 2. 帧跳跃策略 (Frame Skipping):                                          │
│    - 每 3 帧才运行一次 SAM2 推理 (约 10Hz)                                │
│    - 中间帧返回上一次的 mask (last_mask)                                  │
│    - 原因: SAM2 推理较慢，无法达到 30Hz                                   │
│                                                                          │
│ 3. 注入图像到 inference_state["images"] 列表                              │
│                                                                          │
│ 4. 内存清理 (每 100 帧):                                                  │
│    - 只保留 frame_0 (初始条件) + 最近 30 帧                               │
│    - 防止 VRAM 无限增长                                                   │
│                                                                          │
│ 5. SAM2 追踪推理:                                                         │
│    - propagate_in_video(start_frame_idx, max_frame_num_to_track=1)       │
│    - 输出: mask_logits → 二值化 (>0.0) → numpy mask                      │
│                                                                          │
│ 6. 应用红色 Overlay:                                                      │
│    - 公式: output[mask] = original[mask] * 0.7 + red * 0.3               │
│    - 即 70% 原图 + 30% 纯红色                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Overlay 算法 (与 SAM2_Video_Seg.py 一致)

```python
def apply_mask_overlay(self, frame_rgb, mask, alpha=0.3):
    """
    与 SAM2_Video_Seg.py 的 save_frame_async() 函数逻辑完全一致
    """
    # 创建纯红色图层
    red_mask = np.zeros_like(output)
    red_mask[:, :, 0] = 255  # R=255, G=0, B=0
    
    # 只在 mask 区域混合: 70% 原图 + 30% 红色
    output[mask] = (output[mask] * 0.7 + red_mask[mask] * 0.3).astype(np.uint8)
```

### 1.3 ZMQ 通信协议

| 命令 | 请求格式 | 响应格式 |
|------|----------|----------|
| `ping` | `{"cmd": "ping"}` | `{"status": "pong"}` |
| `init` | `{"cmd": "init", "image": np.ndarray(RGB)}` | `{"status": "ok", "success": bool}` |
| `track` | `{"cmd": "track", "image": np.ndarray(RGB)}` | `{"status": "ok", "mask": np.ndarray, "overlayed_image": np.ndarray}` |
| `reset` | `{"cmd": "reset"}` | `{"status": "ok"}` |

---

## 2. eval_g1_sam.py (机器人评估客户端)

### 2.1 工作流程

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         eval_g1_sam.py 流程                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ 1. 启动阶段:                                                               │
│    ├── 连接 ZMQ Gaze Server (tcp://localhost:5556)                        │
│    ├── ping 检查服务器存活                                                 │
│    ├── 设置 RealSense 相机 (image_server 共享内存)                         │
│    └── 加载 GR00T 策略模型                                                 │
│                                                                            │
│ 2. SAM2 初始化 (sam_init_on_start=True):                                  │
│    ├── 从共享内存获取第一帧 (BGR → RGB 转换)                               │
│    ├── 发送到 Gaze Server (init 命令)                                      │
│    ├── 弹出 GUI 让用户点选目标物体                                          │
│    └── 等待用户关闭 GUI 确认                                               │
│                                                                            │
│ 3. 主循环 (30Hz):                                                          │
│    ┌─────────────────────────────────────────────────────────────────┐    │
│    │ while True:                                                      │    │
│    │   ├── process_images_and_observations() 获取 RGB 图像            │    │
│    │   │   (observation["observation.images.cam_head"])              │    │
│    │   ├── 发送到 Gaze Server (track 命令)                           │    │
│    │   ├── 接收: mask + overlayed_image (带红色 overlay)             │    │
│    │   ├── 将 overlayed_image 替换到 observation["..cam_head"]       │    │
│    │   ├── 送入 GR00T 策略推理                                       │    │
│    │   ├── 执行机械臂动作                                            │    │
│    │   └── 控制频率 (sleep)                                          │    │
│    └─────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ 关键图像格式:                                                              │
│   - 共享内存 (tv_img_array): BGR uint8 (H, W, 3)                          │
│   - observation 图像: RGB uint8 (H, W, 3) torch.Tensor                    │
│   - SAM2 输入/输出: RGB uint8 numpy.ndarray (H, W, 3)                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 新增配置参数

```python
@dataclass
class EvalRealSAMConfig(EvalRealConfig):
    use_sam: bool = False           # 是否启用 SAM2 追踪
    gaze_port: int = 5556           # Gaze Server 端口
    sam_init_on_start: bool = True  # 机器人启动前是否初始化 SAM2
```

### 2.3 GazeClient 类

```python
class GazeClient:
    def ping(self) -> bool:
        """检查服务器是否存活"""
        
    def initialize_tracker(self, image_rgb) -> bool:
        """发送第一帧，触发用户点选 GUI"""
        
    def track(self, image_rgb) -> Tuple[np.ndarray, np.ndarray]:
        """发送当前帧，返回 (mask, overlayed_image)"""
        
    def reset(self) -> bool:
        """重置追踪状态"""
```

---

## 3. SAM2 帧处理策略

### 3.1 为什么不是每帧都处理？

```
问题: SAM2 推理速度约 30-50ms/帧，无法达到 30Hz

解决方案: 帧跳跃 (Frame Skipping)
┌─────────────────────────────────────────────────────────────────┐
│ Frame 0:  SAM2 推理 ✓  → 生成 mask_0                           │
│ Frame 1:  跳过         → 复用 mask_0                           │
│ Frame 2:  跳过         → 复用 mask_0                           │
│ Frame 3:  SAM2 推理 ✓  → 生成 mask_3                           │
│ Frame 4:  跳过         → 复用 mask_3                           │
│ Frame 5:  跳过         → 复用 mask_3                           │
│ Frame 6:  SAM2 推理 ✓  → 生成 mask_6                           │
│ ...                                                            │
└─────────────────────────────────────────────────────────────────┘

效果: 实际 SAM2 推理频率 ≈ 10Hz (30Hz / 3)
     Overlay 输出频率 = 30Hz (使用缓存的 last_mask)
```

### 3.2 代码实现

```python
def step(self, frame_rgb):
    self.frame_idx += 1
    
    # 帧跳跃: 只有 frame_idx % 3 == 0 时才运行 SAM2
    if hasattr(self, "last_mask") and (self.frame_idx % 3 != 0):
        # 注入图像但跳过推理
        # ...
        return self.last_mask  # 复用上次的 mask
    
    # 运行 SAM2 追踪
    # ...
    self.last_mask = mask
    return mask
```

---

## 4. 使用方法

### ⚠️ 重要：工作目录

**所有命令必须从项目根目录 `/home/g1/unitree_groot1.5` 运行！**

因为 `robot_arm_ik.py` 使用相对路径加载 URDF 文件：
```python
"unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.urdf"
```

如果从 `unitree_lerobot/lerobot` 目录运行，会报错：
```
ValueError: The file unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.urdf does not contain a valid URDF model.
```

### 4.1 启动顺序

```bash
# 终端 1: 启动 SAM2 服务器 (需要 glasses conda 环境)
conda activate glasses
cd /home/g1/unitree_groot1.5
python unitree_lerobot/eval_robot/run_gaze_server.py --port 5556

# 终端 2: 启动机器人评估 (需要 groot1.5 conda 环境)
conda activate groot1.5
cd /home/g1/unitree_groot1.5
python unitree_lerobot/eval_robot/eval_g1_sam.py \
  --policy.path=unitree_lerobot/lerobot/outputs/train/groot_mask_handover_v2/checkpoints/last/pretrained_model \
  --repo_id=ZUO66/handover_mask_drinks \
  --use_sam=True \
  --gaze_port=5556 \
  --frequency=30 \
  --motion=True \
  --arm="G1_29" \
  --ee="inspire1" \
  --send_real_robot=True \
  --visualization=True
```

### 4.2 交互流程

1. **启动服务器** → 等待 "Gaze Server Ready" 消息
2. **启动客户端** → 自动发送第一帧到服务器
3. **GUI 弹出** → 左键点击要追踪的物体
4. **按空格预览** → 检查分割效果
5. **关闭 GUI 窗口** → 确认选择
6. **按 's' 启动机器人** → 开始实时追踪 + 动作执行

---

## 5. 与 SAM2_Video_Seg.py 的对比

| 特性 | SAM2_Video_Seg.py | run_gaze_server.py |
|------|-------------------|---------------------|
| **处理模式** | 离线批处理 | 实时在线流处理 |
| **输入** | 视频文件/图片序列 | ZMQ 实时接收 |
| **帧处理** | 所有帧都处理 | 每 3 帧处理一次 |
| **Overlay 公式** | `orig*0.7 + red*0.3` | `orig*0.7 + red*0.3` ✓ |
| **输出** | 保存到文件 | 实时返回给客户端 |
| **内存管理** | 无 (离线处理完释放) | 每 100 帧清理旧帧 |
| **初始化** | 交互式点选 | 交互式点选 ✓ |

---

## 6. 性能参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SAM2 模型 | sam2.1_hiera_small | 8GB+ VRAM |
| 推理精度 | bfloat16 | 节省显存 |
| SAM2 推理频率 | ~10Hz | 每 3 帧处理一次 |
| Overlay 输出频率 | 30Hz | 使用缓存 mask |
| 内存清理周期 | 每 100 帧 | 保留最近 30 帧 |
| ZMQ 超时 | 5000ms | 防止阻塞 |

---

## 7. 文件清单

```
unitree_lerobot/eval_robot/
├── run_gaze_server.py      # SAM2 服务器 (glasses 环境)
├── eval_g1_sam.py          # 带 SAM2 的机器人评估 (groot1.5 环境)
├── eval_g1.py              # 原始机器人评估 (无 SAM2)
└── SAM2_README.md          # 本文档
```
