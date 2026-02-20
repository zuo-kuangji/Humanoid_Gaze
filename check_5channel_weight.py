import torch
from safetensors.torch import load_file

# 主模型（当前要检查的）
MODEL_PATH = "/home/g1/unitree_groot1.5/unitree_lerobot/lerobot/outputs/train/groot_pick_paper_ball_30000/pretrained_model/model.safetensors"

# 可选：基准模型（用于对比每个通道变化）；不需要对比就保持 None
BASE_MODEL_PATH = None

KEY = "_groot_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight"


def channel_labels(num_channels: int):
    if num_channels == 3:
        return ["R", "G", "B"]
    if num_channels == 5:
        return ["R", "G", "B", "mask1", "mask2"]
    return [f"ch{i}" for i in range(num_channels)]


def load_weight(path: str, key: str):
    state = load_file(path)
    if key not in state:
        raise KeyError(f"Key not found: {key}\\nPath: {path}")
    return state[key]


def print_basic_info(w: torch.Tensor, title: str):
    print(f"{title}: shape={tuple(w.shape)}, dtype={w.dtype}")
    if w.ndim != 4:
        print("Warning: expected 4D conv weight [out, in, kH, kW].")
        return
    out_ch, in_ch, kh, kw = w.shape
    print(f"  out_channels={out_ch}, in_channels={in_ch}, kernel={kh}x{kw}")
    print(f"  global L2={w.norm():.8f}, std={w.std():.6f}")
    print()


def per_channel_stats(w: torch.Tensor, title: str):
    if w.ndim != 4:
        return
    labels = channel_labels(w.shape[1])
    print(f"{title} per-input-channel stats:")
    for ch in range(w.shape[1]):
        wc = w[:, ch]
        label = labels[ch] if ch < len(labels) else f"ch{ch}"
        print(
            f"  Channel {ch} ({label}): "
            f"L2={wc.norm():.8f}, std={wc.std():.6f}, mean={wc.mean():.6f}, max|w|={wc.abs().max():.6f}"
        )
    print()


def compare_per_channel(base: torch.Tensor, trained: torch.Tensor):
    if base.shape != trained.shape:
        raise ValueError(f"Shape mismatch: base={tuple(base.shape)} vs trained={tuple(trained.shape)}")

    labels = channel_labels(trained.shape[1])
    print("Per-channel diff (trained - base):")
    for ch in range(trained.shape[1]):
        wb = base[:, ch]
        wt = trained[:, ch]
        diff = wt - wb
        label = labels[ch] if ch < len(labels) else f"ch{ch}"
        print(f"  Channel {ch} ({label}):")
        print(f"    base:    L2={wb.norm():.8f}, std={wb.std():.6f}")
        print(f"    trained: L2={wt.norm():.8f}, std={wt.std():.6f}")
        print(f"    diff:    L2={diff.norm():.6f}, max|diff|={diff.abs().max():.6f}")
        print(f"    change:  {diff.norm() / (wb.norm() + 1e-8) * 100:.4f}%")
    print()


if __name__ == "__main__":
    w_model = load_weight(MODEL_PATH, KEY)
    print_basic_info(w_model, "Model")
    per_channel_stats(w_model, "Model")

    if BASE_MODEL_PATH:
        w_base = load_weight(BASE_MODEL_PATH, KEY)
        print_basic_info(w_base, "Base")
        compare_per_channel(w_base, w_model)
    else:
        print("BASE_MODEL_PATH is None, skip diff comparison.")
