
import os
import torch
import sys
import json
from pathlib import Path

# 1. 确保加载的是工程目录下的 lerobot 源码
PROJECT_ROOT = Path("/home/g1/zuo/unitree_IL_lerobot1")
sys.path.insert(0, str(PROJECT_ROOT / "unitree_lerobot/lerobot/src"))

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from transformers import AutoTokenizer

# 直接复现您的推理环境，全自动流程审计。
def main():
    # 路径设定
    checkpoint_path = "/home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/outputs/train/groot_handover/checkpoints/020000/pretrained_model"
    checkpoint_path = os.path.abspath(checkpoint_path)
    repo_id = "ZUO66/handover_drinks"
    
    print(f"[*] 正在加载真实数据集: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    
    print(f"[*] 正在加载策略配置: {checkpoint_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)

    # 1. 初始化模型
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    
    # 2. 初始化预处理器 
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset.meta.stats
    )
    
    # 3. 直接加载 tokenizer (不从 policy 内部找了，直接从 HF hub 加载)
    tokenizer = AutoTokenizer.from_pretrained("lerobot/eagle2hg-processor-groot-n1p5", trust_remote_code=True)
    
    print("[+] 真实预处理流水线加载成功。")

    # 4. 抓取真实样本
    sample = dataset[0]
    task_instruction = sample["task"]
    print(f"[*] 提取数据集第 0 帧指令: '{task_instruction}'")

    # 构造 Observation (使用数据集的真实键名)
    observation = {
        "observation.images.cam_head": sample["observation.images.cam_head"].unsqueeze(0),
        "observation.state": sample["observation.state"].unsqueeze(0),
        "task": task_instruction
    }

    # 5. 执行流程审计 (让 Preprocessor 正常工作)
    processed_batch = preprocessor(observation)
    
    # 提取生成的 Tokens
    input_ids = processed_batch["eagle_input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print("\n" + "="*75)
    print(f"官方流水线真实序列审计 (总长度: {len(tokens)})")
    print("="*75)
    
    # 6. 定位地标
    try:
        idx_start = tokens.index("<img>")
        idx_end = tokens.index("</img>")
    except ValueError:
        print("[错误] 未能定位 <img> 标识符。")
        return

    # 7. 打印 (遵循指令：省略中间 240 个图像 Token)
    for i, t in enumerate(tokens):
        # 功能角色定义
        role = "System/Prompt/Metadata"
        if i == idx_start: role = ">>> 图像起始 <<<"
        elif i == idx_end: role = ">>> 图像结束 <<<"
        elif idx_start < i < idx_end: role = "IMAGE PATCH"
        elif i > idx_end: role = "Task/Action"

        # 打印范围逻辑
        if i <= idx_start + 8 or i >= idx_end - 8:
            clean_t = t.replace("Ġ", " ").replace("Ċ", "\\n")
            print(f"索引 {i:03d} | {clean_t:<25} | {role}")
            
            if i == idx_start + 8:
                print(f"       ... (已省略中间约 240 个图像 Token) ...")

    print("\n" + "="*75)
    print(f"【审计结果】")
    print(f"1. 图像物理位置范围: [{idx_start + 1} : {idx_end}]")
    print(f"2. 图像起始索引 (Start): {idx_start + 1}")
    print(f"3. 图像结束索引 (End):   {idx_end - 1}")
    print(f"4. 图像 Patch 总数:      {idx_end - idx_start - 1}")
    print("="*75)

if __name__ == "__main__":
    main()
