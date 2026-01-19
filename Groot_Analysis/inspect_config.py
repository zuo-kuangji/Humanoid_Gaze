
import sys
import torch
from lerobot.common.policies.factory import make_policy
from lerobot.policies.groot.groot_n1 import GR00TN15Config

# Add local lerobot path
sys.path.append("/home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot")

def inspect_policy_config():
    pretrained_path = "/home/g1/zuo/unitree_IL_lerobot1/unitree_lerobot/lerobot/outputs/train/groot_handover/checkpoints/020000/pretrained_model"
    
    # Load config directly to see raw structure first
    try:
        config = GR00TN15Config.from_pretrained(pretrained_path)
        print("\n[CONFIG] Loaded Config:")
        if hasattr(config, "action_head_cfg"):
            print(f"Action Head Config present: {config.action_head_cfg.keys()}")
            print(f"Interleave Self Attention: {config.action_head_cfg.get('diffusion_model_cfg', {}).get('interleave_self_attention', 'Not Found')}")
        else:
            print("Action Head Config NOT found in main config object.")
            
    except Exception as e:
        print(f"Error loading config: {e}")

if __name__ == "__main__":
    inspect_policy_config()
