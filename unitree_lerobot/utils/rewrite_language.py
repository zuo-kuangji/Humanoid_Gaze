"""
批量修改数据集中所有 episode 的 language 文本（data.json 中的 text.goal 字段）。

==============================================================================
使用方法 (Usage)
==============================================================================

1. 先用 --dry-run 预览哪些文件会被修改（不会实际写入）:
   python unitree_lerobot/utils/rewrite_language.py \
       --data-root /home/g1/zuo/xr_teleoperate/teleop/utils/data/handover_mask_two_drinks \
       --new-language "递蓝色饮料" \
       --dry-run

2. 正式修改（默认会备份原文件为 data.json.bak）:
   python unitree_lerobot/utils/rewrite_language.py \
       --data-root /home/g1/zuo/xr_teleoperate/teleop/utils/data/handover_mask_two_drinks \
       --new-language "递蓝色饮料"

python unitree_lerobot/utils/rewrite_language.py \
       --data-root /home/g1/zuo/xr_teleoperate/teleop/utils/data/Gaze_datasets/pick_crumpled_paper_ball_gaze \
       --new-language "Pick up the red crumpled paper ball and put it in the green bin."	   

3. 如果不想备份，加 --no-backup:
   python unitree_lerobot/utils/rewrite_language.py \
       --data-root /home/g1/zuo/xr_teleoperate/teleop/utils/data/handover_mask_two_drinks \
       --new-language "递蓝色饮料" \
       --no-backup

==============================================================================
参数说明 (Arguments)
==============================================================================
--data-root     数据集根目录，脚本会遍历其下所有 task/episode 子目录，寻找 data.json
--new-language  要写入的新 language 文本
--key-path      JSON 中的键路径，默认 "text.goal"（对应 data.json["text"]["goal"]）
--backup        是否在修改前备份原文件，默认 True
--dry-run       仅打印将要修改的文件，不实际写入

==============================================================================
目录结构示例
==============================================================================
data_root/
├── blue_drink/
│   ├── episode_0000/
│   │   └── data.json   <-- 会被修改
│   ├── episode_0001/
│   │   └── data.json   <-- 会被修改
│   └── ...
└── red_drink/
    ├── episode_0000/
    │   └── data.json   <-- 会被修改
    └── ...

脚本会递归遍历所有 episode 目录，修改每个 data.json 中的 text.goal 字段。
修改后该字段会在转换 HDF5 时写入 language_raw 和 substep_reasonings。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import tyro


def _collect_episode_jsons(data_root: Path) -> list[Path]:
	"""Return all data.json files under task/episode style directories."""

	data_root = data_root.expanduser().resolve()
	json_paths: list[Path] = []

	# Common layout: data_root/<task>/<episode>/data.json
	for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
		for episode_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
			candidate = episode_dir / "data.json"
			if candidate.exists():
				json_paths.append(candidate)

	# Fallback: grab any data.json below the root
	if not json_paths:
		json_paths = sorted(data_root.rglob("data.json"))

	return json_paths


def _set_nested_value(payload: dict, key_path: str, value: str) -> tuple[bool, str | None]:
	"""Set a dotted key (like text.goal) on payload; return (changed, old_value)."""

	keys = key_path.split(".") if key_path else []
	if not keys:
		raise ValueError("key_path cannot be empty")

	node = payload
	for key in keys[:-1]:
		current = node.get(key)
		if not isinstance(current, dict):
			current = {}
		node[key] = current
		node = current

	leaf_key = keys[-1]
	old_value = node.get(leaf_key)
	if old_value == value:
		return False, old_value

	node[leaf_key] = value
	return True, old_value


def rewrite_language(
	data_root: Path,
	new_language: str,
	key_path: str = "text.goal",
	backup: bool = True,
	dry_run: bool = False,
) -> None:
	"""Rewrite the language string for every episode data.json under data_root."""

	json_paths = _collect_episode_jsons(data_root)
	if not json_paths:
		raise FileNotFoundError(f"No data.json found under {data_root}")

	updated, skipped = 0, 0
	for json_path in json_paths:
		try:
			with open(json_path, encoding="utf-8") as f:
				payload = json.load(f)
		except json.decoder.JSONDecodeError as e:
			print(f"ERROR: Malformed JSON detected in {json_path}")
			raise e

		changed, old_value = _set_nested_value(payload, key_path, new_language)
		if not changed:
			skipped += 1
			continue

		if backup:
			backup_path = json_path.with_suffix(json_path.suffix + ".bak")
			if not backup_path.exists():
				shutil.copy(json_path, backup_path)

		if not dry_run:
			with open(json_path, "w", encoding="utf-8") as f:
				json.dump(payload, f, ensure_ascii=False, indent=2)

		updated += 1
		old_preview = str(old_value)[:80] if old_value is not None else "<missing>"
		print(f"Updated {json_path}: '{old_preview}' -> '{new_language}'")

	print(f"Done. Updated={updated}, Unchanged={skipped}, Files={len(json_paths)}")


if __name__ == "__main__":
	tyro.cli(rewrite_language)
