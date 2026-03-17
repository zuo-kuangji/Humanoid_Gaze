"""
Inject `masks` paths into Unitree-style data.json files.

Use this when raw episodes already contain `masks/` images on disk but
`data.json` frames only include `colors`.

Example:
python unitree_lerobot/utils/add_mask_json.py \
    --raw-dir /home/g1/zuo/xr_teleoperate/teleop/utils/data/Gaze_datasets/pick_vegetable \
    --dry-run

python unitree_lerobot/utils/add_mask_json.py \
    --raw-dir /home/g1/zuo/xr_teleoperate/teleop/utils/data/Gaze_datasets/pick_vegetable \
    --no-dry-run
"""

import dataclasses
import json
from pathlib import Path

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    from_field: str = "colors"
    mask_field: str = "masks"
    dry_run: bool = True
    overwrite_existing: bool = False
    strict_exists: bool = True


def _map_path_to_mask(path: str, from_field: str, mask_field: str) -> str:
    if path.startswith(f"{from_field}/"):
        return f"{mask_field}/" + path[len(from_field) + 1 :]
    return path.replace(f"/{from_field}/", f"/{mask_field}/", 1)


def _resolve_existing_mask_rel_path(episode_dir: Path, rel_path: str) -> str:
    p = episode_dir / rel_path
    if p.exists():
        return rel_path

    rel = Path(rel_path)
    candidates = [
        rel.with_suffix(".png"),
        rel.with_suffix(".jpg"),
        rel.with_suffix(".jpeg"),
        rel.with_suffix(".bmp"),
    ]
    for c in candidates:
        if (episode_dir / c).exists():
            return str(c)

    for c in (episode_dir / rel.parent).glob(rel.stem + ".*"):
        if c.is_file():
            return str(rel.parent / c.name)
    return rel_path


def add_mask_json(
    raw_dir: Path,
    from_field: str = "colors",
    mask_field: str = "masks",
    dry_run: bool = True,
    overwrite_existing: bool = False,
    strict_exists: bool = True,
) -> None:
    json_paths = sorted(raw_dir.glob("*/*/data.json"))
    if not json_paths:
        raise FileNotFoundError(f"No data.json found under: {raw_dir}")

    changed_files = 0
    changed_frames = 0

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            episode = json.load(f)

        frames = episode.get("data", [])
        file_changed = False
        local_changed_frames = 0

        for frame in frames:
            colors = frame.get(from_field)
            if not isinstance(colors, dict):
                continue

            existing = frame.get(mask_field)
            if isinstance(existing, dict) and existing and not overwrite_existing:
                continue

            mapped: dict[str, str] = {}
            for cam, rel in colors.items():
                if not isinstance(rel, str):
                    continue
                mapped_rel = _map_path_to_mask(rel, from_field, mask_field)
                mapped_rel = _resolve_existing_mask_rel_path(json_path.parent, mapped_rel)
                if strict_exists:
                    mask_abs = json_path.parent / mapped_rel
                    if not mask_abs.exists():
                        continue
                mapped[cam] = mapped_rel

            if mapped:
                frame[mask_field] = mapped
                file_changed = True
                local_changed_frames += 1

        if file_changed:
            changed_files += 1
            changed_frames += local_changed_frames
            if not dry_run:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(episode, f, ensure_ascii=False, indent=2)
                    f.write("\n")

    mode = "DRY-RUN" if dry_run else "WRITE"
    print(
        f"[{mode}] done. files={len(json_paths)}, changed_files={changed_files}, "
        f"changed_frames={changed_frames}, field={mask_field}"
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    add_mask_json(
        raw_dir=args.raw_dir,
        from_field=args.from_field,
        mask_field=args.mask_field,
        dry_run=args.dry_run,
        overwrite_existing=args.overwrite_existing,
        strict_exists=args.strict_exists,
    )
