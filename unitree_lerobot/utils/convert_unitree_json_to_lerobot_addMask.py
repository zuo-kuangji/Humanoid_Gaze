"""
Script Json to LeRobot (RGB + optional pick mask).

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --vision-field Which visual modality to read from each frame (default: colors)
# --include-pick-mask Whether to export observation.mask_pick (3-channel mask image)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot_addMask.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \
    --include-pick-mask \
    --push_to_hub

# Mask fallback conversion (without editing all data.json files):
# If frame["masks"] is missing, the script maps frame["colors"] path
# from "colors/xxx" to "masks/xxx" and loads grayscale masks as 3-channel RGB.
python unitree_lerobot/utils/convert_unitree_json_to_lerobot_addMask.py \
    --raw-dir /path/to/raw_dataset \
    --repo-id your_name/your_dataset_with_mask \
    --robot_type Unitree_G1_Inspire_Head \
    --vision-field colors \
    --include-pick-mask \
    --no-push-to-hub
"""

import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(
        self,
        data_dirs: Path,
        robot_type: str,
        vision_field: str = "colors",
        *,
        include_pick_mask: bool = True,
        pick_mask_field: str = "masks",
    ) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"
        self.vision_field = self._normalize_vision_field(vision_field)
        self.include_pick_mask = include_pick_mask
        self.pick_mask_field = self._normalize_vision_field(pick_mask_field)

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    @staticmethod
    def _normalize_vision_field(vision_field: str) -> str:
        field_map = {
            "color": "colors",
            "colors": "colors",
            "overlay": "overlays",
            "overlays": "overlays",
            "depth": "depths",
            "depths": "depths",
            "mask": "masks",
            "masks": "masks",
        }
        if vision_field not in field_map:
            valid = ", ".join(sorted(field_map))
            raise ValueError(f"Unsupported vision_field: {vision_field}. Valid values: {valid}")
        return field_map[vision_field]

    @staticmethod
    def _to_overlay_path(path: str) -> str:
        if path.startswith("colors/"):
            return "overlays/" + path[len("colors/") :]
        return path.replace("/colors/", "/overlays/", 1)

    @staticmethod
    def _to_mask_path(path: str) -> str:
        if path.startswith("colors/"):
            return "masks/" + path[len("colors/") :]
        return path.replace("/colors/", "/masks/", 1)

    def _resolve_frame_vision_paths(self, sample_data: dict) -> dict[str, str]:
        paths = sample_data.get(self.vision_field)
        if isinstance(paths, dict):
            return paths

        # Fallback for datasets that only keep colors in JSON while overlay images
        # exist on disk with the same camera keys and file names.
        if self.vision_field == "overlays":
            color_paths = sample_data.get("colors")
            if isinstance(color_paths, dict):
                return {
                    key: self._to_overlay_path(value)
                    for key, value in color_paths.items()
                    if isinstance(value, str)
                }

        return {}

    def _resolve_frame_pick_mask_paths(self, sample_data: dict) -> dict[str, str]:
        paths = sample_data.get(self.pick_mask_field)
        if isinstance(paths, dict) and paths:
            return paths

        # Fallback: masks path inferred from colors path.
        color_paths = sample_data.get("colors")
        if isinstance(color_paths, dict):
            return {
                key: self._to_mask_path(value)
                for key, value in color_paths.items()
                if isinstance(value, str)
            }
        return {}

    @staticmethod
    def _resolve_existing_mask_rel_path(episode_path: str, rel_path: str) -> str:
        """Resolve mask path with suffix fallback (.png/.jpg/.jpeg/.bmp)."""
        abs_path = Path(episode_path) / rel_path
        if abs_path.exists():
            return rel_path

        rel = Path(rel_path)
        candidates = [
            rel.with_suffix(".png"),
            rel.with_suffix(".jpg"),
            rel.with_suffix(".jpeg"),
            rel.with_suffix(".bmp"),
        ]
        for c in candidates:
            if (Path(episode_path) / c).exists():
                return str(c)

        # Last fallback: same stem under same directory, any suffix.
        for c in (Path(episode_path) / rel.parent).glob(rel.stem + ".*"):
            if c.is_file():
                return str(rel.parent / c.name)
        return rel_path

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []

        for task_path in glob.glob(os.path.join(self.data_dirs, "*")):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path, "*"))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> list:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        first_paths = self._resolve_frame_vision_paths(episode_data["data"][0])
        if not first_paths:
            # Fallback to configured camera keys to provide clear error messages per frame.
            first_paths = {k: "" for k in self.camera_to_image_key}

        keys = first_paths.keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = self._resolve_frame_vision_paths(sample_data).get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Image path does not exist: {image_path} (vision_field={self.vision_field}, camera={camera})"
                    )

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        if not any(images.values()):
            raise ValueError(
                f"No images were loaded for vision_field='{self.vision_field}'. "
                "Check JSON modality keys and image folders."
            )

        return images

    def _parse_pick_masks(self, episode_path: str, episode_data: dict) -> list[np.ndarray]:
        """Load mask images and convert them to 3-channel RGB arrays."""
        frame0_paths = self._resolve_frame_pick_mask_paths(episode_data["data"][0])
        if not frame0_paths:
            raise ValueError(
                "No pick mask paths found. Expected frame['masks'] or fallback from frame['colors']."
            )

        # Prefer camera keys defined in robot config. If missing, use the first available key.
        candidate_keys = [k for k in frame0_paths if k in self.camera_to_image_key]
        if not candidate_keys:
            candidate_keys = list(frame0_paths.keys())
        if not candidate_keys:
            raise ValueError("Failed to resolve any mask camera key.")
        camera_key = sorted(candidate_keys)[0]

        masks: list[np.ndarray] = []
        for sample_data in episode_data["data"]:
            curr_paths = self._resolve_frame_pick_mask_paths(sample_data)
            relative_path = curr_paths.get(camera_key)
            if not relative_path and curr_paths:
                # Last fallback if camera key changed unexpectedly.
                relative_path = next(iter(curr_paths.values()))
            if not relative_path:
                raise ValueError(f"Mask path missing for camera={camera_key} in one frame.")

            relative_path = self._resolve_existing_mask_rel_path(episode_path, relative_path)
            mask_path = os.path.join(episode_path, relative_path)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"Failed to read mask image: {mask_path}")

            if mask.ndim == 2:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask_rgb = np.repeat(mask, 3, axis=2)
            elif mask.ndim == 3 and mask.shape[2] == 3:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            elif mask.ndim == 3 and mask.shape[2] == 4:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unsupported mask shape at {mask_path}: {mask.shape}")

            masks.append(mask_rgb)

        return masks

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)
        pick_masks = self._parse_pick_masks(file_path, episode_data) if self.include_pick_mask else None

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "pick_masks": pick_masks,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    include_pick_mask: bool = True,
    pick_mask_feature_key: str = "observation.mask_pick",
    pick_mask_mode: Literal["video", "image"] = "image",
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if include_pick_mask:
        # Keep mask as 3-channel image/video because LeRobot image feature expects 3 channels.
        features[pick_mask_feature_key] = {
            "dtype": pick_mask_mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    vision_field: str = "colors",
    *,
    include_pick_mask: bool = True,
    pick_mask_field: str = "masks",
    pick_mask_feature_key: str = "observation.mask_pick",
) -> LeRobotDataset:
    json_dataset = JsonDataset(
        raw_dir,
        robot_type,
        vision_field=vision_field,
        include_pick_mask=include_pick_mask,
        pick_mask_field=pick_mask_field,
    )
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        pick_masks = episode["pick_masks"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            if include_pick_mask and pick_masks is not None:
                frame[pick_mask_feature_key] = pick_masks[i]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    vision_field: Literal["color", "colors", "overlay", "overlays", "depth", "depths", "mask", "masks"] = "colors",
    include_pick_mask: bool = True,
    pick_mask_field: Literal["mask", "masks"] = "masks",
    pick_mask_feature_key: str = "observation.mask_pick",
    pick_mask_mode: Literal["video", "image"] = "image",
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        include_pick_mask=include_pick_mask,
        pick_mask_feature_key=pick_mask_feature_key,
        pick_mask_mode=pick_mask_mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        vision_field=vision_field,
        include_pick_mask=include_pick_mask,
        pick_mask_field=pick_mask_field,
        pick_mask_feature_key=pick_mask_feature_key,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
