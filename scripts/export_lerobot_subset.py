"""Export a random episode subset from a LeRobot dataset into a new local LeRobot repo."""

import json
import pathlib
import shutil
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

DEFAULT_DATASET_KEYS = {
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
    "next.done",
}


def main(
    source_repo_id: str = "physical-intelligence/libero",
    *,
    source_root: str | None = None,
    output_repo_id: str = "local/libero_overfit",
    output_root: str | None = None,
    num_episodes: int = 50,
    seed: int = 7,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> None:
    source_dataset = LeRobotDataset(
        source_repo_id,
        root=pathlib.Path(source_root).expanduser().resolve() if source_root is not None else None,
    )
    total_episodes = int(source_dataset.meta.total_episodes)
    if num_episodes > total_episodes:
        raise ValueError(f"Requested {num_episodes} episodes, but source dataset only has {total_episodes}")

    episode_rng = np.random.default_rng(seed)
    selected_episodes = sorted(int(idx) for idx in episode_rng.choice(total_episodes, size=num_episodes, replace=False))

    output_root_path = pathlib.Path(output_root).expanduser().resolve() if output_root is not None else HF_LEROBOT_HOME
    output_dataset_path = output_root_path / output_repo_id
    if output_dataset_path.exists():
        shutil.rmtree(output_dataset_path)

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_dataset_path,
        robot_type=_dataset_robot_type(source_dataset),
        fps=_dataset_fps(source_dataset),
        features=_strip_default_features(source_dataset.features),
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    feature_keys = [key for key in source_dataset.features if key not in DEFAULT_DATASET_KEYS]

    for episode_index in selected_episodes:
        start = int(source_dataset.episode_data_index["from"][episode_index])
        stop = int(source_dataset.episode_data_index["to"][episode_index])

        for sample_idx in range(start, stop):
            item = source_dataset[sample_idx]
            frame = {key: item[key] for key in feature_keys if key in item}
            if "task" in item:
                frame["task"] = item["task"]
            elif "task_index" in item:
                frame["task"] = source_dataset.meta.tasks.get(int(item["task_index"]), "")
            output_dataset.add_frame(frame)
        output_dataset.save_episode()

    if hasattr(output_dataset, "consolidate"):
        output_dataset.consolidate()
    if hasattr(output_dataset, "stop_image_writer"):
        output_dataset.stop_image_writer()

    manifest = {
        "source_repo_id": source_repo_id,
        "output_repo_id": output_repo_id,
        "output_dataset_path": str(output_dataset_path),
        "num_episodes": num_episodes,
        "selected_episode_indices": selected_episodes,
    }
    manifest_path = output_dataset_path / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote subset dataset to: {output_dataset_path}")
    print(f"Wrote manifest to: {manifest_path}")


def _strip_default_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key not in DEFAULT_DATASET_KEYS}


def _dataset_fps(dataset: LeRobotDataset) -> int:
    fps = getattr(dataset, "fps", None)
    if fps is not None:
        return int(fps)

    info = getattr(dataset.meta, "info", None)
    if isinstance(info, dict):
        return int(info["fps"])
    if hasattr(info, "fps"):
        return int(info.fps)
    raise ValueError("Could not determine dataset fps from LeRobot metadata")


def _dataset_robot_type(dataset: LeRobotDataset) -> str:
    robot_type = getattr(dataset.meta, "robot_type", None)
    if robot_type is not None:
        return str(robot_type)

    info = getattr(dataset.meta, "info", None)
    if isinstance(info, dict):
        return str(info.get("robot_type", "unknown"))
    if hasattr(info, "robot_type"):
        return str(info.robot_type)
    return "unknown"


if __name__ == "__main__":
    tyro.cli(main)
