"""Audit a LeRobot dataset and export a few visual spot-check samples."""

from collections import Counter
import json
import pathlib
from typing import Any

import imageio
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
    repo_id: str,
    *,
    root: str | None = None,
    output_dir: str = "data/lerobot_audits",
    num_visual_samples: int = 8,
    seed: int = 7,
) -> None:
    dataset_root = pathlib.Path(root).expanduser().resolve() if root is not None else None
    dataset = LeRobotDataset(repo_id, root=dataset_root)

    audit_dir = pathlib.Path(output_dir) / repo_id.replace("/", "__")
    samples_dir = audit_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths = _episode_lengths(dataset)
    frame_indices = np.asarray(_column_to_list(dataset.hf_dataset["frame_index"]), dtype=np.int64)
    episode_indices = np.asarray(_column_to_list(dataset.hf_dataset["episode_index"]), dtype=np.int64)

    action_key = _resolve_action_key(dataset.features)
    actions = np.asarray([_to_numpy(x).reshape(-1) for x in dataset.hf_dataset[action_key]], dtype=np.float32)

    prompts = _extract_prompts(dataset)
    prompt_counter = Counter(prompts)
    non_empty_prompts = [prompt for prompt in prompts if prompt.strip()]

    summary = {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root or (HF_LEROBOT_HOME / repo_id)),
        "num_frames": len(dataset),
        "num_episodes": len(episode_lengths),
        "episode_length": {
            "min": int(np.min(episode_lengths)),
            "max": int(np.max(episode_lengths)),
            "mean": float(np.mean(episode_lengths)),
            "median": float(np.median(episode_lengths)),
            "p95": float(np.percentile(episode_lengths, 95)),
        },
        "frame_index_checks": {
            "starts_at_zero_each_episode": bool(np.all(frame_indices[np.r_[True, np.diff(episode_indices) != 0]] == 0)),
            "increments_within_episode": bool(
                np.all(
                    np.diff(frame_indices)[np.diff(episode_indices) == 0] == 1,
                )
            ),
        },
        "prompts": {
            "empty_count": int(len(prompts) - len(non_empty_prompts)),
            "empty_rate": float((len(prompts) - len(non_empty_prompts)) / len(prompts)) if prompts else 0.0,
            "unique_count": len(set(non_empty_prompts)),
            "repetition_rate": float(1.0 - len(set(non_empty_prompts)) / len(non_empty_prompts))
            if non_empty_prompts
            else 0.0,
            "most_common_prompt": prompt_counter.most_common(1)[0][0] if prompt_counter else "",
            "most_common_prompt_fraction": float(prompt_counter.most_common(1)[0][1] / len(prompts))
            if prompt_counter
            else 0.0,
        },
        "action_key": action_key,
        "action_stats": _summarize_actions(actions),
        "visual_samples_dir": str(samples_dir),
    }

    sample_records = _export_visual_samples(
        dataset,
        samples_dir=samples_dir,
        num_samples=min(num_visual_samples, len(dataset)),
        seed=seed,
    )
    summary["visual_samples"] = sample_records

    summary_path = audit_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote audit summary to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _episode_lengths(dataset: LeRobotDataset) -> np.ndarray:
    episodes = dataset.meta.episodes
    if isinstance(episodes, dict):
        records = [episodes[k] for k in sorted(episodes)]
    else:
        records = list(episodes)
    return np.asarray([record["length"] for record in records], dtype=np.int64)


def _resolve_action_key(features: dict[str, Any]) -> str:
    for key in ("actions", "action"):
        if key in features:
            return key
    raise ValueError(f"Could not find an action feature in dataset features: {sorted(features)}")


def _extract_prompts(dataset: LeRobotDataset) -> list[str]:
    tasks = getattr(dataset.meta, "tasks", {})
    if "task_index" in dataset.hf_dataset.features:
        task_indices = _column_to_list(dataset.hf_dataset["task_index"])
        return [str(tasks.get(int(task_index), "")).strip() for task_index in task_indices]

    prompts = []
    if isinstance(dataset.meta.episodes, dict):
        episode_records = [dataset.meta.episodes[k] for k in sorted(dataset.meta.episodes)]
    else:
        episode_records = list(dataset.meta.episodes)
    for record in episode_records:
        prompt = ""
        if isinstance(record, dict):
            prompt = ", ".join(record.get("tasks", []))
        prompts.extend([prompt] * int(record["length"]))
    return prompts


def _summarize_actions(actions: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "per_dim": [
            {
                "dim": dim,
                "min": float(np.min(actions[:, dim])),
                "max": float(np.max(actions[:, dim])),
                "mean": float(np.mean(actions[:, dim])),
                "std": float(np.std(actions[:, dim])),
                "q01": float(np.quantile(actions[:, dim], 0.01)),
                "q99": float(np.quantile(actions[:, dim], 0.99)),
            }
            for dim in range(actions.shape[1])
        ],
    }


def _export_visual_samples(
    dataset: LeRobotDataset,
    *,
    samples_dir: pathlib.Path,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(dataset), size=num_samples, replace=False)
    sample_records = []

    for sample_num, sample_idx in enumerate(sorted(int(idx) for idx in sample_indices)):
        item = dataset[sample_idx]
        sample_dir = samples_dir / f"sample_{sample_num:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "sample_index": sample_idx,
            "episode_index": int(_maybe_scalar(item.get("episode_index", -1))),
            "frame_index": int(_maybe_scalar(item.get("frame_index", -1))),
            "task": str(item.get("task", "")).strip(),
            "action": _to_numpy(item[_resolve_action_key(dataset.features)]).reshape(-1).tolist(),
            "images": [],
        }

        for image_key in _resolve_image_keys(dataset, item):
            image = _convert_to_uint8_image(item[image_key])
            image_path = sample_dir / f"{image_key.replace('.', '_').replace('/', '_')}.png"
            imageio.imwrite(image_path, image)
            record["images"].append(str(image_path))

        with (sample_dir / "sample.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        sample_records.append(record)

    return sample_records


def _resolve_image_keys(dataset: LeRobotDataset, item: dict[str, Any]) -> list[str]:
    camera_keys = list(getattr(dataset.meta, "camera_keys", []))
    if camera_keys:
        return [key for key in camera_keys if key in item]

    image_keys = []
    for key in dataset.features:
        if key in DEFAULT_DATASET_KEYS or key not in item:
            continue
        value = _to_numpy(item[key])
        if value.ndim == 3 and (value.shape[0] in (1, 3) or value.shape[-1] in (1, 3)):
            image_keys.append(key)
    return image_keys


def _column_to_list(column: Any) -> list[Any]:
    return [_maybe_scalar(value) for value in column]


def _maybe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _convert_to_uint8_image(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected an image array with 3 dimensions, got shape {image.shape}")
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image


if __name__ == "__main__":
    tyro.cli(main)
