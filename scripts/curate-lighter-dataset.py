"""Create local LeRobot train/validation datasets of successful lighter lifts."""

import argparse
import json
from pathlib import Path

import pandas as pd


POSITIONS = "ABCDE"
TASK = "Grasp the green lighter by its body and lift it at least 3 cm for one second."


def select_balanced(rows, count):
    buckets = {position: [] for position in POSITIONS}
    for row in rows:
        buckets[row["position_id"]].append(row)
    selected = []
    while len(selected) < count:
        available = [position for position in POSITIONS if buckets[position]]
        if not available:
            raise ValueError(f"Need {count} episodes, found only {len(selected)}")
        for position in available:
            selected.append(buckets[position].pop(0))
            if len(selected) == count:
                break
    return selected


def choose_splits(annotations):
    clean, recovery = [], []
    for row in annotations:
        if row.get("outcome") != "success":
            continue
        position = row.get("position_id")
        if not isinstance(position, str) or position not in POSITIONS:
            continue  # Older successes need a position before they count toward the quota.
        (recovery if row.get("mistake") else clean).append(row)
    clean = select_balanced(clean, 50)
    recovery = select_balanced(recovery, 10)

    # Two clean held-out examples from each position, plus two recoveries
    # from different positions when the collected data permit it.
    clean_val = select_balanced(clean, 10)
    recovery_val = select_balanced(recovery, 2)
    val_ids = {row["episode_index"] for row in clean_val + recovery_val}
    train = sorted((row for row in clean + recovery if row["episode_index"] not in val_ids),
                   key=lambda row: row["episode_index"])
    val = sorted(clean_val + recovery_val, key=lambda row: row["episode_index"])
    if len(train) != 48 or len(val) != 12:
        raise AssertionError("Expected 48 train and 12 validation episodes")
    return {"train": train, "val": val}


def check_selected_tasks(source, splits):
    """Only relabel empty tasks; never silently change another instruction."""
    for rows in splits.values():
        for row in rows:
            index = row["episode_index"]
            tasks = source.meta.episodes[index]["tasks"]
            if len(tasks) != 1 or tasks[0] not in ("", TASK):
                raise ValueError(f"Episode {index} has a different recorded task: {tasks}")


def normalize_split_task(root):
    """Make frames and LeRobot metadata agree on the intended task."""
    root = Path(root)
    for path in (root / "data").rglob("*.parquet"):
        frames = pd.read_parquet(path)
        frames["task_index"] = 0
        frames.to_parquet(path, index=False)
    for path in (root / "meta" / "episodes").rglob("*.parquet"):
        episodes = pd.read_parquet(path)
        episodes["tasks"] = [[TASK] for _ in range(len(episodes))]
        for column in episodes.columns:
            if column.startswith("stats/task_index/") and not column.endswith("/count"):
                episodes[column] = [[0.0] for _ in range(len(episodes))]
        episodes.to_parquet(path, index=False)
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
    tasks.to_parquet(root / "meta" / "tasks.parquet")
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for name in stats.get("task_index", {}):
            if name != "count":
                stats["task_index"][name] = [0.0]
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = 1
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def verify_split_task(dataset):
    if list(dataset.meta.tasks.index) != [TASK] or dataset.meta.total_tasks != 1:
        raise ValueError(f"Task metadata is inconsistent in {dataset.root}")
    if any(episode["tasks"] != [TASK] for episode in dataset.meta.episodes):
        raise ValueError(f"Episode tasks are inconsistent in {dataset.root}")
    for path in (dataset.root / "data").rglob("*.parquet"):
        if not (pd.read_parquet(path, columns=["task_index"])["task_index"] == 0).all():
            raise ValueError(f"Frame tasks are inconsistent in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="telegrip/episode_data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not (args.source / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Local LeRobot dataset not found: {args.source}")

    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(args.repo_id, root=args.source, video_backend="pyav")
    features = source.features
    if tuple(features["observation.state"]["shape"]) != (6,) or tuple(features["action"]["shape"]) != (6,):
        raise ValueError("Expected a six-joint left-arm dataset")
    if features["observation.state"].get("names") != features["action"].get("names"):
        raise ValueError("Observation and action joint ordering differs")
    images = [key for key in features if key.startswith("observation.images.")]
    if images != ["observation.images.left_wrist_cam"]:
        raise ValueError(f"Expected only the left wrist camera; found {images}")
    if source.fps != 30:
        raise ValueError(f"Expected 30 FPS, found {source.fps}")

    annotation_dir = args.source / "meta" / "telegrip_annotations"
    annotations = []
    for index in range(source.num_episodes):
        path = annotation_dir / f"episode_{index:06d}.json"
        row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"episode_index": index}
        if row.get("episode_index") != index:
            raise ValueError(f"Incorrect index in {path}")
        annotations.append(row)
    splits = choose_splits(annotations)
    check_selected_tasks(source, splits)

    # Decode representative frames before copying many videos. LeRobot checks
    # the full metadata and video segments while constructing each split.
    for row in splits["train"] + splits["val"]:
        index = row["episode_index"]
        first = source.meta.episodes["dataset_from_index"][index]
        end = source.meta.episodes["dataset_to_index"][index]
        for frame_index in {first, (first + end - 1) // 2, end - 1}:
            frame = source[frame_index]
            state, action, image = frame["observation.state"], frame["action"], frame[images[0]]
            if (state.numel() != 6 or action.numel() != 6 or
                    not state.isfinite().all() or not action.isfinite().all()):
                raise ValueError(f"Bad vector frame in episode {index}")
            if image.numel() == 0 or float(image.float().mean()) < 0.01:
                raise ValueError(f"Missing or nearly black wrist frame in episode {index}")

    indices = {name: [row["episode_index"] for row in rows] for name, rows in splits.items()}
    created = split_dataset(source, indices, output_dir=args.output)
    for name in created:
        root = args.output / name
        normalize_split_task(root)
        verify_split_task(LeRobotDataset(args.repo_id + f"_{name}", root=root, video_backend="pyav"))
    manifest = {"task": TASK, "source": str(args.source.resolve()), "source_repo_id": args.repo_id,
                "splits": {name: [{"source_episode": row["episode_index"],
                                   "position_id": row["position_id"], "recovery": bool(row.get("mistake"))}
                                  for row in rows] for name, rows in splits.items()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, dataset in created.items():
        print(f"{name}: {dataset.num_episodes} episodes at {dataset.root} ({dataset.repo_id})")


if __name__ == "__main__":
    main()
