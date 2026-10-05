"""Build a separate, timestamp-aligned LeRobot dataset from timed Telegrip episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from camera_sync import (CAMERA, active_command, frame_times, interpolate_state,
                         load_timing, state_times, validate_calibration)


def nearest_indices(times, targets, tolerance):
    if np.any(np.diff(times) < 0):
        raise ValueError("Camera timestamps are out of order")
    choices = []
    for target in targets:
        right = int(np.searchsorted(times, target))
        options = [index for index in (right - 1, right) if 0 <= index < len(times)]
        chosen = min(options, key=lambda index: abs(times[index] - target))
        if abs(times[chosen] - target) > tolerance:
            raise ValueError("Camera gap exceeds one 30 fps frame")
        choices.append(chosen)
    return choices


def align_episode(source, index, timing, commands, lag, target):
    episode = source.meta.episodes[index]
    start, end = episode["dataset_from_index"], episode["dataset_to_index"]
    if end - start != len(timing):
        raise ValueError(f"Episode {index}: timing row count differs from video")
    captures = frame_times(timing)
    observations = captures - lag
    measured_at = state_times(timing)
    if np.any(np.diff(measured_at) <= 0) or np.max(np.diff(measured_at)) > 0.1:
        raise ValueError(f"Episode {index}: encoder samples have a gap")
    frames = [source[i] for i in range(start, end)]
    states = np.asarray([frame["observation.state"].numpy() for frame in frames])
    if states.shape[1] != 6:
        raise ValueError("Expected six measured left-arm joints")
    first_command = min(event["end_time"] for event in commands)
    valid = ((observations >= measured_at[0]) &
             (observations >= first_command) &
             (observations <= measured_at[-1]))
    if valid.sum() < 2:
        raise ValueError(f"Episode {index}: no camera frames bracketed by encoder samples")
    first, last = observations[valid][0], observations[valid][-1]
    grid = first + np.arange(int(np.floor((last - first) * source.fps)) + 1) / source.fps
    choices = nearest_indices(observations, grid, 1 / source.fps)
    task = episode["tasks"]
    if len(task) != 1:
        raise ValueError(f"Episode {index}: expected one task")
    provenance = []
    previous = None
    for output_index, source_index in enumerate(choices):
        observed_at = float(observations[source_index])
        if observed_at < measured_at[0] or observed_at > measured_at[-1]:
            raise ValueError(f"Episode {index}: image lies outside encoder samples")
        action = active_command(commands, observed_at)
        image = frames[source_index][f"observation.images.{CAMERA}"]
        image = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        target.add_frame({
            "observation.state": interpolate_state(measured_at, states, observed_at),
            "action": action,
            f"observation.images.{CAMERA}": image,
            "task": task[0],
        })
        provenance.append({"frame": output_index, "source_frame": source_index,
                           "grid_time": float(grid[output_index]),
                           "estimated_observation_time": observed_at,
                           "camera_read_time": float(captures[source_index]),
                           "repeated_source_frame": source_index == previous})
        previous = source_index
    target.save_episode(parallel_encoding=False)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--repo-id", default="telegrip/episode_data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    source = LeRobotDataset(args.repo_id, root=args.source, video_backend="pyav")
    if source.fps != 30:
        raise ValueError("Expected 30 fps source dataset")
    image_feature = source.features[f"observation.images.{CAMERA}"]
    images = [key for key in source.features if key.startswith("observation.images.")]
    if images != [f"observation.images.{CAMERA}"]:
        raise ValueError("Expected only the left wrist camera")
    height, width = image_feature["shape"][:2]
    lag = validate_calibration(calibration, width, height, source.fps)
    features = {key: value for key, value in source.features.items()
                if key == "action" or key.startswith("observation.")}
    target = None
    manifest = {"source": str(args.source.resolve()), "calibration": calibration,
                "episodes": []}
    try:
        for index in range(source.num_episodes):
            directory = args.source / "meta" / "telegrip_annotations"
            prefix = f"episode_{index:06d}"
            timing_exists = (directory / f"{prefix}_timing.json").exists()
            commands_exists = (directory / f"{prefix}_commands.json").exists()
            if timing_exists != commands_exists:
                raise ValueError(f"Episode {index} has an incomplete timing sidecar")
            try:
                timing, commands = load_timing(args.source, index)
            except FileNotFoundError:
                print(f"Skipping legacy episode {index}: no timing sidecar")
                continue
            if target is None:
                target = LeRobotDataset.create(args.repo_id, fps=source.fps, features=features,
                                               root=args.output, robot_type="telegrip",
                                               video_backend="pyav")
            output_index = target.num_episodes
            provenance = align_episode(source, index, timing, commands, lag, target)
            directory = args.output / "meta" / "telegrip_annotations"
            directory.mkdir(parents=True, exist_ok=True)
            source_annotation = (args.source / "meta" / "telegrip_annotations" /
                                 f"episode_{index:06d}.json")
            if source_annotation.exists():
                annotation = json.loads(source_annotation.read_text(encoding="utf-8"))
                annotation["episode_index"] = output_index
                source_mistakes = set(annotation.get("mistake_frames", []))
                annotation["mistake_frames"] = sorted({
                    min(provenance, key=lambda row: abs(row["source_frame"] - old))["frame"]
                    for old in source_mistakes
                })
                annotation["mistake"] = bool(annotation.get("mistake") or source_mistakes)
                annotation["source_duration_seconds"] = annotation.get("duration_seconds")
                annotation["duration_seconds"] = len(provenance) / source.fps
                (directory / f"episode_{output_index:06d}.json").write_text(
                    json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
            manifest["episodes"].append({"output_episode": output_index,
                                         "source_episode": index, "frames": provenance})
            print(f"Aligned episode {index} -> {output_index}: {len(provenance)} frames")
        if target is None:
            raise ValueError("No episodes with timing sidecars were found")
    finally:
        if target is not None:
            target.finalize()
    (args.output / "meta" / "telegrip_annotations" / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checked = LeRobotDataset(args.repo_id, root=args.output, video_backend="pyav")
    assert checked.num_episodes == len(manifest["episodes"])
    assert len(checked) == sum(len(row["frames"]) for row in manifest["episodes"])


if __name__ == "__main__":
    main()
