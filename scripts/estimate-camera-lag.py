"""Estimate OpenCV frame-read lag from a supervised wrist-roll recording."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from camera_sync import CAMERA, frame_times, load_timing, state_times


def image_rotation(first, second):
    a = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(a, maxCorners=300, qualityLevel=0.02, minDistance=8)
    if points is None or len(points) < 25:
        raise ValueError("Scene has too few stationary visual features")
    moved, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
    good = status.ravel() == 1
    if good.sum() < 20:
        raise ValueError("Camera motion could not be tracked")
    affine, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC)
    if affine is None or inliers.sum() < 15:
        raise ValueError("Camera motion could not be fit")
    return np.arctan2(affine[1, 0], affine[0, 0])


def estimate_lag(image_times, encoder_times, encoder_angles, rotations):
    if image_times[-1] - image_times[0] < 8 or np.ptp(encoder_angles) < 20:
        raise ValueError("Use at least 8 seconds of varied wrist motion spanning 20 degrees")
    if np.any(np.diff(image_times) <= 0) or np.any(np.diff(encoder_times) <= 0):
        raise ValueError("Diagnostic recording has repeated or unordered samples")
    camera_velocity = rotations / np.diff(image_times)
    encoder_velocity = np.gradient(encoder_angles, encoder_times)
    midpoints = (image_times[1:] + image_times[:-1]) / 2
    candidates = np.arange(0, 0.251, 0.002)
    scored = []
    for lag in candidates:
        valid = (midpoints - lag >= encoder_times[0]) & (midpoints - lag <= encoder_times[-1])
        if valid.sum() < 100:
            scored.append(-np.inf)
            continue
        reference = np.interp(midpoints[valid] - lag, encoder_times, encoder_velocity)
        scored.append(float(np.corrcoef(camera_velocity[valid], reference)[0, 1] ** 2))
    best = int(np.nanargmax(scored))
    if not np.isfinite(scored[best]) or scored[best] < 0.35 or best in (0, len(candidates) - 1):
        raise ValueError("Camera/encoder motion correlation is too weak or outside 0-250 ms")
    plausible = candidates[np.asarray(scored) >= scored[best] - 0.02]
    return {"lag_ms": round(float(candidates[best] * 1000), 1),
            "uncertainty_ms": round(float(max(candidates[best] - plausible[0],
                                              plausible[-1] - candidates[best]) * 1000 + 1000 / 30), 1),
            "correlation_r2": round(float(scored[best]), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="telegrip/episode_data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = LeRobotDataset(args.repo_id, root=args.source, video_backend="pyav")
    if dataset.fps != 30:
        raise ValueError("Expected a 30 fps diagnostic episode")
    timing, _ = load_timing(args.source, args.episode)
    episode = dataset.meta.episodes[args.episode]
    start, end = episode["dataset_from_index"], episode["dataset_to_index"]
    if end - start != len(timing):
        raise ValueError("Timing sidecar does not match episode video")
    frames = [dataset[i] for i in range(start, end)]
    images = [(frame[f"observation.images.{CAMERA}"].permute(1, 2, 0).numpy() * 255)
              .clip(0, 255).astype(np.uint8) for frame in frames]
    rotations = np.asarray([image_rotation(a, b) for a, b in zip(images, images[1:])])
    names = dataset.features["observation.state"]["names"]
    roll = names.index("left_wrist_roll.pos")
    angles = np.asarray([frame["observation.state"][roll].item() for frame in frames])
    result = estimate_lag(frame_times(timing), state_times(timing), angles, rotations)
    height, width = images[0].shape[:2]
    result.update({"camera": CAMERA, "width": width, "height": height, "fps": dataset.fps,
                   "method": "wrist_roll_image_encoder_correlation",
                   "source": str(args.source.resolve()), "episode": args.episode,
                   "created_utc": datetime.now(timezone.utc).isoformat()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
