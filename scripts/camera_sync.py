"""Timing helpers shared by the left-wrist lag estimator and dataset aligner."""

import json
from pathlib import Path

import numpy as np


CAMERA = "left_wrist_cam"


def load_timing(root: Path, episode: int):
    directory = Path(root) / "meta" / "telegrip_annotations"
    prefix = f"episode_{episode:06d}"
    frames = json.loads((directory / f"{prefix}_timing.json").read_text(encoding="utf-8"))
    commands = json.loads((directory / f"{prefix}_commands.json").read_text(encoding="utf-8"))
    if not frames or not commands:
        raise ValueError(f"Episode {episode} has no complete timing data")
    if [row["frame"] for row in frames] != list(range(len(frames))):
        raise ValueError(f"Episode {episode} has a frame-index gap")
    for row in frames:
        if CAMERA not in row.get("camera_capture_times", {}) or "left" not in row.get("state_read_times", {}):
            raise ValueError(f"Episode {episode} lacks camera or encoder timing")
    return frames, commands


def frame_times(frames):
    return np.asarray([row["camera_capture_times"][CAMERA] for row in frames], dtype=float)


def state_times(frames):
    return np.asarray([np.mean(row["state_read_times"]["left"]) for row in frames], dtype=float)


def interpolate_state(times, states, target):
    if target < times[0] or target > times[-1]:
        raise ValueError("Image time lies outside measured encoder samples")
    return np.asarray([np.interp(target, times, states[:, joint])
                       for joint in range(states.shape[1])], dtype=np.float32)


def active_command(commands, target):
    events = [event for event in commands if event["end_time"] <= target]
    if not events:
        raise ValueError("No command was active at image time")
    return np.asarray(events[-1]["action"], dtype=np.float32)


def validate_calibration(calibration, width, height, fps):
    expected = {"camera": CAMERA, "width": width, "height": height, "fps": fps}
    if any(calibration.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Camera calibration does not match {expected}")
    lag = calibration.get("lag_ms")
    if not isinstance(lag, (float, int)) or not 0 <= lag <= 250:
        raise ValueError("Camera lag must be between 0 and 250 ms")
    return lag / 1000
