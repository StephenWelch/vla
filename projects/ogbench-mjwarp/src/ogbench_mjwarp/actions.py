"""Explicit recording/checkpoint contracts for native and absolute joint actions."""

import json
from pathlib import Path

import numpy as np


def joint_profile(base, config):
    return {
        "revision": 1,
        "mode": "joint",
        "fps": 20,
        "names": [base._model.joint(int(i)).name for i in base._arm_joint_ids]
        + ["gripper.closure"],
        "units": ["radian"] * 6 + ["normalized"],
        "bounds": base._model.actuator_ctrlrange[base._arm_actuator_ids].tolist()
        + [[0, 1]],
        "max_velocity": config.max_velocity,
        "max_acceleration": config.max_acceleration,
    }


def validate_profile(profile):
    if (
        not isinstance(profile, dict)
        or profile.get("mode") != "joint"
        or profile.get("revision") != 1
    ):
        raise ValueError("Expected revision 1 absolute joint action profile")
    bounds = np.asarray(profile.get("bounds"), dtype=float)
    names = profile.get("names", [])
    if len(names) != 7 or len(set(names)) != 7 or names[-1] != "gripper.closure":
        raise ValueError("Joint dataset requires an explicit seven-action profile")
    if (
        bounds.shape != (7, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("Joint action bounds must be finite ordered [7, 2]")
    if (
        profile.get("units") != ["radian"] * 6 + ["normalized"]
        or profile.get("fps") != 20
        or not np.array_equal(bounds[-1], [0, 1])
    ):
        raise ValueError(
            "Joint profile requires radians, normalized closure and native 20 Hz"
        )
    limits = [profile.get(key, 0) for key in ("max_velocity", "max_acceleration")]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("Joint controller limits must be finite and positive")
    return profile


def action_profile(dataset, checkpoint=None):
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    profile = manifest.get("action_profile")
    if manifest.get("format") == "ogbench-mjwarp-3":
        validate_profile(profile)
    elif profile is not None:
        raise ValueError("Joint action profile requires dataset format v3")
    if checkpoint is not None:
        saved = Path(checkpoint) / "action_profile.json"
        actual = json.loads(saved.read_text()) if saved.exists() else None
        if actual != profile:
            raise ValueError("Checkpoint and dataset action profiles differ")
    return profile
