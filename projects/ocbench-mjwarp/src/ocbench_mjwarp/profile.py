"""Reject action, camera and simulator provenance mismatches."""

import json
from pathlib import Path

from .config import validate_action


def profiles(dataset, checkpoint=None):
    root = Path(dataset)
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Incomplete dataset")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "ocbench-mjwarp-1":
        raise ValueError("Expected an OCBench dataset")
    validate_action(manifest["action_profile"])
    if checkpoint:
        for filename, key in (
            ("action_profile.json", "action_profile"),
            ("rendering.json", "rendering"),
        ):
            p = Path(checkpoint) / filename
            if not p.exists() or json.loads(p.read_text()) != manifest[key]:
                raise ValueError(f"Checkpoint {key} differs from OCBench dataset")
    return manifest["rendering"], manifest["action_profile"]
