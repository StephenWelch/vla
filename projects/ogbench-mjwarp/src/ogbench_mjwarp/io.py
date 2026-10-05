"""Shared atomic metadata I/O for recording and dataset tools."""

import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def episode_metadata(root, outcome="all", require_contact_valid=False):
    """Read and select raw or exported episodes, preserving their stored order."""
    root = Path(root)
    manifest = root / "manifest.json"
    rows = (
        json.loads(manifest.read_text())["episodes"]
        if manifest.exists()
        else [
            json.loads(path.read_text()) for path in sorted(root.glob("episode-*.json"))
        ]
    )
    return [
        row
        for row in rows
        if (outcome == "all" or row["outcome"] == outcome)
        and (
            not require_contact_valid
            or row.get("contact_quality", {}).get("valid") is True
        )
    ]


def rollout_path(root, episode):
    """Resolve raw episode IDs or exported LeRobot episode indices."""
    root = Path(root)
    manifest = root / "manifest.json"
    index, archive = (
        ("episode_index", "replay") if manifest.exists() else ("episode_id", "archive")
    )
    for row in episode_metadata(root):
        if row[index] == episode:
            return row, root / row[archive]
    raise ValueError(f"Episode {episode} was not found in {root}")


def load_sim_states(archive):
    """Decompress simulator arrays once, skipping images, and check frame alignment."""
    with np.load(archive, allow_pickle=False) as data:
        states = {key[4:]: data[key] for key in data.files if key.startswith("sim/")}
    frames = len(states.get("qpos", []))
    if not frames or any(len(value) != frames for value in states.values()):
        raise ValueError(
            "Rollout needs aligned simulator snapshots, including terminal state"
        )
    return states


@lru_cache(maxsize=1)
def versions():
    result = {
        name: importlib.metadata.version(name)
        for name in ("ogbench", "mujoco", "mujoco-warp", "warp-lang", "torch")
    }
    from . import __version__

    result["ogbench-mjwarp"] = __version__
    direct = importlib.metadata.distribution("ogbench").read_text("direct_url.json")
    result["ogbench_revision"] = (
        json.loads(direct).get("vcs_info", {}).get("commit_id", "unknown")
        if direct
        else "unknown"
    )
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    result["integration_source_sha256"] = digest.hexdigest()
    return result
