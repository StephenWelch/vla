"""Import pinned OCBench state trajectories without replaying their physics."""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from vla_tools.tracking import write_json

from .config import ACTION, TASK


@dataclass
class HubConfig:
    task: str = TASK
    cache: Path = Path.home() / ".cache/ocbench/datasets"
    shards: int = 1
    train_episodes: int = 100
    val_episodes: int = 20


@dataclass(kw_only=True)
class ImportConfig(HubConfig):
    output: Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def boundaries(terminals, metadata):
    ends = np.flatnonzero(terminals) + 1
    offsets = np.r_[0, ends]
    if not len(ends) or ends[-1] != len(terminals) or len(ends) != len(metadata):
        raise ValueError("Invalid episode boundaries or metadata count")
    for i, row in enumerate(metadata):
        start, end = map(int, offsets[i : i + 2])
        if (row["start"], row["end"], row["length"]) != (start, end - 1, end - start):
            raise ValueError("Metadata and transition boundaries differ")
    return offsets


def convert_episode(env, arrays):
    """Produce N pre-action observations; never synthesize a terminal state."""
    import mujoco

    model, data = env.unwrapped.model, env.unwrapped.data
    base = env.unwrapped
    q, v, action = arrays["qpos"], arrays["qvel"], arrays["actions"]
    n = len(action)
    if q.shape != (n, model.nq) or v.shape != (n, model.nv) or action.shape != (n, 7):
        raise ValueError("Unexpected stacking state/action dimensions")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite imported trajectory")
    if any(len(arrays[k]) != n for k in ("rewards", "masks", "terminals")):
        raise ValueError("Transition arrays have different lengths")
    if (
        not np.isin(arrays["masks"], [0, 1]).all()
        or not np.isin(arrays["rewards"], [0, 1]).all()
    ):
        raise ValueError("Expected binary stacking rewards and masks")
    state = np.empty((n, 18), np.float32)
    arm_q = model.jnt_qposadr[base._arm_joint_ids]
    arm_v = model.jnt_dofadr[base._arm_joint_ids]
    grip = model.jnt_qposadr[base._gripper_opening_joint_id]
    grip_v = model.jnt_dofadr[base._gripper_opening_joint_id]
    for t in range(n):
        data.qpos[:], data.qvel[:] = q[t], v[t]
        mujoco.mj_kinematics(model, data)
        mat = data.site_xmat[base._pinch_site_id].reshape(3, 3)
        state[t] = np.r_[
            q[t, arm_q],
            v[t, arm_v],
            data.site_xpos[base._pinch_site_id],
            np.arctan2(mat[1, 0], mat[0, 0]),
            q[t, grip] / 0.8,
            v[t, grip_v] / 0.8,
        ]
    return {
        "action": action,
        "state": state,
        "success": arrays["rewards"].astype(bool),
        "reward": arrays["rewards"],
        "done": arrays["terminals"].astype(bool),
        "terminated": arrays["masks"] == 0,
        "truncated": arrays["terminals"].astype(bool) & (arrays["masks"] == 1),
        "sim/qpos": q,
        "sim/qvel": v,
        "sim/ctrl": np.zeros((n, model.nu), np.float32),
        "sim/mocap_pos": np.repeat(data.mocap_pos[None], n, axis=0).astype(np.float32),
        "sim/mocap_quat": np.repeat(data.mocap_quat[None], n, axis=0).astype(
            np.float32
        ),
        "sim/time": (np.arange(n) / 50).astype(np.float32),
    }


def import_dataset(config):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ocbench
    from ocbench.dataset_utils import DATASET_REPO, DATASET_REVISION

    if config.task != TASK:
        raise ValueError(f"Import currently supports only {TASK}")
    if min(config.shards, config.train_episodes, config.val_episodes) < 1:
        raise ValueError("Positive shard and split episode budgets required")
    root = config.output
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(asdict(config), default=str)) | {
        "repo_id": DATASET_REPO,
        "revision": DATASET_REVISION,
        "format": "ocbench-hub-1",
    }
    path = root / "import.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Import configuration changed; use a fresh output")
    if (root / "collection.json").exists():
        raise ValueError("Cannot import into a generated collection")
    write_json(path, spec)
    (root / "raw").mkdir(exist_ok=True)
    cache = config.cache.expanduser().resolve()
    train, val = ocbench.download_datasets(
        config.task, dataset_root=cache, num_shards=config.shards
    )
    env = ocbench.make("block-cpu-double-task2-v0")
    records = []
    try:
        for split, paths, limit in [
            ("train", train, config.train_episodes),
            ("val", val, config.val_episodes),
        ]:
            count = 0
            for shard, filename in enumerate(paths):
                if count == limit:
                    break
                source = Path(filename)
                meta_path = source.with_name(source.stem + "-metadata.pkl")
                # Only the official, revision-pinned release is accepted here.
                with meta_path.open("rb") as stream:
                    metadata = pickle.load(stream)["episodes"]
                provenance = {
                    "repo_id": DATASET_REPO,
                    "revision": DATASET_REVISION,
                    "file": source.relative_to(cache).as_posix(),
                    "sha256": digest(source),
                    "metadata_sha256": digest(meta_path),
                }
                with (
                    zipfile.ZipFile(source) as archive,
                    tempfile.TemporaryDirectory(dir=cache) as temporary,
                ):
                    keys = ("qpos", "qvel", "actions", "rewards", "masks", "terminals")
                    required = sum(archive.getinfo(k + ".npy").file_size for k in keys)
                    if shutil.disk_usage(cache).free < required + 4 * 2**30:
                        raise RuntimeError(
                            "Insufficient space for memory-mapped shard arrays"
                        )
                    arrays = {}
                    for key in keys:
                        dest = Path(temporary) / (key + ".npy")
                        with archive.open(key + ".npy") as src, dest.open("wb") as out:
                            shutil.copyfileobj(src, out, length=8 * 2**20)
                        arrays[key] = np.load(dest, mmap_mode="r", allow_pickle=False)
                    if any(len(a) != len(arrays["terminals"]) for a in arrays.values()):
                        raise ValueError("Published transition array lengths differ")
                    if len(metadata) >= 1_000_000 or shard >= 1000:
                        raise ValueError("Shard exceeds the replay-ID namespace")
                    offsets = boundaries(arrays["terminals"], metadata)
                    for episode in range(min(len(metadata), limit - count)):
                        start, end = map(int, offsets[episode : episode + 2])
                        # Stable replay IDs, deliberately not claimed to be reset seeds.
                        identity = (
                            (0 if split == "train" else 1_000_000_000)
                            + shard * 1_000_000
                            + episode
                        )
                        filename = f"episode-{identity:010d}"
                        row = {
                            "episode_id": identity,
                            "env_id": TASK,
                            "task_id": 2,
                            "seed": identity,
                            "seed_kind": "replay_id",
                            "reset_seed": None,
                            "archive": filename + ".npz",
                            "length": end - start,
                            "dataset_split": split,
                            "source_kind": "ocbench_hub",
                            "state_alignment": "pre_action",
                            "terminal_state_available": False,
                            "action_profile": ACTION,
                            "native_success": bool(metadata[episode]["success"]),
                            "physical_valid": None,
                            "contact_valid": None,
                            "stable_stack": None,
                            "num_pick_retries": None,
                            "termination_reason": "horizon"
                            if arrays["masks"][end - 1]
                            else "terminated",
                            "source": provenance | {"episode_index": episode},
                            "randomization": {
                                "initial_state_fingerprint": hashlib.sha256(
                                    arrays["qpos"][start].tobytes()
                                    + arrays["qvel"][start].tobytes()
                                ).hexdigest()
                            },
                            "upstream": metadata[episode],
                            "reconstruction": {
                                "ctrl": "zero initialization; original commands unavailable",
                                "mocap": "fixed stacking environment defaults",
                                "time": "frame_index / 50",
                            },
                        }
                        # Metadata may contain NumPy target arrays/scalars.
                        row = json.loads(json.dumps(row, default=lambda x: x.tolist()))
                        record_path = root / "raw" / (filename + ".json")
                        raw_path = root / "raw" / row["archive"]
                        if record_path.exists():
                            saved = json.loads(record_path.read_text())
                            checksum = saved.pop("archive_sha256", None)
                            if (
                                saved != row
                                or not raw_path.exists()
                                or checksum != digest(raw_path)
                            ):
                                raise ValueError(
                                    "Imported episode or source differs on resume"
                                )
                        else:
                            env.reset(seed=0)
                            converted = convert_episode(
                                env, {k: a[start:end] for k, a in arrays.items()}
                            )
                            temporary_path = raw_path.with_suffix(".tmp.npz")
                            np.savez_compressed(temporary_path, **converted)
                            temporary_path.replace(raw_path)
                            row["archive_sha256"] = digest(raw_path)
                            write_json(record_path, row)
                        records.append(row)
                        count += 1
                        write_json(
                            root / "import-status.json",
                            {"status": "importing", "episodes": len(records)},
                        )
                del arrays
            if count != limit:
                raise ValueError(
                    f"Only {count} {split} episodes available; requested {limit}"
                )
    finally:
        env.close()
    write_json(
        root / "import-status.json", {"status": "complete", "episodes": len(records)}
    )
    return {
        "episodes": len(records),
        "train": config.train_episodes,
        "val": config.val_episodes,
    }
