"""LeRobot v3 export, filtered action chunks, and reproducible simulator replay."""

import json
import time
from itertools import chain
from pathlib import Path

import numpy as np

from .io import episode_metadata, load_sim_states, rollout_path, write_json
from .tasks import image_shape

STATE_NAMES = (
    [f"joint_{i}.position" for i in range(6)]
    + [f"joint_{i}.velocity" for i in range(6)]
    + [
        "effector.x",
        "effector.y",
        "effector.z",
        "effector.yaw",
        "gripper.opening",
        "gripper.velocity",
    ]
)
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_yaw", "delta_gripper"]
REFERENCE_NAMES = [
    "effector.x",
    "effector.y",
    "effector.z",
    "effector.yaw",
    "gripper.opening",
]


def export_dataset(
    source,
    output,
    repo_id="local/ogbench-mjwarp",
    outcome="all",
    require_contact_valid=False,
    diverse_per_task=None,
    streaming_encoding=True,
    encoder_queue_size=30,
    encoder_threads=2,
    progress=None,
    quality="all",
):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started = time.perf_counter()
    source_wait_seconds = 0.0
    roots = (
        [Path(source)] if isinstance(source, (str, Path)) else (Path(p) for p in source)
    )
    sources = []
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output already exists; choose a new directory")
    if min(encoder_queue_size, encoder_threads) < 1:
        raise ValueError("Encoder queue size and threads must be positive")

    def episodes():
        nonlocal source_wait_seconds
        iterator = iter(roots)
        while True:
            waiting = time.perf_counter()
            root = next(iterator, None)
            source_wait_seconds += time.perf_counter() - waiting
            if root is None:
                break
            sources.append(root)
            for row in episode_metadata(root, outcome, require_contact_valid, quality):
                if row["length"]:
                    yield root, row

    rows = episodes()
    if diverse_per_task is not None:
        from .diversity import diverse_selection

        rows = iter(diverse_selection(list(rows), diverse_per_task))
    first = next(rows, None)
    if first is None:
        raise ValueError("No completed nonempty episodes match the requested outcome")
    if not first[1].get("record_images", True):
        raise ValueError(
            "LeRobot export requires recorded images; regenerate with record_images=True"
        )
    profile = first[1].get("rendering")
    if not profile:
        raise ValueError(
            "Export requires fresh recordings with one MJWarp rendering profile"
        )
    if profile.get("revision") != 2:
        raise ValueError(
            "OGBench images use a faulty renderer; collect a fresh dataset"
        )
    size, fps = first[1]["image_size"], first[1]["fps"]
    actions = first[1].get("action_profile")
    if actions:
        from .actions import validate_profile

        validate_profile(actions)
    action_names = actions["names"] if actions else ACTION_NAMES
    features = {
        "observation.state": {"dtype": "float32", "shape": (18,), "names": STATE_NAMES},
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for name in ("skill_id", "phase_id", "route_id"):
        features[f"annotation.{name}"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    features["annotation.available"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["annotation.reference"] = {
        "dtype": "float32",
        "shape": (5,),
        "names": REFERENCE_NAMES,
    }
    spline_annotations = first[1].get("annotation_schema_version") == 2
    if spline_annotations:
        features["annotation.target_pose"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }
        features["annotation.task_error"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    for view in ("front", "wrist"):
        features[f"observation.images.{view}"] = {
            "dtype": "video",
            "shape": (3, *image_shape(size)),
            "names": ["channels", "height", "width"],
        }
    if streaming_encoding:
        from .encoding import StreamingDataset

        dataset_type = StreamingDataset
    else:
        dataset_type = LeRobotDataset
    dataset = dataset_type.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type="ogbench_ur5e",
        features=features,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_size,
        encoder_threads=encoder_threads,
        image_writer_threads=0 if streaming_encoding else 4,
    )
    write_json(output / "INCOMPLETE.json", {"completed_episodes": 0})
    manifest = []
    replay_dir = output / "replay"
    replay_dir.mkdir()
    try:
        for index, (source_root, row) in enumerate(chain([first], rows)):
            if (row.get("annotation_schema_version") == 2) != spline_annotations:
                raise ValueError("Export requires one annotation schema")
            if not row.get("record_images", True):
                raise ValueError("LeRobot export requires recorded images")
            if row.get("rendering") != profile:
                raise ValueError("Export requires one MJWarp rendering profile")
            if row.get("action_profile") != actions:
                raise ValueError("Export requires one action profile")
            if (row["image_size"], row["fps"]) != (size, fps):
                raise ValueError(
                    "All episodes must have the same image size and frame rate"
                )
            row = row | {
                "randomization": row.get(
                    "randomization",
                    {
                        "schema_version": 1,
                        "available": False,
                        "reason": "legacy recording without randomization provenance",
                    },
                )
            }
            with np.load(source_root / row["archive"], allow_pickle=False) as archive:
                arrays = {
                    key: archive[key]
                    for key in (
                        "state",
                        "action",
                        "front",
                        "wrist",
                        "success",
                        "done",
                        "truncated",
                    )
                }
                annotated = row["randomization"].get("available") is True and all(
                    f"annotation/{name}" in archive
                    for name in ("skill_id", "phase_id", "route_id", "reference")
                )
                if row["randomization"].get("available") is True and not annotated:
                    raise ValueError("Annotated episode is missing frame annotations")
                annotations = (
                    {
                        name: archive[f"annotation/{name}"]
                        for name in ("skill_id", "phase_id", "route_id", "reference")
                    }
                    if annotated
                    else {}
                )
                if spline_annotations:
                    annotations.update(
                        {
                            name: archive[f"annotation/{name}"]
                            for name in ("target_pose", "task_error")
                        }
                    )
                for tick in range(row["length"]):
                    frame = {
                        "observation.state": arrays["state"][tick].astype(np.float32),
                        "action": arrays["action"][tick].astype(np.float32),
                        "task": row["instruction"],
                    }
                    frame["annotation.available"] = np.array([annotated], dtype=bool)
                    for name in ("skill_id", "phase_id", "route_id"):
                        frame[f"annotation.{name}"] = np.array(
                            [annotations[name][tick] if annotated else -1],
                            dtype=np.int64,
                        )
                    frame["annotation.reference"] = (
                        annotations["reference"][tick].astype(np.float32)
                        if annotated
                        else np.zeros(5, dtype=np.float32)
                    )
                    if spline_annotations:
                        frame["annotation.target_pose"] = annotations["target_pose"][
                            tick
                        ].astype(np.float32)
                        frame["annotation.task_error"] = np.array(
                            [annotations["task_error"][tick]], dtype=bool
                        )
                    for view in ("front", "wrist"):
                        frame[f"observation.images.{view}"] = arrays[view][tick]
                    for key in ("success", "done", "truncated"):
                        frame[f"next.{key}"] = np.asarray(
                            [arrays[key][tick]], dtype=bool
                        )
                    dataset.add_frame(frame)
                dataset.save_episode()
                replay_path = replay_dir / f"episode-{index:06d}.npz"
                np.savez_compressed(
                    replay_path,
                    **{
                        key: archive[key]
                        for key in archive.files
                        if key.startswith(("sim/", "annotation/")) or key == "action"
                    },
                )
            manifest.append(
                row
                | {
                    "source_episode_id": row["episode_id"],
                    "source_root": str(source_root),
                    "episode_index": index,
                    "replay": replay_path.relative_to(output).as_posix(),
                }
            )
            if progress:
                progress(
                    {
                        "event": "export",
                        "exported_episodes": len(manifest),
                        "exported_frames": sum(r["length"] for r in manifest),
                        "export_seconds": time.perf_counter() - started,
                    }
                )
        dataset.finalize()
    except BaseException:
        # Keep the partial output available for diagnosis, but mark it unusable.
        write_json(output / "INCOMPLETE.json", {"completed_episodes": len(manifest)})
        dataset.writer.close_writer()
        raise
    write_json(
        output / "manifest.json",
        {
            "format": "ogbench-mjwarp-3" if actions else "ogbench-mjwarp-2",
            **({"action_profile": actions} if actions else {}),
            "rendering": profile,
            "randomization_schema_version": 1,
            "repo_id": repo_id,
            "selection": {
                "method": "farthest_point_phase_aligned"
                if diverse_per_task
                else "outcome_filter",
                "diverse_per_task": diverse_per_task,
                "requires_success_and_contact_valid": diverse_per_task is not None,
            },
            "episodes": manifest,
        },
    )
    write_json(
        output / "generation.json",
        {
            "runs": [
                json.loads((root / "run.json").read_text())
                for root in sources
                if (root / "run.json").exists()
            ]
        },
    )
    performance = {
        "streaming_encoding": streaming_encoding,
        "export_seconds": time.perf_counter() - started,
        "source_wait_seconds": source_wait_seconds,
        "export_active_seconds": time.perf_counter() - started - source_wait_seconds,
        "episodes": len(manifest),
        "frames": sum(r["length"] for r in manifest),
    }
    write_json(output / "export_metrics.json", performance)
    (output / "INCOMPLETE.json").unlink()
    return {
        "root": str(output),
        **performance,
    }


def load_dataset(
    root, chunk_length=16, outcome="all", require_contact_valid=False, quality="all"
):
    """Load policy inputs and padded action chunks without crossing episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    from .rendering import dataset_profile

    dataset_profile(root)
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Dataset export did not complete")
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = [
        row["episode_index"]
        for row in episode_metadata(root, outcome, require_contact_valid, quality)
    ]
    if not episodes:
        raise ValueError("No episodes match the requested outcome")
    info = json.loads((root / "meta" / "info.json").read_text())
    return LeRobotDataset(
        repo_id=manifest.get("repo_id", "local/ogbench-mjwarp"),
        root=root,
        episodes=episodes,
        video_backend="pyav",
        delta_timestamps={"action": [i / info["fps"] for i in range(chunk_length)]},
    )


def replay(root, episode=0, restore_frames=False, video=None):
    import torch

    from .config import PlannerConfig
    from .environment import BatchEnvironment
    from .tasks import make_env

    row, archive_path = rollout_path(root, episode)
    states = load_sim_states(archive_path)
    with np.load(archive_path, allow_pickle=False) as data:
        actions = data["action"]
    if len(states["qpos"]) != len(actions) + 1:
        raise ValueError(
            "Rollout needs one simulator snapshot per action plus terminal state"
        )
    env = make_env(row["env_id"], row["seed"], row["task_id"], row["image_size"])
    errors, frames = [], []
    try:
        sim = BatchEnvironment(env, 1, PlannerConfig(**row["planner"]))

        def state_at(tick):
            return {key: value[tick : tick + 1] for key, value in states.items()}

        sim.restore(state_at(0))
        for tick, action in enumerate(actions):
            if restore_frames:
                sim.restore(state_at(tick))
            if video:
                frames.append(sim.render(0)["front"])
            sim.step(torch.as_tensor(action[None], device=sim.device))
            recorded = states["qpos"][tick + 1]
            errors.append(float(np.max(np.abs(sim.qpos[0].cpu().numpy() - recorded))))
        success = bool(sim.success()[0])
        if video and frames:
            import av

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), "w") as container:
                stream = container.add_stream("libx264", rate=row["fps"])
                stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
                stream.pix_fmt = "yuv420p"
                for image in frames:
                    for packet in stream.encode(
                        av.VideoFrame.from_ndarray(image, format="rgb24")
                    ):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        nonfinite = sum(not np.isfinite(error) for error in errors)
        return {
            "episode": episode,
            "steps": len(errors),
            "max_qpos_error": None if nonfinite else max(errors, default=0),
            "nonfinite_steps": nonfinite,
            "success": success,
            "recorded_outcome": row["outcome"],
            "outcome_matches": success == (row["outcome"] == "success"),
            "restore_frames": restore_frames,
        }
    finally:
        env.close()
