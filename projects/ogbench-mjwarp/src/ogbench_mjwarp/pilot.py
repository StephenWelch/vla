"""Matched stacking workflow pilot; CEM stays the default regardless of results."""

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig

from .config import PlannerConfig
from .io import episode_metadata, jsonable, load_sim_states, write_json
from .recording import generate
from .stack_audit import audit_stack
from .tasks import make_env


@dataclass
class PilotConfig:
    output: Path
    episodes: int = 50
    batch_sizes: tuple[int, ...] = (1, 32)
    repeats: int = 3
    seed: int = 42000
    max_steps: int = 1000
    render: bool = True
    export: bool = True
    integration_train: bool = True
    resume: bool = False
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(enable=True))


def success_interval(successes, attempts):
    """Wilson 95% interval per independent reset bank, not pooled repetitions."""
    p, z = successes / attempts, 1.959963984540054
    denominator = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    half = (
        z
        * np.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts * attempts))
        / denominator
    )
    return [max(0, center - half), min(1, center + half)]


def paired_outcomes(trials):
    pairs = []
    for cem in (trial for trial in trials if trial["backend"] == "cem"):
        cu = next(
            (
                trial
                for trial in trials
                if trial["backend"] == "curobo"
                and (trial["batch_size"], trial["repeat"])
                == (cem["batch_size"], cem["repeat"])
            ),
            None,
        )
        if cu:
            counts = Counter(
                (a["stable_success"], b["stable_success"])
                for a, b in zip(cem["outcomes"], cu["outcomes"], strict=True)
            )
            pairs.append(
                {
                    "batch_size": cem["batch_size"],
                    "repeat": cem["repeat"],
                    "both": counts[True, True],
                    "neither": counts[False, False],
                    "cem_only": counts[True, False],
                    "curobo_only": counts[False, True],
                }
            )
    return pairs


def video_from_archive(archive, output, fps=20):
    """Encode both saved camera streams with bounded memory; no extra rendering."""
    import av

    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(archive) as saved,
        saved.open("front.npy") as front,
        saved.open("wrist.npy") as wrist,
    ):
        headers = [
            np.lib.format.read_array_header_1_0(stream)
            for stream in (front, wrist)
            if np.lib.format.read_magic(stream) == (1, 0)
        ]
        if len(headers) != 2 or headers[0] != headers[1]:
            raise ValueError("Expected matching RGB camera arrays")
        shape, fortran, dtype = headers[0]
        if fortran or dtype != np.uint8:
            raise ValueError("Expected contiguous RGB bytes")
        length, height, width, _ = shape
        with av.open(str(output), "w") as container:
            encoder = container.add_stream("libx264", rate=fps)
            encoder.width, encoder.height, encoder.pix_fmt = (
                width * 2,
                height,
                "yuv420p",
            )
            for _ in range(length):
                frame = np.concatenate(
                    [
                        np.frombuffer(
                            stream.read(height * width * 3), dtype=np.uint8
                        ).reshape(height, width, 3)
                        for stream in (front, wrist)
                    ],
                    axis=1,
                )
                for packet in encoder.encode(
                    av.VideoFrame.from_ndarray(frame, format="rgb24")
                ):
                    container.mux(packet)
            for packet in encoder.encode():
                container.mux(packet)


def run(config):
    if min(config.episodes, config.repeats, config.max_steps, *config.batch_sizes) < 1:
        raise ValueError("Pilot budgets must be positive")
    if (
        not config.batch_sizes
        or len(set(config.batch_sizes)) != len(config.batch_sizes)
        or config.seed < 0
    ):
        raise ValueError("Provide distinct batch sizes and a nonnegative seed")
    if config.output.exists() and not config.resume:
        raise FileExistsError("Use a fresh pilot output directory")
    config.output.mkdir(parents=True, exist_ok=True)
    settings_path = config.output / "config.json"
    if config.resume and settings_path.exists():
        previous = json.loads(settings_path.read_text())
        current = jsonable(asdict(config))
        if {k: v for k, v in previous.items() if k != "resume"} != {
            k: v for k, v in current.items() if k != "resume"
        }:
            raise ValueError("Resume cannot change the pilot configuration")
    tracker = Tracker(
        config.output,
        config.wandb,
        "planner-pilot",
        asdict(config),
        resume=config.resume,
    )
    write_json(config.output / "config.json", asdict(config))
    source = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("*.py")
    }
    write_json(config.output / "source.json", source)
    write_json(
        config.output / "status.json",
        {
            "status": "running",
            "expected_trials": 2 * config.repeats * len(config.batch_sizes),
        },
    )
    results_path = config.output / "results.json"
    trials = (
        json.loads(results_path.read_text())["trials"]
        if config.resume and results_path.exists()
        else []
    )
    rows_by_trial = {}
    fingerprints = {}
    failed = True
    try:
        for batch_size in config.batch_sizes:
            for repeat in range(config.repeats):
                order = ("cem", "curobo") if repeat % 2 == 0 else ("curobo", "cem")
                for backend in order:
                    planner = PlannerConfig(backend=backend, episodes=batch_size)
                    # Separate disjoint warmups from timed reset bank. Their archives
                    # remain available; no success-quota refill or selection is used.
                    name = f"{backend}-batch{batch_size}-repeat{repeat}"
                    if any(trial["name"] == name for trial in trials):
                        rows_by_trial[name] = episode_metadata(config.output / name)
                        for row in rows_by_trial[name]:
                            fingerprints[row["episode_id"]] = row["randomization"][
                                "initial_state_fingerprint"
                            ]
                        continue
                    generate(
                        config.output / f"warmup-{name}",
                        "cube-double-v0",
                        3,
                        [5],
                        config.seed + 10000,
                        planner,
                        max_steps=20,
                        record_images=False,
                    )
                    root = config.output / name
                    if (
                        config.resume
                        and root.exists()
                        and not (root / "summary.json").exists()
                    ):
                        raise ValueError(
                            "A timed trial was interrupted; use a fresh pilot output to avoid biased timing"
                        )
                    summary = generate(
                        root,
                        "cube-double-v0",
                        config.episodes,
                        [5],
                        config.seed,
                        planner,
                        max_steps=config.max_steps,
                        record_images=False,
                        settle_steps=20,
                    )
                    rows = episode_metadata(root)
                    env = make_env("cube-double-v0", task_id=5, size=32)
                    try:
                        for row in rows:
                            state = load_sim_states(root / row["archive"])
                            row["stack_quality"] = audit_stack(env, state)
                            row["stable_success"] = bool(
                                row["stack_quality"]["valid"]
                                and row["contact_quality"]["valid"]
                                and row["outcome"] == "success"
                            )
                            controls = state["ctrl"][:, env.unwrapped._arm_actuator_ids]
                            row["target_smoothness"] = {
                                "velocity_rms_rad_s": float(
                                    np.sqrt(np.mean(np.diff(controls, axis=0) ** 2))
                                    * 20
                                ),
                                "acceleration_rms_rad_s2": float(
                                    np.sqrt(
                                        np.mean(np.diff(controls, n=2, axis=0) ** 2)
                                    )
                                    * 400
                                )
                                if len(controls) > 2
                                else 0,
                            }
                            fingerprint = row["randomization"][
                                "initial_state_fingerprint"
                            ]
                            previous = fingerprints.setdefault(
                                row["episode_id"], fingerprint
                            )
                            if previous != fingerprint:
                                raise RuntimeError(
                                    "Matched initial reset fingerprints differ"
                                )
                            write_json(
                                root / f"episode-{row['episode_id']:06d}.json", row
                            )
                    finally:
                        env.close()
                    stable = sum(row["stable_success"] for row in rows)
                    seconds = summary["performance"]["steady_generation_seconds"]
                    trial = {
                        "name": name,
                        "backend": backend,
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "attempts": len(rows),
                        "native_successes": sum(
                            row.get("native_success", row["outcome"] == "success")
                            for row in rows
                        ),
                        "stable_successes": stable,
                        "stable_success_rate": stable / len(rows),
                        "stable_success_interval_95": success_interval(
                            stable, len(rows)
                        ),
                        "failure_counts": dict(
                            Counter(
                                row["reason"]
                                for row in rows
                                if row["outcome"] != "success"
                            )
                        ),
                        "smoothness": {
                            key: float(
                                np.mean([row["target_smoothness"][key] for row in rows])
                            )
                            for key in rows[0]["target_smoothness"]
                        },
                        "valid_demos_per_minute": stable * 60 / seconds,
                        "performance": summary["performance"],
                        "outcomes": [
                            {
                                "episode_id": row["episode_id"],
                                "stable_success": row["stable_success"],
                                "reason": row["reason"],
                                "stack_quality": row["stack_quality"],
                            }
                            for row in rows
                        ],
                    }
                    trials.append(trial)
                    rows_by_trial[name] = rows
                    write_json(
                        config.output / "results.json",
                        {"trials": trials, "paired": paired_outcomes(trials)},
                    )
                    tracker.log(
                        {
                            "pilot/trial": len(trials),
                            **{
                                f"pilot/{backend}/batch{batch_size}/{k}": v
                                for k, v in trial.items()
                                if isinstance(v, (float, int))
                            },
                        },
                        reports=[config.output / "results.json"],
                    )
        if config.render:
            from .dataset import export_dataset
            from .rerender import rerender

            for backend in ("cem", "curobo"):
                name = f"{backend}-batch{max(config.batch_sizes)}-repeat0"
                root = config.output / name
                selected = set(range(min(10, config.episodes)))
                if backend == "curobo" and config.export:
                    selected.update(
                        row["episode_id"]
                        for row in rows_by_trial[name]
                        if row["stable_success"]
                    )
                rendered = config.output / f"rendered-{backend}"
                render_metrics = rerender(root, rendered, 32, selected)
                tracker.log({f"render/{backend}": render_metrics})
                for episode in range(min(10, config.episodes)):
                    video = config.output / "videos" / f"{backend}-{episode:03d}.mp4"
                    video_from_archive(rendered / f"episode-{episode:06d}.npz", video)
                    tracker.log({}, videos={f"pilot/{backend}/episode{episode}": video})
                if backend == "curobo" and config.export:
                    accepted = config.output / "accepted-curobo"
                    accepted.mkdir()
                    for row in rows_by_trial[name]:
                        if row["stable_success"]:
                            for suffix in ("npz", "json"):
                                path = (
                                    rendered
                                    / f"episode-{row['episode_id']:06d}.{suffix}"
                                )
                                (accepted / path.name).symlink_to(path.resolve())
                    if any(accepted.glob("*.json")):
                        export_metrics = export_dataset(
                            accepted,
                            config.output / "dataset",
                            outcome="success",
                            require_contact_valid=True,
                        )
                        tracker.log({"export": export_metrics})
        dataset = config.output / "dataset"
        if config.integration_train and dataset.exists():
            from .evaluate import EvalConfig, evaluate
            from .train import TrainConfig, train

            training = config.output / "act-integration"
            train(
                TrainConfig(
                    dataset=dataset,
                    policy=None,
                    output=training,
                    policy_type="act",
                    steps=2,
                    batch_size=2,
                    workers=0,
                    save_freq=2,
                    overrides=["--log_freq=1"],
                    wandb=replace(
                        config.wandb, name=f"{config.output.name}-act-integration"
                    ),
                )
            )
            checkpoint = training / "checkpoints/000002/pretrained_model"
            evaluate(
                EvalConfig(
                    checkpoint=checkpoint,
                    dataset=dataset,
                    output=config.output / "act-integration-eval",
                    env="cube-double-v0",
                    task_ids=(5,),
                    episodes=2,
                    batch_size=2,
                    max_steps=20,
                    videos=1,
                )
            )
            tracker.log(
                {"integration/act_updates": 2},
                reports=[config.output / "act-integration-eval/eval_info.json"],
            )
        report = [
            "# Stacking workflow pilot",
            "",
            "CEM remains the default. This compares planners and controllers together.",
            "",
            "| Backend | Batch | Repeat | Stable successes | Valid demos/min | Steady seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for trial in trials:
            report.append(
                f"| {trial['backend']} | {trial['batch_size']} | {trial['repeat']} | {trial['stable_successes']}/{trial['attempts']} | {trial['valid_demos_per_minute']:.3f} | {trial['performance']['steady_generation_seconds']:.2f} |"
            )
        report.extend(
            [
                "",
                "Confidence intervals and paired outcomes are in results.json. Repeats reuse the reset bank; do not pool them as independent scenarios.",
                "",
                "ACT received two integration updates and bounded evaluation."
                if config.integration_train and dataset.exists()
                else "ACT integration was skipped (disabled or no accepted dataset).",
            ]
        )
        (config.output / "report.md").write_text("\n".join(report) + "\n")
        tracker.log({}, reports=[config.output / "report.md"])
        failed = False
    finally:
        write_json(
            config.output / "status.json",
            {
                "status": "failed" if failed else "complete",
                "completed_trials": len(trials),
            },
        )
        tracker.finish(failed=failed)
    return trials


def main():
    run(parse_args(PilotConfig))


if __name__ == "__main__":
    main()
