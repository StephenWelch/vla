"""Isolated matched trials, quality-filtered datasets, and W&B reporting."""

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig

from .io import episode_metadata, write_json


@dataclass
class SplinePilotConfig:
    output: Path
    episodes: int = 50
    tuning_episodes: int = 10
    repeats: int = 2
    batch_size: int = 32
    seed: int = 62000
    tuning_seed: int = 52000
    max_steps: int = 1600
    memory_limit_gib: float = 28.0
    # Wait for a known existing job before starting measured GPU work.
    wait_pid: int | None = None
    render_export: bool = True
    trial: Literal["baseline", "nominal", "moderate", "mixed"] | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


def trial(config):
    import numpy as np

    from .config import CuroboConfig, PlannerConfig, SplineConfig
    from .execution_ablation import inspect
    from .recording import generate

    previous = episode_metadata(config.output) if config.output.exists() else []
    if previous and (
        len(previous) != config.episodes
        or not (config.output / "summary.json").exists()
    ):
        raise ValueError(
            "An interrupted measured trial cannot reuse partial timing; choose a fresh trial output"
        )

    backend = "curobo" if config.trial == "baseline" else "spline"
    planner = PlannerConfig(
        backend=backend,
        episodes=config.batch_size,
        curobo=CuroboConfig(execution="timed"),
        spline=SplineConfig(
            variation=config.trial if backend == "spline" else "nominal"
        ),
    )
    result = generate(
        config.output,
        "cube-double-v0",
        config.episodes,
        [5],
        config.seed,
        planner,
        max_steps=config.max_steps,
        record_images=False,
        settle_steps=20,
    )
    # Baseline recorder predates whole-attempt quality labels. Audit identically,
    # including the cost of that audit in measured generation time.
    started = time.perf_counter()
    if backend == "curobo":
        rows = inspect(config.output, config.episodes, config.output)
        for row in rows:
            row["quality"] = {
                "physical_valid": bool(
                    row["contact_quality"]["valid"]
                    and row["reason"] not in ("capacity_overflow", "numerical_failure")
                ),
                "completed": row["outcome"] == "success",
                "stable_success": row["stable_success"],
                "planning_rejected": row["reason"] == "planning_failure",
            }
            write_json(config.output / f"episode-{row['episode_id']:06d}.json", row)
        result["performance"]["steady_generation_seconds"] += (
            time.perf_counter() - started
        )
    rows = episode_metadata(config.output)
    seconds = result["performance"]["steady_generation_seconds"]
    successes = len(episode_metadata(config.output, quality="validated-success"))
    failures = len(episode_metadata(config.output, quality="valid-failure"))
    result["performance"].update(
        validated_successes_per_minute=60 * successes / seconds,
        valid_failures_per_minute=60 * failures / seconds,
    )
    result.update(
        mode=config.trial,
        attempts=len(rows),
        validated_successes=successes,
        valid_failures=failures,
        validated_successes_per_minute=60 * successes / seconds,
        valid_failures_per_minute=60 * failures / seconds,
        invalid_attempts=sum(r["quality"]["physical_valid"] is False for r in rows),
        planning_rejections=sum(r["quality"]["planning_rejected"] for r in rows),
        mean_duration_seconds=float(np.mean([r["length"] / r["fps"] for r in rows])),
        motion={
            name: float(
                np.mean(
                    [
                        r["motion"]["actual"][name]
                        for r in rows
                        if "actual" in r.get("motion", {})
                    ]
                )
            )
            for name in ("acceleration_rms_rad_s2", "jerk_rms_rad_s3", "pause_fraction")
            if any("actual" in r.get("motion", {}) for r in rows)
        },
        memory_within_budget=max(
            result["performance"]["torch_peak_reserved_bytes"],
            result["performance"].get("device_peak_used_bytes", 0),
        )
        <= config.memory_limit_gib * 2**30,
    )
    write_json(config.output / "trial.json", result)
    if not result["memory_within_budget"]:
        raise RuntimeError("Trial exceeded the configured GPU memory budget")
    return result


def render_datasets(root, trials, tracker):
    from .dataset import export_dataset
    from .pilot import video_from_archive
    from .rerender import rerender

    rendered = []
    for path in trials:
        if not path.name.startswith("pilot-mixed"):
            continue
        selected = episode_metadata(
            path, quality="validated-success"
        ) + episode_metadata(path, quality="valid-failure")
        if selected:
            output = root / "rendered" / path.name
            if not output.exists():
                output.parent.mkdir(exist_ok=True)
                rerender(
                    path, output, 32, episode_ids=[r["episode_id"] for r in selected]
                )
            rendered.append(output)
    seeds = sorted({r["seed"] for p in rendered for r in episode_metadata(p)})
    validation = set(seeds[::5])
    for path in rendered:
        for row in episode_metadata(path):
            row["dataset_split"] = "val" if row["seed"] in validation else "train"
            write_json(path / f"episode-{row['episode_id']:06d}.json", row)
        for quality in ("validated-success", "valid-failure"):
            for row in episode_metadata(path, quality=quality)[:10]:
                video = root / "videos" / f"{path.name}-{row['episode_id']:06d}.mp4"
                video.parent.mkdir(exist_ok=True)
                video_from_archive(path / row["archive"], video)
                tracker.log(
                    {},
                    videos={f"review/{quality}/{path.name}/{row['episode_id']}": video},
                )
    exports = {}
    for name, quality in (
        ("successes", "validated-success"),
        ("failures", "valid-failure"),
    ):
        if not any(episode_metadata(p, quality=quality) for p in rendered):
            exports[name] = {"episodes": 0, "reason": "No matching completed attempts"}
            continue
        destination = root / "datasets" / name
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            exports[name] = export_dataset(
                rendered, destination, f"local/ogbench-spline-{name}", quality=quality
            )
        else:
            if (destination / "INCOMPLETE.json").exists():
                raise RuntimeError(
                    f"Incomplete export needs a fresh destination: {destination}"
                )
            exports[name] = {"root": str(destination)}
    write_json(root / "exports.json", exports)
    # Explicit cross-export split, grouped by reset seed. Repeats never leak.
    write_json(
        root / "dataset-split.json",
        {
            "method": "reset seed grouped across outcomes, variants, repeats; every fifth sorted seed held out",
            "validation_seeds": sorted(validation),
            "train_seeds": sorted(set(seeds) - validation),
        },
    )
    tracker.log({}, reports=[root / "exports.json", root / "dataset-split.json"])


def run(config):
    if (
        min(config.episodes, config.tuning_episodes, config.repeats, config.batch_size)
        < 1
    ):
        raise ValueError("Episode counts, repeats and batch size must be positive")
    if set(range(config.seed, config.seed + config.episodes)) & set(
        range(config.tuning_seed, config.tuning_seed + config.tuning_episodes)
    ):
        raise ValueError("Tuning and held-out reset banks must be disjoint")
    config.output.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(
        config.output,
        config.wandb,
        "spline-pilot",
        asdict(config),
        resume=(config.output / "tracking.json").exists(),
    )
    failed, trials = True, []
    try:
        if config.wait_pid:
            process = Path(f"/proc/{config.wait_pid}/stat")
            identity = process.read_text().split()[21] if process.exists() else None
            write_json(
                config.output / "status.json",
                {"status": "waiting", "pid": config.wait_pid},
            )
            while process.exists() and process.read_text().split()[21] == identity:
                time.sleep(30)
        # Fresh processes release all Torch/Warp/cuRobo allocations between trials.
        schedule = [
            ("smoke", "nominal", 1, 1, config.tuning_seed, 0),
            (
                "memory",
                "mixed",
                config.batch_size,
                config.batch_size,
                config.tuning_seed,
                0,
            ),
        ]
        schedule += [
            (stage, mode, episodes, config.batch_size, seed, repeat)
            for stage, episodes, seed, repeats in (
                ("tuning", config.tuning_episodes, config.tuning_seed, 1),
                ("pilot", config.episodes, config.seed, config.repeats),
            )
            for repeat in range(repeats)
            for mode in ("baseline", "nominal", "moderate", "mixed")
        ]
        for stage, mode, episodes, batch, seed, repeat in schedule:
            path = config.output / f"{stage}-{mode}-{repeat}"
            write_json(
                config.output / "status.json",
                {
                    "status": "running",
                    "trial": path.name,
                    "completed_trials": len(trials),
                },
            )
            if not (path / "trial.json").exists():
                child = asdict(config) | {
                    "output": str(path),
                    "episodes": episodes,
                    "batch_size": batch,
                    "seed": seed,
                    "trial": mode,
                    "wait_pid": None,
                    "wandb": {"enable": False},
                }
                child_file = config.output / "child.json"
                write_json(child_file, child)
                with (config.output / f"{path.name}.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "ogbench_mjwarp.spline_pilot",
                            "--config",
                            str(child_file),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            result = json.loads((path / "trial.json").read_text())
            if not result["memory_within_budget"]:
                raise RuntimeError(f"Memory budget exceeded in {path}")
            trials.append(path)
            tracker.log({f"{stage}/{mode}": result}, reports=[path / "trial.json"])
            if stage == "tuning" and mode == "mixed":
                candidates = {
                    name: json.loads(
                        (config.output / f"tuning-{name}-0/trial.json").read_text()
                    )["validated_successes_per_minute"]
                    for name in ("nominal", "moderate", "mixed")
                }
                write_json(
                    config.output / "tuning-choice.json",
                    {
                        "criterion": "maximum validated successes per generation minute on tuning seeds; no acceptance-quota refill",
                        "selected": max(candidates, key=candidates.get),
                        "rates": candidates,
                        "note": "All four fixed configurations are still compared on held-out seeds; production defaults remain opt-in.",
                    },
                )
            if stage == "smoke" and not any(
                r["quality"]["completed"] for r in episode_metadata(path)
            ):
                raise RuntimeError(
                    "Nominal smoke did not complete an attempt; inspect before spending the pilot budget"
                )
        results = {p.name: json.loads((p / "trial.json").read_text()) for p in trials}
        write_json(config.output / "results.json", results)
        lines = [
            "# Spline stacking pilot",
            "",
            "Matched resets; fixed attempt budgets. Repeats share scenarios and are not independent trials.",
            "",
            "| Trial | Stable successes | Valid failures | Successes/min |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {name} | {r['validated_successes']}/{r['attempts']} | {r['valid_failures']} | {r['validated_successes_per_minute']:.3f} |"
            for name, r in results.items()
        ]
        lines += [
            "",
            "Generation timing includes audits and archive writing. Warmup and rendering are separate. Review rendered failures before describing them as humanlike.",
        ]
        (config.output / "report.md").write_text("\n".join(lines) + "\n")
        tracker.log(
            {}, reports=[config.output / "results.json", config.output / "report.md"]
        )
        if config.render_export:
            render_datasets(config.output, trials, tracker)
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


if __name__ == "__main__":
    config = parse_args(SplinePilotConfig)
    if config.trial:
        trial(config)
    else:
        run(config)
