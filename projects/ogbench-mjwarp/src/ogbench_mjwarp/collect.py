"""Generate attempts, render audited episodes, and export both outcome classes."""

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig

from .config import PlannerConfig
from .io import episode_metadata, write_json
from .spline import quality_selection


@dataclass
class CollectionConfig:
    output: Path
    episodes: int = 500
    seed: int = 72000
    env: str = "cube-double-v0"
    task_ids: list[int] = field(default_factory=lambda: [5])
    max_steps: int = 2000
    render_batch_size: int = 32
    preview_per_quality: int = 3
    render_export: bool = True
    # Whole-attempt planners should refill complete batches, not replan a full
    # GPU batch every time one shorter episode finishes.
    refill_slots: bool = False
    planner: PlannerConfig = field(
        default_factory=lambda: PlannerConfig(backend="spline", episodes=32)
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)


def factor_summary(rows):
    """Report realized proposal/selection coverage separately for each outcome."""
    result = {}
    for name, quality in (
        ("all", "all"),
        ("successes", "validated-success"),
        ("failures", "valid-failure"),
    ):
        selected = [r for r in rows if quality_selection(r, quality)]
        speeds, radii, tilts, offsets = [], [], [], []
        for row in selected:
            sample = row["randomization"]["factors"]["spline"]
            speeds.append(sample["execution_speed"])
            radii.extend(
                np.linalg.norm(
                    np.asarray(sample["approach_vectors"])[..., :2], axis=-1
                ).ravel()
            )
            for i in sample.get("selected_candidates", []):
                grasp = sample["candidates"][i]
                tilts.append(np.rad2deg(grasp["tilt_radians"]))
                offsets.append(np.linalg.norm(grasp["offset_xy"]))
        result[name] = {"episodes": len(selected)}
        for key, values in (
            ("execution_speed", speeds),
            ("approach_radius_m", radii),
            ("selected_tilt_degrees", tilts),
            ("selected_grasp_radius_m", offsets),
        ):
            if values:
                result[name][key] = dict(
                    zip(
                        ("min", "median", "max"),
                        map(float, np.quantile(values, [0, 0.5, 1])),
                        strict=True,
                    )
                )
    return result


def render_datasets(config, rows, tracker):
    root = config.output
    raw = root / "raw"
    from .dataset import export_dataset, load_dataset
    from .pilot import video_from_archive
    from .rerender import rerender

    accepted = [
        r
        for r in rows
        if any(quality_selection(r, q) for q in ("validated-success", "valid-failure"))
    ]
    rendered = []
    for offset in range(0, len(accepted), config.render_batch_size):
        batch = accepted[offset : offset + config.render_batch_size]
        destination = root / "rendered" / f"batch-{offset:06d}"
        write_json(
            root / "status.json",
            {
                "status": "rendering",
                "completed": offset,
                "total": len(accepted),
            },
        )
        if not (destination / "run.json").exists():
            if shutil.disk_usage(root).free < 4 * 2**30:
                raise RuntimeError(
                    "Rendering needs at least 4 GiB free; raw attempts and completed batches are retained"
                )
            rerender(
                raw,
                destination,
                config.render_batch_size,
                episode_ids=[r["episode_id"] for r in batch],
            )
        if {r["episode_id"] for r in episode_metadata(destination)} != {
            r["episode_id"] for r in batch
        }:
            raise RuntimeError("Rendered batch does not match selected episodes")
        rendered.append(destination)
        tracker.log(
            {
                "rendering": {
                    "completed_episodes": offset + len(batch),
                    "total_episodes": len(accepted),
                }
            }
        )
    exports = {}
    for name, quality in (
        ("successes", "validated-success"),
        ("failures", "valid-failure"),
    ):
        write_json(root / "status.json", {"status": "exporting", "quality": quality})
        count = sum(quality_selection(r, quality) for r in rows)
        if not count:
            exports[name] = {"episodes": 0}
            continue
        destination = root / "datasets" / name
        if (destination / "INCOMPLETE.json").exists():
            raise RuntimeError(
                f"Incomplete export requires a fresh destination: {destination}"
            )
        if not destination.exists():
            exports[name] = export_dataset(
                rendered,
                destination,
                f"local/ogbench-stack-diverse-{name}",
                quality=quality,
                progress=lambda report, name=name: tracker.log(
                    {"export_progress": {name: report}}
                ),
            )
        dataset = load_dataset(destination, quality=quality)
        assert dataset.num_episodes == count
        exports.setdefault(name, {"episodes": count, "root": str(destination)})
        exports[name]["sample_action_shape"] = list(dataset[0]["action"].shape)
        previews = 0
        for batch in rendered:
            for row in episode_metadata(batch, quality=quality):
                if previews >= config.preview_per_quality:
                    break
                video = root / "videos" / f"{name}-{row['episode_id']:06d}.mp4"
                video_from_archive(batch / row["archive"], video)
                tracker.log({}, videos={f"review/{name}/{row['episode_id']}": video})
                previews += 1
    write_json(root / "exports.json", exports)
    tracker.log(
        {"exports": exports},
        reports=[root / "exports.json", root / "dataset-split.json"],
    )


def run(config):
    if (
        min(config.episodes, config.render_batch_size, config.max_steps) < 1
        or config.preview_per_quality < 0
    ):
        raise ValueError("Invalid collection counts")
    root = config.output
    root.mkdir(parents=True, exist_ok=True)
    resolved = json.loads(json.dumps(asdict(config), default=str))
    config_path = root / "collection.json"
    if config_path.exists() and json.loads(config_path.read_text()) != resolved:
        raise ValueError("Collection configuration changed; choose a fresh output")
    write_json(config_path, resolved)
    tracker = Tracker(
        root,
        config.wandb,
        "collection",
        resolved,
        resume=(root / "tracking.json").exists(),
    )
    failed = True
    try:
        raw = root / "raw"
        generate_config = {
            k: resolved[k]
            for k in (
                "episodes",
                "seed",
                "env",
                "task_ids",
                "max_steps",
                "planner",
                "refill_slots",
            )
        }
        generate_config.update(output=str(raw), record_images=False)
        write_json(root / "generate.json", generate_config)
        write_json(
            root / "status.json", {"status": "generating", "attempts": config.episodes}
        )
        # Release planner allocations before the batched renderer starts.
        with (root / "generate.log").open("a") as log:
            command = [
                sys.executable,
                "-u",
                "-m",
                "ogbench_mjwarp.cli",
                "generate",
                "--config",
                str(root / "generate.json"),
            ]
            with subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            ) as child:
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        progress = {
                            "completed_attempts": len(list(raw.glob("episode-*.json"))),
                            "total_attempts": config.episodes,
                        }
                        write_json(
                            root / "status.json", {"status": "generating", **progress}
                        )
                        tracker.log({"generation_progress": progress})
                if code:
                    raise subprocess.CalledProcessError(code, command)
        rows = episode_metadata(raw)
        if len(rows) != config.episodes:
            raise RuntimeError("Attempt count mismatch")
        validation = set(range(config.seed, config.seed + config.episodes, 5))
        for row in rows:
            row["dataset_split"] = "val" if row["seed"] in validation else "train"
            write_json(raw / f"episode-{row['episode_id']:06d}.json", row)
        write_json(
            root / "dataset-split.json",
            {
                "method": "every fifth reset seed, assigned before outcome filtering",
                "validation_seeds": sorted(validation),
            },
        )
        coverage = factor_summary(rows)
        write_json(root / "factor-summary.json", coverage)
        summary = json.loads((raw / "summary.json").read_text())
        tracker.log(
            {"generation": summary, "factors": coverage},
            reports=[raw / "summary.json", root / "factor-summary.json", config_path],
        )
        if config.render_export:
            render_datasets(config, rows, tracker)
        failed = False
    finally:
        write_json(root / "status.json", {"status": "failed" if failed else "complete"})
        tracker.finish(failed=failed)


if __name__ == "__main__":
    run(parse_args(CollectionConfig))
