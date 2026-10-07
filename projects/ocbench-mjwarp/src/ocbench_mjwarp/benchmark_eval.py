"""Serial parity/throughput sweep; refuses to compete with active training."""

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

from vla_tools.config import parse_args


@dataclass
class BenchmarkConfig:
    checkpoint: Path
    dataset: Path
    output: Path
    episodes: int = 4
    max_steps: int = 250
    wait_for_queue: Path | None = None


def wait_for_training(queue):
    import psutil

    if queue:
        while True:
            status = json.loads(queue.read_text())["status"]
            if status == "complete":
                break
            if status != "running":
                raise RuntimeError(f"Training queue is {status}; benchmark not started")
            time.sleep(30)
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if any(
            module in command
            for module in ("ocbench_mjwarp.train", "ocbench_mjwarp.training")
        ):
            raise RuntimeError(
                f"Training PID {process.pid} is active; benchmark not started"
            )


def benchmark(config):
    wait_for_training(config.wait_for_queue)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    from lerobot.utils.random_utils import set_seed
    from vla_tools.policy import load_policy
    from vla_tools.tracking import write_json

    from .evaluate import EvalConfig, evaluate_task, prepare_evaluation

    if config.output.exists():
        raise FileExistsError(config.output)
    config.output.mkdir(parents=True)
    # Resolve `last` once, so every case uses exactly the same weights.
    checkpoint = config.checkpoint.resolve()
    manifest = json.loads((config.dataset / "manifest.json").read_text())
    seeds = sorted({row["seed"] for row in manifest["episodes"]})[: config.episodes]
    if len(seeds) != config.episodes:
        raise ValueError("Not enough unique dataset reset seeds")
    base = EvalConfig(
        checkpoint=checkpoint,
        dataset=config.dataset,
        output=config.output / "unused",
        episodes=config.episodes,
        max_steps=config.max_steps,
        seeds=seeds,
        videos=0,
    )
    environment, _ = prepare_evaluation(base)
    policy, pre, post = load_policy(checkpoint, "cuda")
    report = {
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "max_steps": config.max_steps,
        "parity": [],
        "cases": [],
        "status": "running",
    }
    path = config.output / "benchmark.json"

    def run(cfg, trace=None):
        # Recheck before every case if the user starts another training job.
        wait_for_training(None)
        set_seed(1000)
        return evaluate_task(cfg, environment, policy, pre, post, 2, trace=trace)

    try:
        for worlds in (1, 2, 4):
            ref_cfg = replace(base, batch_size=worlds, rollout_backend="lerobot")
            reference, repeated = [], []
            first = run(
                replace(ref_cfg, output=config.output / f"parity-{worlds}-reference"),
                reference,
            )
            run(
                replace(ref_cfg, output=config.output / f"parity-{worlds}-repeat"),
                repeated,
            )
            for batch in (1, 2, 4):
                actual = []
                metrics = run(
                    replace(
                        ref_cfg,
                        output=config.output / f"parity-{worlds}-{batch}",
                        rollout_backend="chunked",
                        render_batch_frames=batch,
                    ),
                    actual,
                )
                checks = {
                    "trace_length": len(actual) == len(reference) == len(repeated)
                }
                errors = {}
                if checks["trace_length"]:
                    for key in ("action", "qpos", "done"):
                        expected = np.stack([r[key] for r in reference])
                        baseline = np.stack([r[key] for r in repeated])
                        candidate = np.stack([r[key] for r in actual])
                        repeat_error = float(
                            np.max(np.abs(expected.astype(float) - baseline))
                        )
                        error = float(
                            np.max(np.abs(expected.astype(float) - candidate))
                        )
                        tolerance = 0 if key == "done" else max(1e-6, repeat_error * 8)
                        checks[key] = error <= tolerance
                        errors[key] = {
                            "max_error": error,
                            "reference_repeat_error": repeat_error,
                            "tolerance": tolerance,
                        }
                keys = (
                    "seed",
                    "steps",
                    "success",
                    "task_success",
                    "contact_valid",
                    "physics_valid",
                    "truncated",
                    "stable_stack",
                    "stable_stack_failures",
                )
                checks["diagnostics"] = all(
                    all(a[k] == b[k] for k in keys)
                    for a, b in zip(
                        first["per_episode"], metrics["per_episode"], strict=True
                    )
                )
                report["parity"].append(
                    {
                        "worlds": worlds,
                        "render_batch_frames": batch,
                        "checks": checks,
                        "errors": errors,
                    }
                )
                write_json(path, report)
                if not all(checks.values()):
                    raise RuntimeError("Rollout parity failed; see benchmark.json")
            for videos in (0, 1):
                for backend, temporal in (
                    ("lerobot", 1),
                    ("chunked", 1),
                    ("chunked", 2),
                    ("chunked", 4),
                ):
                    cfg = replace(
                        ref_cfg,
                        videos=videos,
                        rollout_backend=backend,
                        render_batch_frames=temporal,
                        output=config.output
                        / f"timing-{worlds}-{videos}-{backend}-{temporal}",
                    )
                    result = run(cfg)
                    report["cases"].append(
                        {
                            "worlds": worlds,
                            "videos": videos,
                            "backend": backend,
                            "render_batch_frames": temporal,
                            "performance": result["performance"],
                        }
                    )
                    write_json(path, report)
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed_or_interrupted"
        raise
    finally:
        write_json(path, report)


if __name__ == "__main__":
    benchmark(parse_args(BenchmarkConfig))
