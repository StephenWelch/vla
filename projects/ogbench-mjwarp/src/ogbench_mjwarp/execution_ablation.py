"""Matched cuRobo executor comparison with physical checks and joint traces."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from vla_tools.config import parse_args
from vla_tools.tracking import Tracker, WandbConfig

from .config import CuroboConfig, PlannerConfig
from .io import episode_metadata, load_sim_states, write_json
from .recording import generate
from .stack_audit import audit_stack
from .tasks import make_env


@dataclass
class ExecutionAblationConfig:
    output: Path
    episodes: int = 4
    seed: int = 42000
    max_steps: int = 1000
    # Reuse an archived waypoint trial, avoiding another baseline GPU run.
    baseline: Path | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


def motion_metrics(command, position, velocity, timestep=0.05):
    """Report local peaks and substantial reversals alongside aggregate smoothness."""
    command_velocity = np.diff(command, axis=0) / timestep
    result = {}
    for name, values in (("command", command_velocity), ("actual", velocity)):
        acceleration = np.diff(values, axis=0) / timestep
        jerk = np.diff(acceleration, axis=0) / timestep
        # Count crossings after suppressing tiny velocities, including crossings
        # through a zero-velocity interval. These also include intended reversals.
        reversals = 0
        for joint in values.T:
            signs = np.sign(joint[np.abs(joint) > 0.05])
            reversals += int(np.count_nonzero(np.diff(signs)))
        result[name] = {
            "velocity_rms_rad_s": float(np.sqrt(np.mean(values**2))),
            "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2)))
            if acceleration.size
            else 0.0,
            "acceleration_peak_rad_s2": float(np.max(np.abs(acceleration)))
            if acceleration.size
            else 0.0,
            "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
            "jerk_peak_rad_s3": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
            "pause_fraction": float(np.mean(np.max(np.abs(values), axis=1) < 0.02))
            if values.size
            else 0.0,
            "reversals_above_0_05_rad_s": reversals,
            "reversals_per_second": reversals / ((len(command) - 1) * timestep),
        }
    result["tracking_error_rms_rad"] = float(
        np.sqrt(np.mean((command - position) ** 2))
    )
    return result


def inspect(root, episodes, output):
    rows = episode_metadata(root)[:episodes]
    if len(rows) != episodes:
        raise ValueError("Comparison requires all requested episodes")
    env = make_env("cube-double-v0", task_id=5, size=32)
    try:
        model = env.unwrapped._model
        q = model.jnt_qposadr[env.unwrapped._arm_joint_ids]
        v = model.jnt_dofadr[env.unwrapped._arm_joint_ids]
        for row in rows:
            states = load_sim_states(root / row["archive"])
            command = states["ctrl"][:, env.unwrapped._arm_actuator_ids]
            position, velocity = states["qpos"][:, q], states["qvel"][:, v]
            row["motion"] = motion_metrics(command, position, velocity)
            with np.load(root / row["archive"]) as archive:
                if "annotation/requested_action" in archive:
                    requested = archive["annotation/requested_action"][:, :6]
                    row["motion"]["limiter_error_rms_rad"] = float(
                        np.sqrt(np.mean((command[1:] - requested) ** 2))
                    )
            row["stack_quality"] = audit_stack(env, states)
            row["stable_success"] = bool(
                row["outcome"] == "success"
                and row["contact_quality"]["valid"]
                and row["stack_quality"]["valid"]
            )
            trace = output / f"episode-{row['episode_id']:06d}-joints.npz"
            np.savez_compressed(
                trace,
                time=np.arange(len(command)) * 0.05,
                command=command,
                position=position,
                velocity=velocity,
            )
    finally:
        env.close()
    return rows


def report(results, output):
    lines = [
        "# cuRobo executor comparison",
        "",
        "Matched reset seeds; both modes use the same contact and final stack audit.",
        "",
        "| Executor | Stable stacks | Mean simulated duration (s) | Actual acceleration RMS (rad/s²) | Actual reversals/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, trial in results.items():
        rows = trial["outcomes"]
        acceleration = np.mean(
            [r["motion"]["actual"]["acceleration_rms_rad_s2"] for r in rows]
        )
        reversals = np.mean(
            [r["motion"]["actual"]["reversals_per_second"] for r in rows]
        )
        lines.append(
            f"| {mode} | {trial['stable_successes']}/{len(rows)} | {trial['mean_duration_seconds']:.2f} | {acceleration:.3f} | {reversals:.3f} |"
        )
    lines += [
        "",
        "Duration and motion statistics include failed attempts. A shorter failed rollout is not a speed improvement. Reversals include intentional direction changes; inspect the joint traces. This small pilot does not establish a reliable success rate or isolated GPU throughput.",
        "",
        "Per-episode outcomes, contact checks, final stack audits and motion metrics are in `results.json`. Aligned command, measured position and measured velocity are in `*-traces/*.npz`. Phase metadata and requested actions are retained in the raw recordings.",
        "",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines))
    return path


def run(config):
    if config.episodes < 1 or config.max_steps < 1 or config.seed < 0:
        raise ValueError("Need positive budgets and a nonnegative seed")
    if config.output.exists():
        raise FileExistsError("Choose a fresh ablation output directory")
    config.output.mkdir(parents=True)
    tracker = Tracker(config.output, config.wandb, "executor-ablation", asdict(config))
    write_json(config.output / "config.json", asdict(config))
    write_json(config.output / "status.json", {"status": "running"})
    write_json(
        config.output / "source.json",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    )
    results = {}
    try:
        for execution in ("waypoint", "timed"):
            root = config.baseline if execution == "waypoint" else None
            if root is not None:
                saved = json.loads((root / "run.json").read_text())
                if (
                    saved["seed"] != config.seed
                    or saved["env_id"] != "cube-double-v0"
                    or saved["task_ids"] != [5]
                    or saved["planner"]["backend"] != "curobo"
                    or saved["planner"]["episodes"] != 1
                    or saved["planner"]["curobo"].get("execution", "waypoint")
                    != "waypoint"
                ):
                    raise ValueError(
                        "Baseline must be a matching batch-1 waypoint trial"
                    )
                expected = PlannerConfig(backend="curobo", episodes=1).to_dict()
                expected["curobo"].pop("execution")
                actual = saved["planner"]
                actual["curobo"].pop("execution", None)
                if actual != expected or saved["execution"].get("settle_steps") != 20:
                    raise ValueError(
                        "Baseline planner limits and final hold must match"
                    )
            else:
                root = config.output / execution
                generate(
                    root,
                    "cube-double-v0",
                    config.episodes,
                    [5],
                    config.seed,
                    PlannerConfig(
                        backend="curobo",
                        episodes=1,
                        curobo=CuroboConfig(execution=execution),
                    ),
                    max_steps=config.max_steps,
                    record_images=False,
                    settle_steps=20,
                )
            traces = config.output / f"{execution}-traces"
            traces.mkdir()
            rows = inspect(root, config.episodes, traces)
            results[execution] = {
                "root": str(root),
                "outcomes": rows,
                "stable_successes": sum(row["stable_success"] for row in rows),
                "mean_duration_seconds": float(
                    np.mean([row["length"] * 0.05 for row in rows])
                ),
            }
            write_json(config.output / "results.json", results)
            tracker.log(
                {
                    execution: {
                        k: v for k, v in results[execution].items() if k != "outcomes"
                    }
                },
                tables={f"{execution}/episodes": rows},
            )
        for old, new in zip(
            results["waypoint"]["outcomes"], results["timed"]["outcomes"], strict=True
        ):
            if (
                old["randomization"]["initial_state_fingerprint"]
                != new["randomization"]["initial_state_fingerprint"]
            ):
                raise RuntimeError("Matched reset fingerprints differ")
        write_json(
            config.output / "status.json",
            {"status": "complete", "matched_resets": True},
        )
        tracker.log(
            reports=[report(results, config.output), config.output / "results.json"]
        )
        tracker.finish()
    except BaseException:
        write_json(config.output / "status.json", {"status": "failed"})
        tracker.finish(failed=True)
        raise
    return results


def main():
    run(parse_args(ExecutionAblationConfig))


if __name__ == "__main__":
    main()
