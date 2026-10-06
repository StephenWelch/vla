"""Typed research commands; defaults < YAML < explicit CLI arguments."""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import tyro
from omegaconf import OmegaConf

from .config import PlannerConfig, RandomizationConfig


@dataclass
class ListTasks:
    command: ClassVar[str] = "list-tasks"


@dataclass
class Doctor:
    command: ClassVar[str] = "doctor"
    env: str = "cube-single-v0"


@dataclass
class Generate(Doctor):
    command: ClassVar[str] = "generate"
    output: Path = tyro.MISSING
    episodes: int = 32
    task_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    seed: int = 0
    size: int | tuple[int, int] = (480, 640)
    max_steps: int | None = None
    record_images: bool = True
    refill_slots: bool = True
    batched_cpu: bool = True
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)


@dataclass
class Benchmark(Doctor):
    command: ClassVar[str] = "benchmark"
    steps: int = 5
    size: int = 32
    planner: PlannerConfig = field(default_factory=PlannerConfig)


@dataclass
class Export:
    command: ClassVar[str] = "export"
    source: list[Path] = tyro.MISSING
    output: Path = tyro.MISSING
    repo_id: str = "local/ogbench-mjwarp"
    outcome: Literal["all", "success", "failure"] = "all"
    require_contact_valid: bool = False
    quality: Literal["all", "validated-success", "valid-failure"] = "all"
    diverse_per_task: int | None = None
    streaming_encoding: bool = True
    encoder_queue_size: int = 30
    encoder_threads: int = 2


@dataclass
class Rerender:
    command: ClassVar[str] = "rerender"
    source: Path = tyro.MISSING
    output: Path = tyro.MISSING
    batch_size: int = 32


@dataclass
class Inspect:
    command: ClassVar[str] = "inspect"
    root: Path = tyro.MISSING
    chunk_length: int = 16
    outcome: Literal["all", "success", "failure"] = "all"
    require_contact_valid: bool = False
    quality: Literal["all", "validated-success", "valid-failure"] = "all"


@dataclass
class Diversity:
    command: ClassVar[str] = "diversity"
    source: list[Path] = tyro.MISSING
    samples: int = 8
    output: Path | None = None


@dataclass
class Ablate:
    command: ClassVar[str] = "ablate"
    output: Path = tyro.MISSING
    env: str = "cube-double-v0"
    task_id: int = 2
    variants: int = 4
    seed: int = 2026
    size: int = 32
    record_images: bool = False
    planner: PlannerConfig = field(
        default_factory=lambda: PlannerConfig(
            episodes=4, candidates=8, horizon=8, iterations=1, joint_target_noise=0.01
        )
    )
    randomization: RandomizationConfig = field(
        default_factory=lambda: RandomizationConfig(
            order=True,
            cube_grasps=True,
            position_noise=0.01,
            yaw_noise=0.1,
            duration_scale_min=0.9,
            duration_scale_max=1.2,
        )
    )


@dataclass
class Replay:
    command: ClassVar[str] = "replay"
    root: Path = tyro.MISSING
    episode: int = 0
    restore_frames: bool = False
    video: Path | None = None


@dataclass
class View:
    """Interactive recorded-state playback in MuJoCo (WSLg supported)."""

    command: ClassVar[str] = "view"
    root: Path = tyro.MISSING
    episode: int = 0
    speed: float = 1.0
    paused: bool = False
    loop: bool = True
    camera: Literal["free", "front", "wrist"] = "free"
    seconds: float | None = None


@dataclass
class AuditContacts:
    """Audit recorded states for deep or unintended robot/environment contacts."""

    command: ClassVar[str] = "audit-contacts"
    root: Path = tyro.MISSING
    episode: int = 0
    max_nonpad_penetration: float = 0.001
    max_penetration: float = 0.003
    output: Path | None = None


@dataclass
class Validate:
    command: ClassVar[str] = "validate"
    output: Path = tyro.MISSING
    envs: list[str] | None = None
    attempts: int = 1
    seed: int = 1000
    size: int = 64
    record_images: bool = False
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    planner: PlannerConfig = field(
        default_factory=lambda: PlannerConfig(
            episodes=5, candidates=8, horizon=8, iterations=1
        )
    )


COMMANDS = {
    cls.command: cls()
    for cls in (
        ListTasks,
        Doctor,
        Generate,
        Benchmark,
        Export,
        Rerender,
        Inspect,
        Diversity,
        Ablate,
        Replay,
        View,
        AuditContacts,
        Validate,
    )
}


def yaml_arguments(values, prefix=""):
    """Flatten YAML fields into typed Tyro arguments, including nested planner fields."""
    if not isinstance(values, dict):
        raise TypeError("Configuration must be a YAML mapping")
    result = []
    for key, value in values.items():
        if not isinstance(key, str) or key.startswith("-"):
            raise ValueError("Configuration keys must be field names")
        name = prefix + key.replace("_", "-")
        if isinstance(value, dict):
            result.extend(yaml_arguments(value, name + "."))
        elif isinstance(value, bool):
            result.append(
                "--" + (name if value else prefix + "no-" + key.replace("_", "-"))
            )
        else:
            result.append("--" + name)
            result.extend(
                str(v) if v is not None else "None"
                for v in (value if isinstance(value, list) else [value])
            )
    return result


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config = None
    for index, token in enumerate(argv):
        if token == "--config" or token.startswith("--config="):
            if token == "--config":
                if index + 1 == len(argv):
                    raise ValueError("--config requires a YAML path")
                config = argv.pop(index + 1)
            else:
                config = token.split("=", 1)[1]
            argv.pop(index)
            break
    if config is not None:
        if not argv or argv[0] not in COMMANDS:
            raise ValueError("Select a command when using --config")
        values = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
        argv = [argv[0], *yaml_arguments(values), *argv[1:]]
    return tyro.cli(
        tyro.extras.subcommand_type_from_defaults(COMMANDS),
        args=argv,
        description="OGBench GPU planning and LeRobot demonstrations. --config PATH loads YAML; CLI flags override it.",
    )


def main(argv=None):
    args = parse_args(argv)
    if args.command == "rerender":
        from .rerender import rerender

        result = rerender(args.source, args.output, args.batch_size)
    elif args.command == "list-tasks":
        from .tasks import make_env, task_registry

        result = []
        for env_id, spec in task_registry().items():
            env = make_env(env_id)
            try:
                result.append(
                    {
                        "env": env_id,
                        "task_ids": list(range(1, env.unwrapped.num_tasks + 1)),
                        "max_steps": spec.max_episode_steps,
                    }
                )
            finally:
                env.close()
    elif args.command == "doctor":
        import torch

        from .environment import BatchEnvironment
        from .io import versions
        from .tasks import make_env

        env = make_env(args.env)
        try:
            sim = BatchEnvironment(env, 2, PlannerConfig(candidates=2))
            sim.step(torch.zeros((2, 5), device=sim.device))
            images = sim.render(0)
            if not bool(sim.valid().all()):
                raise RuntimeError("Physics capacity or numerical validation failed")
            result = {
                "versions": versions(),
                "gpu": torch.cuda.get_device_name(),
                "images": {key: list(value.shape) for key, value in images.items()},
                "valid": sim.valid().cpu().tolist(),
                "gripper": sim.grip.cpu().tolist(),
            }
        finally:
            env.close()
    elif args.command in ("generate", "benchmark"):
        config = args.planner
        if args.command == "generate":
            from .recording import generate as generate_run

            result = generate_run(
                args.output,
                args.env,
                args.episodes,
                args.task_ids,
                args.seed,
                config,
                args.size,
                args.max_steps,
                record_images=args.record_images,
                randomization=args.randomization,
                refill_slots=args.refill_slots,
                batched_cpu=args.batched_cpu,
            )
        else:
            result = benchmark(args.env, config, args.steps, args.size)
    elif args.command == "export":
        from .dataset import export_dataset

        result = export_dataset(
            args.source,
            args.output,
            args.repo_id,
            args.outcome,
            args.require_contact_valid,
            diverse_per_task=args.diverse_per_task,
            streaming_encoding=args.streaming_encoding,
            encoder_queue_size=args.encoder_queue_size,
            encoder_threads=args.encoder_threads,
            quality=args.quality,
        )
    elif args.command == "inspect":
        from .dataset import load_dataset

        dataset = load_dataset(
            args.root, args.chunk_length, args.outcome, args.require_contact_valid, args.quality
        )
        item = dataset[0]
        result = {
            "frames": len(dataset),
            "episodes": dataset.num_episodes,
            "features": {
                key: list(value.shape) if hasattr(value, "shape") else value
                for key, value in item.items()
            },
        }
    elif args.command == "diversity":
        from .diversity import measure_diversity
        from .io import write_json

        result = measure_diversity(args.source, args.samples)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    elif args.command == "ablate":
        from .diversity import run_ablations

        result = run_ablations(
            args.output,
            args.env,
            args.task_id,
            args.variants,
            args.seed,
            args.planner,
            args.randomization,
            args.size,
            args.record_images,
        )
    elif args.command == "validate":
        result = validate_tasks(
            args.output,
            args.envs,
            args.attempts,
            args.seed,
            args.size,
            args.planner,
            record_images=args.record_images,
            randomization=args.randomization,
        )
        print(json.dumps(result, indent=2))
        if not result["all_tasks_succeeded"]:
            raise SystemExit(1)
        return
    elif args.command == "audit-contacts":
        from .contacts import audit_contacts
        from .io import write_json

        result = audit_contacts(
            args.root, args.episode, args.max_nonpad_penetration, args.max_penetration
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    elif args.command == "view":
        from .viewer import view

        result = view(
            args.root,
            args.episode,
            args.speed,
            args.paused,
            args.loop,
            args.camera,
            args.seconds,
        )
    else:
        from .dataset import replay as replay_run

        result = replay_run(args.root, args.episode, args.restore_frames, args.video)
    print(json.dumps(result, indent=2))


def validate_tasks(
    output, envs, attempts, seed, size, config, record_images=False, randomization=None
):
    from .contacts import audit_contacts
    from .io import episode_metadata, write_json
    from .recording import generate
    from .tasks import make_env, task_registry

    if attempts < 1:
        raise ValueError("attempts must be positive")
    randomization = randomization or RandomizationConfig()
    report = {
        "all_tasks_succeeded": True,
        "all_successes_contact_valid": True,
        "environments": [],
    }
    for env_id in envs or task_registry():
        env = make_env(env_id)
        task_ids = list(range(1, env.unwrapped.num_tasks + 1))
        env.close()
        result = generate(
            output / env_id,
            env_id,
            len(task_ids) * attempts * randomization.variants_per_reset,
            task_ids,
            seed,
            config,
            size=size,
            record_images=record_images,
            randomization=randomization,
        )
        audits, successes = [], set()
        for row in episode_metadata(output / env_id):
            audit = audit_contacts(
                output / env_id,
                row["episode_id"],
                config.max_nonpad_penetration,
                config.max_penetration,
            )
            quality = row.get("contact_quality", {})
            audit.update(task_id=row["task_id"], seed=row["seed"], substeps=quality)
            audits.append(audit)
            if row["outcome"] == "success":
                valid = audit["contact_valid"] and quality.get("valid") is True
                report["all_successes_contact_valid"] &= valid
                if valid:
                    successes.add(row["task_id"])
        missing = sorted(set(task_ids) - successes)
        report["environments"].append(
            {
                "env": env_id,
                "missing_successes": missing,
                "summary": result,
                "contact_audits": audits,
            }
        )
        report["all_tasks_succeeded"] &= not missing
        write_json(output / "validation.json", report)
    return report


def benchmark(env_id, config, steps, size=32):
    import numpy as np
    import torch

    from .environment import BatchEnvironment
    from .planner import SamplingMPC
    from .skills import SkillPlan
    from .tasks import make_env

    if steps < 1:
        raise ValueError("steps must be positive")
    env = make_env(env_id, size=size)
    try:
        sim = BatchEnvironment(env, config.episodes, config)
        sim.reset(0)
        planner = SamplingMPC(sim, config)
        refs = []
        for world in range(sim.worlds):
            sim.sync_cpu(world)
            refs.append(SkillPlan(env, world).references(config.horizon))
        refs = np.stack(refs)
        action, _ = planner.plan(refs)
        sim.step(action)
        latencies = []
        start = time.perf_counter()
        for _ in range(steps):
            action, metrics = planner.plan(refs)
            sim.step(action)
            latencies.append(metrics["seconds"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        render_start = time.perf_counter()
        sim.render_batch()
        render_setup_seconds = time.perf_counter() - render_start
        render_start = time.perf_counter()
        for _ in range(steps):
            sim.render_batch()
        render_seconds = (time.perf_counter() - render_start) / steps
        free, total = torch.cuda.mem_get_info()
        return {
            "config": config.to_dict(),
            "steps": steps,
            "seconds": elapsed,
            "execution_steps_per_second": sim.worlds * steps / elapsed,
            "rollout_steps_per_second": sim.worlds
            * config.candidates
            * config.horizon
            * config.iterations
            * steps
            / elapsed,
            "planner_seconds": latencies,
            "render_setup_seconds": render_setup_seconds,
            "render_batch_seconds": render_seconds,
            "rendered_images": 2 * sim.worlds,
            "image_size": size,
            "device_used_bytes": total - free,
            "torch_peak_bytes": torch.cuda.max_memory_allocated(),
            "valid": sim.valid().cpu().tolist(),
            "success": sim.success().cpu().tolist(),
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
