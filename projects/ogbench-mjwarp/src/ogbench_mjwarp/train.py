"""Train ACT, SmolVLA or pi0.5 on local LeRobot datasets; defaults < YAML < CLI."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from vla_tools.config import parse_args
from vla_tools.policy import checkpoint_step, runtime_environment
from vla_tools.tracking import WandbConfig


@dataclass
class TrainConfig:
    dataset: Path
    policy: Path | None
    output: Path
    policy_type: Literal["smolvla", "act", "pi05"] = "smolvla"
    repo_id: str | None = None
    camera_keys: list[str] | None = None
    camera_names: list[str] | None = None
    episodes: list[int] | None = None
    steps: int = 20_000
    batch_size: int = 8
    workers: int = 2
    save_freq: int = 5_000
    seed: int = 1000
    device: str = "cuda"
    hf_home: Path | None = None
    overrides: list[str] = field(default_factory=list)
    dry_run: bool = False
    validation_fraction: float = 0.0
    loss_eval_freq: int = 1000
    rollout_eval_freq: int = 2000
    probe_frames: int = 1024
    eval_episodes: int = 10
    eval_max_steps: int = 250
    resume: Path | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


def training_plan(config):
    """Validate the data/checkpoint contract and build native LeRobot arguments."""
    if min(config.steps, config.batch_size, config.save_freq) < 1 or config.workers < 0:
        raise ValueError(
            "Steps, batch size and save frequency must be positive; workers >=0"
        )
    if config.output.exists() and config.resume is None:
        raise FileExistsError(f"Output already exists: {config.output}")
    if (
        config.policy is not None
        and not (config.policy / "model.safetensors").is_file()
    ):
        raise FileNotFoundError(f"Missing checkpoint: {config.policy}")
    if config.policy is None and config.policy_type != "act":
        raise ValueError("SmolVLA and pi0.5 require a pretrained checkpoint")
    if any(
        (config.dataset / name).exists() for name in ("INCOMPLETE", "INCOMPLETE.json")
    ):
        raise ValueError("Dataset export is incomplete")
    info_path = config.dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    policy_path = config.policy / "config.json" if config.policy is not None else None
    policy = json.loads(policy_path.read_text()) if policy_path else {"type": "act"}
    if policy["type"] != config.policy_type:
        raise ValueError("Checkpoint type differs from --policy-type")
    if info["total_episodes"] < 1 or info["total_frames"] < 1:
        raise ValueError("Dataset is empty")
    features = info["features"]
    inputs = {}
    for key, kind, limit in (
        ("observation.state", "STATE", "max_state_dim"),
        ("action", "ACTION", "max_action_dim"),
    ):
        shape = features[key]["shape"]
        maximum = policy.get(limit, shape[0])
        if len(shape) != 1 or not 0 < shape[0] <= maximum:
            raise ValueError(
                f"{key} shape {shape} exceeds checkpoint {limit}={maximum}"
            )
        if kind == "STATE":
            inputs[key] = {"type": kind, "shape": shape}
    keys = config.camera_keys or [
        k for k in features if k.startswith("observation.images.")
    ]
    names = (
        config.camera_names
        or [
            k
            for k, v in policy.get("input_features", {}).items()
            if v["type"] == "VISUAL"
        ][: len(keys)]
        or keys
    )
    if not keys or len(keys) != len(names) or len(set(names)) != len(names):
        raise ValueError("Choose one unique checkpoint camera name per dataset camera")
    if len(set(keys)) != len(keys):
        raise ValueError("Dataset camera keys must be unique")
    rename = {}
    for key, name in zip(keys, names, strict=True):
        if not key.startswith("observation.images.") or key not in features:
            raise ValueError(f"Missing dataset camera: {key}")
        if config.policy is not None and (
            name not in policy["input_features"]
            or policy["input_features"][name]["type"] != "VISUAL"
        ):
            raise ValueError(f"Missing checkpoint camera: {name}")
        shape = features[key]["shape"]
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Camera must use RGB CHW: {key} {shape}")
        if key != name:
            rename[key] = name
        inputs[name] = {"type": "VISUAL", "shape": shape}
    manifest_path = config.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    from ogbench_mjwarp.profile import ogbench_profile

    rendering = ogbench_profile(config.dataset, config.resume)
    from ogbench_mjwarp.actions import action_profile

    actions = action_profile(config.dataset, config.resume) if rendering else None
    repo_id = config.repo_id or manifest.get("repo_id")
    if not repo_id:
        raise ValueError("Pass --repo-id for datasets without repo_id in manifest.json")
    if config.episodes is not None and (
        not config.episodes
        or len(set(config.episodes)) != len(config.episodes)
        or any(i < 0 or i >= info["total_episodes"] for i in config.episodes)
    ):
        raise ValueError("Episodes must be unique valid exported dataset indices")
    native = {
        "dataset.repo_id": repo_id,
        "dataset.root": str(config.dataset.resolve()),
        "dataset.video_backend": "pyav",
        "dataset.eval_split": 0,
        "policy.device": config.device,
        "policy.push_to_hub": False,
        "policy.input_features": inputs,
        "save_checkpoint_to_hub": False,
        "output_dir": str(config.output.resolve()),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "save_freq": config.save_freq,
        "seed": config.seed,
        "eval_steps": 0,
        "env_eval_freq": 0,
        "wandb.enable": config.wandb.enable and config.wandb.mode != "disabled",
        "wandb.project": config.wandb.project,
        "wandb.mode": config.wandb.mode,
        "wandb.disable_artifact": True,
    }
    if config.wandb.entity:
        native["wandb.entity"] = config.wandb.entity
    if config.policy is not None:
        native["policy.path"] = str(config.policy.resolve())
        native["rename_map"] = rename
    else:
        if rename:
            raise ValueError(
                "ACT from scratch uses dataset camera names without renaming"
            )
        native["policy.type"] = config.policy_type
    if config.episodes is not None:
        native["dataset.episodes"] = config.episodes
    split = None
    if config.validation_fraction:
        import torch

        from ogbench_mjwarp.hooks import scene_split
        from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig  # noqa: F401

        if torch.device(config.device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "Periodic OGBench evaluation requires the CUDA/WSL policy environment"
            )

        if (
            min(
                config.loss_eval_freq,
                config.rollout_eval_freq,
                config.probe_frames,
                config.eval_episodes,
                config.eval_max_steps,
            )
            < 1
        ):
            raise ValueError("Evaluation budgets must be positive")
        if config.rollout_eval_freq % config.loss_eval_freq:
            raise ValueError("Rollout frequency must be a multiple of loss frequency")
        split = scene_split(
            config.dataset, config.validation_fraction, config.seed, config.episodes
        )
        tasks = {
            (row["env_id"], row["task_id"])
            for name in ("train", "val")
            for row in split[name]
        }
        if len(tasks) != 1:
            raise ValueError(
                "Periodic rollouts require one environment and task per dataset"
            )
        native["save_freq"] = config.loss_eval_freq
        native["dataset.use_imagenet_stats"] = False
    if config.resume:
        checkpoint_step(config.resume)
        if not (config.resume / "train_config.json").is_file():
            raise FileNotFoundError(
                "Resume requires a pretrained_model directory with train_config.json"
            )
        native.pop("policy.path", None)
        native.pop("policy.type", None)
        native["resume"] = True
        native["config_path"] = str(config.resume.resolve() / "train_config.json")
        old_path = config.output / "experiment.json"
        if not old_path.exists():
            old_path = config.output.with_name(config.output.name + ".experiment.json")
        old = json.loads(old_path.read_text())
        immutable = (
            "dataset",
            "policy_type",
            "camera_keys",
            "camera_names",
            "episodes",
            "batch_size",
            "seed",
            "validation_fraction",
            "overrides",
            "loss_eval_freq",
            "rollout_eval_freq",
            "probe_frames",
            "eval_episodes",
            "eval_max_steps",
        )
        current = json.loads(json.dumps(asdict(config), default=str))
        if any(old["config"][key] != current[key] for key in immutable):
            raise ValueError(
                "Resume changes dataset, architecture, split, or training configuration"
            )
        if split != old.get("split"):
            raise ValueError("Resume split or dataset provenance changed")
        saved_step = json.loads(
            (config.resume.parent / "training_state/training_step.json").read_text()
        )["step"]
        if config.steps <= saved_step:
            raise ValueError("Total updates must exceed the resumed checkpoint step")
        completed_steps = [
            json.loads(path.read_text())["step"]
            for path in (config.output / "metrics").glob("*.json")
            if json.loads(path.read_text())["status"] == "complete"
        ]
        if completed_steps and max(completed_steps) > saved_step:
            raise ValueError(
                "Resume from the latest checkpoint, not an older evaluated checkpoint"
            )
    args = [
        f"--{key}={json.dumps(value) if isinstance(value, (dict, list, bool)) else value}"
        for key, value in native.items()
    ]
    for override in config.overrides:
        if "=" not in override or override.startswith("-"):
            raise ValueError("Overrides must use native LeRobot key=value syntax")
        if override.split("=", 1)[0] in native:
            raise ValueError(
                "Use the typed CLI fields to override dataset/training settings"
            )
        args.append("--" + override)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            info_path,
            config.dataset / "meta/stats.json",
            manifest_path,
            policy_path,
        )
        if p is not None and p.is_file()
    }
    if config.resume and hashes != old["metadata_sha256"]:
        raise ValueError("Resume dataset metadata or starting policy changed")
    return {
        "schema_version": 1,
        "versions": {
            name: importlib.metadata.version(name) for name in ("lerobot", "torch")
        },
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "dataset": {
            "repo_id": repo_id,
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "selected_episodes": config.episodes,
            "state": features["observation.state"],
            "action": features["action"],
            "provenance": str(manifest_path.resolve()) if manifest else None,
        },
        "camera_map": rename,
        "rendering": rendering,
        "action_profile": actions,
        "split": split,
        "metadata_sha256": hashes,
        "command": (
            [sys.executable, "-m", "ogbench_mjwarp.hooks"]
            if split or rendering or config.wandb.enable
            else [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
        )
        + args,
    }


def run_training(command, environment, log_path):
    """Stream native training output to the terminal and a persistent run log."""
    with (
        log_path.open("a", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process,
    ):
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def train(config):
    plan = training_plan(config)
    preview = {
        **plan,
        "split": {name: len(plan["split"][name]) for name in ("train", "val")}
        if plan["split"]
        else None,
    }
    print(json.dumps(preview, indent=2), flush=True)
    if config.dry_run:
        return
    record = config.output.with_name(config.output.name + ".experiment.json")
    if record.exists() and config.resume is None:
        raise FileExistsError(f"Experiment record already exists: {record}")
    record.parent.mkdir(parents=True, exist_ok=True)
    plan["status"] = "running"
    record.write_text(json.dumps(plan, indent=2) + "\n")
    log_path = config.output.with_name(config.output.name + ".train.log")
    if config.resume:
        saved_step = json.loads(
            (config.resume.parent / "training_state/training_step.json").read_text()
        )["step"]
        metrics_path = config.output / "optimizer_metrics.jsonl"
        if metrics_path.exists():
            # Updates after the checkpoint were lost; remove their stale optimizer metrics.
            lines = [
                line
                for line in metrics_path.read_text().splitlines()
                if json.loads(line)["step"] <= saved_step
            ]
            metrics_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    try:
        environment = runtime_environment(config.hf_home)
        if plan["split"] or plan["rendering"] or config.wandb.enable:
            environment["VLA_TRAIN_SETTINGS"] = str(record.resolve())
        run_training(plan["command"], environment, log_path)
        plan["status"] = "complete"
    except BaseException:
        plan["status"] = "failed_or_interrupted"
        raise
    finally:
        content = json.dumps(plan, indent=2) + "\n"
        record.write_text(content)
        if config.output.is_dir():
            (config.output / "experiment.json").write_text(content)
            if log_path.exists():
                import shutil

                shutil.copyfile(log_path, config.output / "train.log")


def main(argv=None):
    return train(parse_args(TrainConfig, argv))


if __name__ == "__main__":
    main()
