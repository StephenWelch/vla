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
class TrainingConfig:
    policy: Path | None = None
    policy_type: Literal["smolvla", "act", "pi05"] = "smolvla"
    repo_id: str | None = None
    camera_keys: list[str] | None = None
    camera_names: list[str] | None = None
    episodes: list[int] | None = None
    steps: int = 20_000
    batch_size: int = 8
    workers: int = 2
    video_backend: Literal["pyav", "torchcodec"] = "pyav"
    save_freq: int = 5_000
    seed: int = 1000
    device: str = "cuda"
    hf_home: Path | None = None
    overrides: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    validation_fraction: float = 0.0
    image_size: tuple[int, int] | None = None  # Policy input (height, width).
    use_amp: bool = False
    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    image_normalization: Literal["auto", "imagenet", "dataset"] = "auto"
    loss_eval_freq: int = 1000
    rollout_eval_freq: int = 2000
    probe_frames: int = 1024
    eval_episodes: int = 10
    train_eval_seeds: list[int] | None = None
    val_eval_seeds: list[int] | None = None
    eval_max_steps: int = 250
    resume: Path | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(kw_only=True)
class TrainConfig(TrainingConfig):
    dataset: Path
    output: Path


def normalize_record(values):
    """Read current-era saved experiments; new public configs use only canonical fields."""
    values = dict(values)
    values.pop("backend", None)
    if "absolute_arm" in values or "absolute_gripper" in values:
        arm = values.pop("absolute_arm", False)
        gripper = values.pop("absolute_gripper", False)
        if arm and not gripper:
            raise ValueError("Absolute arm targets require absolute gripper targets")
        values["action_mode"] = (
            "absolute" if arm else "absolute_gripper" if gripper else "delta"
        )
    if isinstance(values.get("overrides"), list):
        values["overrides"] = dict(item.split("=", 1) for item in values["overrides"])
    return values


def training_plan(config, validate_profiles=None, worker="vla_tools.hooks"):
    """Validate the data/checkpoint contract and build native LeRobot arguments."""
    if config.use_amp and not config.device.startswith("cuda"):
        raise ValueError("Explicit ACT AMP requires CUDA")
    if config.image_size is not None and (
        len(config.image_size) != 2 or min(config.image_size) < 16
    ):
        raise ValueError("image_size must be (height, width), each >=16")
    if (
        config.image_size is not None or config.use_amp
    ) and config.policy_type != "act":
        raise ValueError("Explicit image resizing and AMP currently support ACT")
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
        inputs[name] = {
            "type": "VISUAL",
            "shape": [3, *config.image_size] if config.image_size else shape,
        }
    manifest_path = config.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rendering, actions = (
        validate_profiles(config) if validate_profiles else (None, None)
    )
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
        "dataset.video_backend": config.video_backend,
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
    if config.policy_type == "act":
        native["policy.use_amp"] = config.use_amp
    if config.wandb.entity:
        native["wandb.entity"] = config.wandb.entity
    if getattr(config, "percentile_normalization", False):
        native["policy.normalization_mapping"] = {
            "VISUAL": "MEAN_STD",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        }
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
    if (
        config.train_eval_seeds is not None or config.val_eval_seeds is not None
    ) and not config.validation_fraction:
        raise ValueError("Explicit evaluation seeds require validation")
    if config.validation_fraction:
        import torch

        from vla_tools.hooks import scene_split

        if torch.device(config.device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "Periodic simulator evaluation requires the CUDA/WSL policy environment"
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
        for name in ("train", "val"):
            seeds = getattr(config, f"{name}_eval_seeds")
            if seeds is not None:
                available = {r["seed"] for r in split[name]}
                if (
                    not seeds
                    or len(set(seeds)) != len(seeds)
                    or not set(seeds) <= available
                ):
                    raise ValueError(
                        f"Evaluation seeds must be unique members of {name}"
                    )
                if len(seeds) > config.eval_episodes:
                    raise ValueError("Explicit seeds exceed eval_episodes")
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
        native["dataset.use_imagenet_stats"] = (
            config.policy_type == "act"
            if config.image_normalization == "auto"
            else config.image_normalization == "imagenet"
        )
    elif config.image_normalization != "auto":
        native["dataset.use_imagenet_stats"] = config.image_normalization == "imagenet"
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
        old["config"] = normalize_record(old["config"])
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
            "train_eval_seeds",
            "image_normalization",
            "val_eval_seeds",
        )
        current = json.loads(json.dumps(asdict(config), default=str))
        if any(
            old["config"].get(k, False) != current.get(k, False)
            for k in ("action_mode", "percentile_normalization")
        ):
            raise ValueError("Resume changes action representation or normalization")
        for key, default in (
            ("image_size", None),
            ("use_amp", False),
            ("amp_dtype", "bfloat16"),
        ):
            if old["config"].get(key, default) != current[key]:
                raise ValueError("Resume changes image size or precision")
        if any(old["config"].get(key) != current[key] for key in immutable):
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
    from omegaconf import OmegaConf

    overrides = OmegaConf.to_container(
        OmegaConf.from_dotlist(
            [f"{key}={value}" for key, value in config.overrides.items()]
        ),
        resolve=True,
    )
    # Typed fields own their settings; arbitrary native settings remain available.
    for key in config.overrides:
        if any(
            key == field or key.startswith(field + ".") or field.startswith(key + ".")
            for field in native
        ):
            raise ValueError(
                "Use the typed CLI fields to override dataset/training settings"
            )
    native_values = {}
    for key, value in native.items():
        target = native_values
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    native_values = OmegaConf.to_container(
        OmegaConf.merge(native_values, overrides), resolve=True
    )
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
        "schema_version": 2,
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
        "native": native_values,
        "command": [sys.executable, "-m", worker],
    }


def native_config(record):
    """Construct the native typed config once in the isolated worker."""
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from omegaconf import OmegaConf

    values = dict(record["native"])
    resume_path = values.pop("config_path", None)
    policy = dict(values.get("policy", {}))
    pretrained = policy.pop("path", None)
    if resume_path:
        base = json.loads(Path(resume_path).read_text())
        # LeRobot's validate() resolves resume state from this one hint.
        sys.argv = [sys.argv[0], f"--config_path={resume_path}"]
    elif pretrained:
        base = {"policy": json.loads((Path(pretrained) / "config.json").read_text())}
        policy["pretrained_path"] = pretrained
        sys.argv = sys.argv[:1]
    else:
        base = {}
        sys.argv = sys.argv[:1]
    values["policy"] = policy
    merged = OmegaConf.to_container(OmegaConf.merge(base, values), resolve=True)
    # Feature maps are replacements, not recursive additions to pretrained cameras.
    for key in ("input_features", "output_features", "normalization_mapping"):
        if key in policy:
            merged["policy"][key] = policy[key]
    import importlib

    kind = merged["policy"]["type"]
    importlib.import_module(f"lerobot.policies.{kind}.configuration_{kind}")
    return draccus.decode(TrainPipelineConfig, merged)


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


def train(config, validate_profiles=None, worker="vla_tools.hooks"):
    plan = training_plan(config, validate_profiles, worker)
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
