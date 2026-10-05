"""Single-task LIBERO ACT training and simulator testing; defaults < YAML < CLI."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.policy import latest_checkpoint, runtime_environment
from vla_tools.tracking import WandbConfig, install_native_logging, write_json


@dataclass
class Config:
    suite: str = "libero_goal"
    task_id: int = 0
    instruction: str = "open the middle drawer of the cabinet"
    repo_id: str = "lerobot/libero"
    revision: str = "main"
    output: Path = Path("outputs/libero-drawer-act")
    cache: Path = Path("outputs/libero-cache")
    assets: Path = Path("outputs/libero/libero-assets")
    steps: int = 20_000
    batch_size: int = 8
    workers: int = 2
    chunk_size: int = 16
    validation_fraction: float = 0.2
    loss_eval_freq: int = 1000
    rollout_eval_freq: int = 2000
    eval_episodes: int = 10
    test_episodes: int = 20
    seed: int = 1000
    test_seed: int = 2027
    device: str = "cuda"
    dry_run: bool = False
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(project="vla-libero")
    )


def select_episodes(rows, instruction, fraction):
    """Match language rather than confusing simulator IDs with dataset task indices."""
    normalize = lambda text: " ".join(text.lower().replace("_", " ").split())
    ids = sorted(
        row["episode_index"]
        for row in rows
        if len(row["tasks"]) == 1
        and normalize(row["tasks"][0]) == normalize(instruction)
    )
    if len(ids) != len(set(ids)) or len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError(
            "Need unique episodes for the exact task and a valid holdout fraction"
        )
    count = math.ceil(len(ids) * fraction)
    if count == len(ids):
        raise ValueError("Holdout leaves no training episodes")
    return {"train": ids[:-count], "val": ids[-count:]}


def episode_rows(meta, root):
    """Recover episode language from frame task indices when metadata omits it."""
    if "tasks" in meta.episodes.column_names:
        return meta.episodes
    import pyarrow.dataset as ds
    from huggingface_hub import snapshot_download

    snapshot_download(
        meta.repo_id,
        repo_type="dataset",
        revision=meta.revision,
        local_dir=root,
        allow_patterns=["data/**"],
    )
    labels = {int(row.task_index): language for language, row in meta.tasks.iterrows()}
    table = ds.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "task_index"]
    )
    pairs = table.group_by(["episode_index", "task_index"]).aggregate([]).to_pylist()
    tasks = {}
    for row in pairs:
        tasks.setdefault(row["episode_index"], []).append(labels[row["task_index"]])
    return [
        {"episode_index": index, "tasks": languages}
        for index, languages in tasks.items()
    ]


def native_config(config, episodes, revision, root):
    from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.act.configuration_act import ACTConfig

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=config.repo_id,
            root=str(root),
            revision=revision,
            episodes=episodes,
            eval_split=config.validation_fraction,
            video_backend="pyav",
            return_uint8=True,
            use_imagenet_stats=True,
        ),
        policy=ACTConfig(
            device=config.device,
            push_to_hub=False,
            chunk_size=config.chunk_size,
            n_action_steps=config.chunk_size,
        ),
        env=LiberoEnv(
            task=config.suite,
            task_ids=[config.task_id],
            observation_height=256,
            observation_width=256,
            control_mode="relative",
            max_parallel_tasks=1,
        ),
        output_dir=config.output.resolve(),
        job_name=config.output.name,
        steps=config.steps,
        batch_size=config.batch_size,
        num_workers=config.workers,
        seed=config.seed,
        eval_steps=config.loss_eval_freq,
        env_eval_freq=config.rollout_eval_freq,
        save_freq=config.rollout_eval_freq,
        max_eval_samples=1024,
        log_freq=100,
        eval=EvalConfig(
            n_episodes=config.eval_episodes, batch_size=1, use_async_envs=False
        ),
        wandb=WandBConfig(
            enable=config.wandb.enable and config.wandb.mode != "disabled",
            disable_artifact=True,
            project=config.wandb.project,
        ),
    )


def test_command(config, checkpoint):
    values = {
        "policy.path": str(checkpoint),
        "policy.device": config.device,
        "env.type": "libero",
        "env.task": config.suite,
        "env.task_ids": [config.task_id],
        "env.control_mode": "relative",
        "env.observation_height": 256,
        "env.observation_width": 256,
        "env.max_parallel_tasks": 1,
        "eval.batch_size": 1,
        "eval.n_episodes": config.test_episodes,
        "eval.use_async_envs": False,
        "seed": config.test_seed,
        "output_dir": str(config.output.resolve() / "test"),
    }
    return [sys.executable, "-m", "lerobot.scripts.lerobot_eval"] + [
        f"--{key}={json.dumps(value) if isinstance(value, (list, bool)) else value}"
        for key, value in values.items()
    ]


def configure_libero(config):
    """Keep LIBERO's configuration in the workspace and reuse existing assets."""
    import yaml

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Install lerobot[libero]==0.6.1 in the Linux training environment"
        )
    if not config.assets.is_dir():
        raise FileNotFoundError(f"Missing LIBERO assets: {config.assets}")
    package = Path(spec.origin).parent / "libero"
    # hf-libero 0.1.4 resolves meshes from package/assets, ignoring YAML's asset path.
    package_assets = package / "assets"
    if not package_assets.exists():
        package_assets.symlink_to(config.assets.resolve(), target_is_directory=True)
    directory = config.cache.resolve() / "libero-config"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": package,
        "bddl_files": package / "bddl_files",
        "init_states": package / "init_files",
        "datasets": config.cache.resolve(),
        "assets": config.assets.resolve(),
    }
    (directory / "config.yaml").write_text(
        yaml.safe_dump({k: str(v) for k, v in paths.items()})
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    from libero.libero import benchmark

    task = benchmark.get_benchmark_dict()[config.suite](task_order_index=0).get_task(
        config.task_id
    )
    if task.language.lower().strip() != config.instruction.lower().strip():
        raise ValueError(f"Simulator task language differs: {task.language}")
    return task.name


def install_data_and_checkpoint_hooks(trainer, split):
    """Use training-only state/action statistics and retain one resumable checkpoint."""
    import numpy as np
    import torch
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.factory import make_dataset
    from lerobot.utils.constants import IMAGENET_STATS

    original_save = trainer.save_checkpoint

    def datasets(cfg):
        # The compact Hub dataset omits episode task labels and episode stats.
        # Use native loaders with explicit IDs, avoiding the native label-based splitter.
        datasets = []
        for name in ("train", "val"):
            selected_cfg = copy.deepcopy(cfg)
            selected_cfg.dataset.episodes = split[name]
            selected_cfg.dataset.eval_split = 0
            if name == "val":
                selected_cfg.dataset.image_transforms.enable = False
            datasets.append(make_dataset(selected_cfg))
        train, val = datasets
        stats = dict(train.meta.stats)
        for key in ("observation.state", "action"):
            values = np.asarray(train.hf_dataset[key], dtype=np.float32)
            stats[key] = get_feature_stats(values, axis=0, keepdims=False)
        for key in train.meta.camera_keys:
            stats[key] = {
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in IMAGENET_STATS.items()
            }
        train.meta.stats = val.meta.stats = stats
        return train, val

    def save(**kwargs):
        original_save(**kwargs)
        latest = Path(kwargs["checkpoint_dir"])
        for path in latest.parent.iterdir():
            if (
                path.is_dir()
                and not path.is_symlink()
                and path.name.isdecimal()
                and path != latest
            ):
                shutil.rmtree(path)

    trainer.make_train_eval_datasets = datasets
    trainer.save_checkpoint = save


def main(argv=None):
    config = parse_args(Config, argv)
    if (
        min(
            config.steps,
            config.batch_size,
            config.chunk_size,
            config.loss_eval_freq,
            config.rollout_eval_freq,
            config.eval_episodes,
            config.test_episodes,
        )
        < 1
        or config.workers < 0
    ):
        raise ValueError("Budgets must be positive; workers must be nonnegative")
    if not 0 < config.validation_fraction < 1 or config.seed == config.test_seed:
        raise ValueError("Require a holdout fraction and a separate test seed")
    if config.output.exists():
        raise FileExistsError(f"Choose a fresh output: {config.output}")
    if config.dry_run:
        print(json.dumps(asdict(config), default=str, indent=2))
        return
    if sys.platform != "linux":
        raise RuntimeError(
            "LIBERO simulator training/evaluation requires Linux (WSL or Docker)"
        )
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("This workflow requires LeRobot 0.6.1")
    os.environ.update(runtime_environment(config.cache / "hf"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    task_name = configure_libero(config)
    from huggingface_hub import HfApi
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    revision = HfApi().dataset_info(config.repo_id, revision=config.revision).sha
    root = config.cache.resolve() / "datasets" / revision
    meta = LeRobotDatasetMetadata(config.repo_id, root=root, revision=revision)
    split = select_episodes(
        episode_rows(meta, root), config.instruction, config.validation_fraction
    )
    if tuple(meta.features["observation.state"]["shape"]) != (8,) or tuple(
        meta.features["action"]["shape"]
    ) != (7,):
        raise ValueError("Expected LIBERO relative state8/action7 dataset")
    cameras = [key for key in meta.features if key.startswith("observation.images.")]
    if set(cameras) != {"observation.images.image", "observation.images.image2"} or any(
        tuple(meta.features[key]["shape"]) not in ((256, 256, 3), (3, 256, 256))
        for key in cameras
    ):
        raise ValueError("Expected agentview and wrist cameras at 256x256 RGB")
    record = {
        "status": "running",
        "config": asdict(config),
        "split": split,
        "dataset": {
            "repo_id": config.repo_id,
            "revision": revision,
            "task_name": task_name,
            "instruction": config.instruction,
            "root": str(root),
            "fps": meta.fps,
        },
        "camera_map": {key: key for key in cameras},
        "metadata_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "meta").rglob("*.parquet")
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("lerobot", "torch")
        },
    }
    record_path = config.output.with_name(config.output.name + ".experiment.json")
    if record_path.exists():
        raise FileExistsError(f"Choose a fresh experiment record: {record_path}")
    write_json(record_path, record)
    from lerobot.scripts import lerobot_train

    get_tracker = install_native_logging(
        lerobot_train, {"wandb": asdict(config.wandb)}, record
    )
    install_data_and_checkpoint_hooks(lerobot_train, split)
    cfg = native_config(config, sorted(split["train"] + split["val"]), revision, root)
    failed = True
    try:
        # Native validation reads sys.argv for pretrained/resume options.
        sys.argv = [sys.argv[0]]
        lerobot_train.train(cfg)
        checkpoint = latest_checkpoint(config.output)
        command = test_command(config, checkpoint)
        write_json(config.output / "test-command.json", command)
        with (config.output / "test.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report_path = config.output / "test/eval_info.json"
        report = json.loads(report_path.read_text())
        tracker = get_tracker()
        if tracker:
            tracker.log(
                {"test": report["overall"]},
                update=config.steps,
                reports=[report_path, record_path],
                videos={
                    f"test/{path.relative_to(config.output / 'test').with_suffix('').as_posix()}": path
                    for path in (config.output / "test").rglob("*.mp4")
                },
            )
        record["test"] = {
            "checkpoint": str(checkpoint),
            "seed": config.test_seed,
            "metrics": report,
        }
        failed = False
    finally:
        record["status"] = "failed_or_interrupted" if failed else "complete"
        write_json(record_path, record)
        tracker = get_tracker()
        if tracker:
            tracker.finish(failed)


if __name__ == "__main__":
    main()
