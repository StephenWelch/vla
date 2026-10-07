"""Scene splits and synchronous checkpoint probes for pinned LeRobot training."""

import copy
import hashlib
import json
import math
import random
from itertools import zip_longest
from pathlib import Path


def scene_split(dataset, fraction=0.2, seed=1000, episodes=None, *, overfit=False):
    """Keep reset-seed and state-fingerprint aliases together, including variants."""
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = manifest["episodes"]
    selected = set(episodes if episodes is not None else range(len(rows)))
    ids = {row["episode_index"] for row in rows}
    if len(ids) != len(rows) or not selected <= ids:
        raise ValueError("Manifest must contain one row per selected episode")
    if overfit:
        if len(selected) != 1 or fraction != 0:
            raise ValueError(
                "Overfit requires exactly one episode and no validation fraction"
            )
        return {
            "schema_version": 1,
            "method": "single_episode_overfit",
            "seed": seed,
            "validation_fraction": 0,
            "manifest_sha256": hashlib.sha256(
                (dataset / "manifest.json").read_bytes()
            ).hexdigest(),
            "train": [row for row in rows if row["episode_index"] in selected],
            "val": [],
        }
    groups, aliases = {}, {}
    for row in rows:
        index = row["episode_index"]
        identity = (row["env_id"], row["task_id"])
        fingerprint = row.get("randomization", {}).get("initial_state_fingerprint")
        keys = [(*identity, "seed", row["seed"])]
        if fingerprint:
            keys.append((*identity, "state", fingerprint))
        found = {aliases[k] for k in keys if k in aliases}
        group = min(found) if found else index
        groups.setdefault(group, [])
        for other in found - {group}:
            groups[group].extend(groups.pop(other))
            aliases = {k: group if v == other else v for k, v in aliases.items()}
        groups[group].append(row)
        for key in keys:
            aliases[key] = group
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    by_task = {}
    prescribed = any("dataset_split" in row for row in rows)
    split = {name: [] for name in ("train", "val")}
    for group in groups.values():
        group = [row for row in group if row["episode_index"] in selected]
        if group:
            if prescribed:
                labels = {row.get("dataset_split") for row in group}
                if len(labels) != 1 or not labels <= {"train", "val"}:
                    raise ValueError(
                        "Recorded split must label every episode consistently within each reset group"
                    )
                split[labels.pop()].extend(group)
                continue
            by_task.setdefault((group[0]["env_id"], group[0]["task_id"]), []).append(
                group
            )
    rng = random.Random(seed)
    for identity, task_groups in sorted(by_task.items()):
        if len(task_groups) < 2:
            raise ValueError(
                f"Need >=2 independent scenes for {identity}; same-reset variants cannot validate"
            )
        rng.shuffle(task_groups)
        n_val = min(len(task_groups) - 1, math.ceil(len(task_groups) * fraction))
        for i, group in enumerate(task_groups):
            split["val" if i < n_val else "train"].extend(group)
    if not split["train"] or not split["val"]:
        raise ValueError("Empty train/validation split")
    return {
        "schema_version": 1,
        "method": "recorded_cross_dataset_split"
        if prescribed
        else "grouped_random_split",
        "seed": seed,
        "validation_fraction": fraction,
        "manifest_sha256": hashlib.sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        **{
            name: sorted(rows, key=lambda r: r["episode_index"])
            for name, rows in split.items()
        },
    }


def loss_probe(policy, preprocessor, dataset, frames, batch_size, seed):
    """Inference-mode, sample-weighted losses on a fixed episode-balanced probe."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    rng = np.random.default_rng(seed)
    episode_indices = []
    for episode in dataset.episodes:
        start = int(dataset.meta.episodes[episode]["dataset_from_index"])
        end = int(dataset.meta.episodes[episode]["dataset_to_index"])
        indices = [dataset.absolute_to_relative_idx[i] for i in range(start, end)]
        rng.shuffle(indices)
        episode_indices.append(indices)
    indices = [
        i for layer in zip_longest(*episode_indices) for i in layer if i is not None
    ][:frames]
    totals, weights, samples = {}, {}, 0
    from vla_tools.preprocessing import policy_autocast

    with torch.inference_mode(), policy_autocast(policy):
        for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size):
            count = len(batch["action"])
            for key in dataset.meta.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255
            loss, details = policy.forward(preprocessor(batch))
            weight = (
                int((~batch["action_is_pad"]).sum())
                if policy.config.type == "act"
                else count
            )
            for key, value in {"loss": loss.item(), **details}.items():
                if isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite probe loss: {key}")
                    totals[key] = totals.get(key, 0.0) + value * weight
                    weights[key] = weights.get(key, 0) + weight
            samples += count
    return {
        **{key: value / max(weights[key], 1) for key, value in totals.items()},
        "samples": samples,
        "loss_definition": "inference_mode_l1"
        if policy.config.type == "act"
        else "flow_matching",
    }


def install_hooks(
    trainer,
    settings,
    get_tracker=lambda: None,
    *,
    prepare_views=None,
    evaluate_rollout=None,
):
    """Bind dataset/checkpoint hooks and observe upstream optimizer metrics."""
    import importlib.metadata

    import numpy as np
    import torch
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.utils import unflatten_dict
    from pyarrow import parquet

    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("Research training hooks require LeRobot 0.6.1")
    output = Path(settings["output"])
    split = settings["split"]
    datasets = {}
    parts = [name for name in ("train", "val") if split and split[name]]
    original_dataset, original_save = (
        trainer.make_train_eval_datasets,
        trainer.save_checkpoint,
    )
    original_update = trainer.update_policy

    def update(*args, **kwargs):
        expected = {"bfloat16": "bf16", "float16": "fp16"}[
            settings.get("amp_dtype", "bfloat16")
        ]
        if settings.get("use_amp"):
            accelerator = kwargs.get("accelerator") or args[5]
            if accelerator.mixed_precision != expected:
                raise RuntimeError(
                    f"Expected {expected} AMP, got {accelerator.mixed_precision}"
                )
        metrics, details = original_update(*args, **kwargs)
        if settings.get("use_amp"):
            if not math.isfinite(metrics.loss.val) or not math.isfinite(
                metrics.grad_norm.val
            ):
                raise FloatingPointError("Nonfinite AMP loss or gradient norm")
            details = {**(details or {}), "amp_bf16": int(expected == "bf16")}
        step = metrics.steps + 1
        if step % 25 == 0:
            with (output / "optimizer_metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "train/optimizer_loss": metrics.loss.val,
                            "train/grad_norm": metrics.grad_norm.val,
                            "train/update_seconds": metrics.update_s.val,
                            **{
                                f"train/{k}": v
                                for k, v in (details or {}).items()
                                if isinstance(v, (int, float))
                            },
                        }
                    )
                    + "\n"
                )
        return metrics, details

    def make_datasets(cfg):
        if not split:
            return original_dataset(cfg)
        full, _ = original_dataset(cfg)
        train_ids = {row["episode_index"] for row in split["train"]}
        episode_stats = {}
        for path in sorted(Path(cfg.dataset.root).glob("meta/episodes/**/*.parquet")):
            columns = [
                key
                for key in parquet.read_schema(path).names
                if key == "episode_index" or key.startswith("stats/")
            ]
            for row in parquet.read_table(path, columns=columns).to_pylist():
                if row["episode_index"] in train_ids:
                    episode_stats[row["episode_index"]] = unflatten_dict(
                        {
                            key.removeprefix("stats/"): np.asarray(value)
                            for key, value in row.items()
                            if key.startswith("stats/")
                        }
                    )
        if set(episode_stats) != train_ids or any(
            not value for value in episode_stats.values()
        ):
            raise ValueError(
                "Training episodes require per-episode normalization statistics"
            )
        stats = aggregate_stats([episode_stats[index] for index in sorted(train_ids)])
        if cfg.dataset.use_imagenet_stats:
            # The native factory already applied the backbone's fixed RGB stats.
            # Keep train-only state/action stats when replacing the split loaders.
            for key in full.meta.camera_keys:
                for field in ("mean", "std"):
                    stats[key][field] = np.asarray(full.meta.stats[key][field]).copy()
        for name in parts:
            datasets[name] = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=[r["episode_index"] for r in split[name]],
                delta_timestamps=resolve_delta_timestamps(cfg.policy, full.meta),
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                image_transforms=full.image_transforms if name == "train" else None,
            )
            datasets[name].meta.stats = stats
        if prepare_views:
            prepare_views(datasets, stats, settings)
        output.mkdir(parents=True, exist_ok=True)
        (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        (output / "train_stats.json").write_text(
            json.dumps(stats, default=lambda x: x.tolist(), indent=2, sort_keys=True)
            + "\n"
        )
        return datasets["train"], None

    def save(**kwargs):
        original_save(**kwargs)
        tracker = get_tracker()
        from vla_tools.tracking import write_json

        metadata = {
            "precision": {
                "use_amp": settings.get("use_amp", False),
                "amp_dtype": settings.get("amp_dtype", "bfloat16"),
            },
            "rendering": settings.get("rendering"),
            "action_profile": settings.get("action_profile"),
            "tracking": tracker.state if tracker and tracker.run else None,
        }
        for name, value in metadata.items():
            if value is not None:
                write_json(
                    kwargs["checkpoint_dir"] / "pretrained_model" / f"{name}.json",
                    value,
                )
        if tracker and tracker.run:
            tracker.run.summary["latest_checkpoint"] = str(kwargs["checkpoint_dir"])
        if not split:
            return
        step, policy = kwargs["step"], kwargs["policy"]
        report_path = output / "metrics" / f"{step:06d}.json"
        if (
            report_path.exists()
            and json.loads(report_path.read_text())["status"] == "complete"
        ):
            return
        report = {
            "step": step,
            "status": "running",
            "checkpoint": str(kwargs["checkpoint_dir"]),
            "policy_type": policy.config.type,
            "scope": "single_episode_overfit"
            if settings.get("overfit")
            else "train_validation",
            "probe_seed": settings["seed"],
            "dataset_manifest_sha256": split.get("manifest_sha256"),
            "train_stats_sha256": hashlib.sha256(
                (output / "train_stats.json").read_bytes()
            ).hexdigest(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        mode = policy.training
        queues = {
            key: copy.deepcopy(getattr(policy, key))
            for key in ("_queues", "_action_queue", "temporal_ensembler")
            if hasattr(policy, key)
        }
        try:
            policy.eval()
            for name in parts:
                random.seed(settings["seed"])
                np.random.seed(settings["seed"])
                torch.manual_seed(settings["seed"])
                # Probe without train-time augmentation.
                transforms = datasets[name].image_transforms
                datasets[name].image_transforms = None
                try:
                    report[f"{name}/probe"] = loss_probe(
                        policy,
                        kwargs["preprocessor"],
                        datasets[name],
                        settings["probe_frames"],
                        settings["batch_size"],
                        settings["seed"],
                    )
                finally:
                    datasets[name].image_transforms = transforms
            if step % settings["rollout_eval_freq"] == 0 or step == settings["steps"]:
                for name in parts:
                    task = split[name][0]
                    seeds = sorted({r["seed"] for r in split[name]})[
                        : settings["eval_episodes"]
                    ]
                    if settings.get(f"{name}_eval_seeds") is not None:
                        seeds = settings[f"{name}_eval_seeds"]
                    torch.manual_seed(settings["seed"])
                    random.seed(settings["seed"])
                    np.random.seed(settings["seed"])
                    if evaluate_rollout is None:
                        raise ValueError(
                            "Periodic rollout evaluation requires an evaluation function"
                        )
                    metrics = evaluate_rollout(
                        settings,
                        datasets["train"],
                        task,
                        seeds,
                        name,
                        kwargs["checkpoint_dir"],
                        step,
                        policy,
                        kwargs["preprocessor"],
                        kwargs["postprocessor"],
                    )
                    records = metrics["per_episode"]
                    report[f"{name}/rollout"] = {
                        "episodes": records,
                        **{
                            f"pc_{key}": 100
                            * sum(r[key] for r in records)
                            / len(records)
                            for key in (
                                "success",
                                "task_success",
                                "contact_valid",
                                "physics_valid",
                                "truncated",
                            )
                        },
                    }
                    if records and all("stable_stack" in r for r in records):
                        report[f"{name}/rollout"]["pc_stable_stack"] = (
                            100 * sum(r["stable_stack"] for r in records) / len(records)
                        )
            report["status"] = "complete"
            if tracker and tracker.run:
                from vla_tools.tracking import evaluation_log

                evaluation_log(tracker, report, output / "eval" / f"{step:06d}", step)
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            policy.reset()
            for key, value in queues.items():
                setattr(policy, key, value)
            policy.train(mode)
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            torch.cuda.set_rng_state_all(state[3])
            report_path.write_text(
                json.dumps(report, default=lambda x: x.tolist(), indent=2) + "\n"
            )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, default=lambda x: x.tolist()) + "\n")
        print(
            f"Periodic evaluation step {step}: "
            + ", ".join(
                f"{name} loss={report[f'{name}/probe']['loss']:.5f}" for name in parts
            ),
            flush=True,
        )
        retain_checkpoints(output, kwargs["checkpoint_dir"], report)
        if tracker and tracker.run:
            from vla_tools.tracking import write_json

            tracker.log(
                {},
                update=step,
                reports=[
                    report_path,
                    output / "best.json",
                    output / "split.json",
                    output / "train_stats.json",
                    kwargs["checkpoint_dir"] / "pretrained_model/rendering.json",
                ],
            )
            if (output / "best.json").exists():
                tracker.run.summary["best"] = json.loads(
                    (output / "best.json").read_text()
                )
            tracker.run.summary["latest_checkpoint"] = str(kwargs["checkpoint_dir"])
            write_json(
                kwargs["checkpoint_dir"] / "pretrained_model/tracking.json",
                tracker.state,
            )

    trainer.make_train_eval_datasets, trainer.save_checkpoint, trainer.update_policy = (
        make_datasets,
        save,
        update,
    )


def retain_checkpoints(output, latest, report):
    """Keep resumable latest and the best validation checkpoint, recording ties."""
    import shutil

    best_path = output / "best.json"
    if "val/rollout" in report:
        score = [report["val/rollout"]["pc_success"], -report["val/probe"]["loss"]]
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if best is None or score > best["score"]:
            best_path.write_text(
                json.dumps(
                    {"checkpoint": str(latest), "step": report["step"], "score": score},
                    indent=2,
                )
                + "\n"
            )
    best = (
        json.loads(best_path.read_text())["checkpoint"] if best_path.exists() else None
    )
    for checkpoint in (output / "checkpoints").iterdir():
        if (
            checkpoint.is_dir()
            and not checkpoint.is_symlink()
            and checkpoint.name.isdecimal()
            and str(checkpoint) not in (str(latest), best)
        ):
            shutil.rmtree(checkpoint)


def run_worker(prepare_views=None, evaluate_rollout=None):
    import os

    from lerobot.scripts import lerobot_train

    from vla_tools.preprocessing import configure_training
    from vla_tools.tracking import install_native_logging
    from vla_tools.train import native_config, normalize_record

    os.environ.setdefault("MUJOCO_GL", "egl")
    record = json.loads(Path(os.environ["VLA_TRAIN_SETTINGS"]).read_text())
    settings = {
        **normalize_record(record["config"]),
        "split": record["split"],
        "rendering": record.get("rendering"),
        "action_profile": record.get("action_profile"),
    }
    configure_training(lerobot_train, settings)
    get_tracker = install_native_logging(lerobot_train, settings, record)
    install_hooks(
        lerobot_train,
        settings,
        get_tracker,
        prepare_views=prepare_views,
        evaluate_rollout=evaluate_rollout,
    )
    failed = True
    try:
        lerobot_train.train(native_config(record))
        failed = False
    finally:
        tracker = get_tracker()
        if tracker:
            tracker.finish(failed)


if __name__ == "__main__":
    run_worker()
