"""Optimizer logging supports trainers without a GPU-memory meter."""

import json
from types import SimpleNamespace

import pytest
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from vla_tools.hooks import install_hooks


@pytest.mark.parametrize("gpu_memory", [None, 0.0, 1.5])
def test_optimizer_logging_with_optional_gpu_meter(tmp_path, gpu_memory):
    values = {"loss": 0.25, "grad_norm": 0.5, "update_s": 0.1}
    if gpu_memory is not None:
        values["gpu_mem_gb"] = gpu_memory
    meters = {name: AverageMeter(name, ":.3f") for name in values}
    for name, value in values.items():
        meters[name].update(value)
    metrics = MetricsTracker(8, 100, 10, meters, initial_step=24)
    details = {"l1_loss": 0.2}
    trainer = SimpleNamespace(
        make_train_eval_datasets=lambda cfg: None,
        save_checkpoint=lambda **kwargs: None,
        update_policy=lambda: (metrics, details),
    )
    install_hooks(trainer, {"output": str(tmp_path), "split": {}})

    result = trainer.update_policy()

    assert result[0] is metrics and result[1] is details
    record = json.loads((tmp_path / "optimizer_metrics.jsonl").read_text())
    expected = {
        "step": 25,
        "train/optimizer_loss": 0.25,
        "train/grad_norm": 0.5,
        "train/update_seconds": 0.1,
        "train/l1_loss": 0.2,
    }
    assert record == expected
