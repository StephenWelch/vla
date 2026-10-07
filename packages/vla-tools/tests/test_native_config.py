"""Structured native configuration preserves modern checkpoint contracts."""

import json
from types import SimpleNamespace

import pytest
from vla_tools.hooks import install_hooks
from vla_tools.train import native_config, normalize_record


def test_native_act_config_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["test"])
    values = {
        "policy": {
            "type": "act",
            "device": "cpu",
            "chunk_size": 25,
            "n_action_steps": 25,
            "push_to_hub": False,
        },
        "dataset": {"repo_id": "local/test", "root": str(tmp_path / "dataset")},
        "output_dir": str(tmp_path / "run"),
        "steps": 100000,
        "batch_size": 8,
        "wandb": {"enable": False},
    }
    cfg = native_config({"native": values})
    cfg.validate()
    assert cfg.policy.chunk_size == cfg.policy.n_action_steps == 25
    cfg.save_pretrained(tmp_path / "checkpoint" / "pretrained_model")
    path = tmp_path / "checkpoint" / "pretrained_model" / "train_config.json"
    resumed = native_config(
        {
            "native": {
                **values,
                "policy": {"device": "cpu"},
                "resume": True,
                "config_path": str(path),
            }
        }
    )
    resumed.validate()
    assert resumed.policy.chunk_size == 25
    assert resumed.checkpoint_path == path.parent.parent
    assert resumed.policy.pretrained_path == path.parent


def test_saved_record_conversion():
    result = normalize_record(
        {
            "backend": "ocbench",
            "image_normalization": "auto",
            "absolute_gripper": True,
            "absolute_arm": True,
            "overrides": ["policy.chunk_size=25"],
        }
    )
    assert result == {
        "image_normalization": "auto",
        "action_mode": "absolute",
        "overrides": {"policy.chunk_size": "25"},
    }
    with pytest.raises(ValueError, match="Absolute arm"):
        normalize_record({"absolute_arm": True})


def test_checkpoint_without_split_keeps_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    pretrained = checkpoint / "pretrained_model"
    calls = []

    def save(**kwargs):
        pretrained.mkdir(parents=True)
        calls.append(kwargs)

    trainer = SimpleNamespace(
        save_checkpoint=save,
        make_train_eval_datasets=lambda cfg: ("train", None),
        update_policy=lambda: None,
    )
    settings = {
        "output": str(tmp_path),
        "split": None,
        "rendering": {"revision": 2},
        "action_profile": {"mode": "absolute"},
        "amp_dtype": "bfloat16",
        "use_amp": False,
    }
    install_hooks(trainer, settings)
    assert trainer.make_train_eval_datasets(None) == ("train", None)
    trainer.save_checkpoint(checkpoint_dir=checkpoint)
    assert len(calls) == 1
    assert (
        json.loads((pretrained / "rendering.json").read_text()) == settings["rendering"]
    )
    assert (
        json.loads((pretrained / "action_profile.json").read_text())
        == settings["action_profile"]
    )
    assert json.loads((pretrained / "precision.json").read_text()) == {
        "use_amp": False,
        "amp_dtype": "bfloat16",
    }
