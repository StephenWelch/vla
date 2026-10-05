"""Logging must preserve run identities and never upload training data or weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_tracking import (
    Tracker,
    WandbConfig,
    evaluation_log,
    install_native_logging,
)


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    run = Mock()
    run.step = 0
    run.entity = "test-workspace"
    run.dir = str(tmp_path / "wandb/offline-run-test/files")
    run.summary = {}
    module = SimpleNamespace(
        init=Mock(return_value=run),
        util=SimpleNamespace(generate_id=lambda: "testid"),
        Settings=Mock(),
        Artifact=Mock(),
        Table=Mock(),
        Video=Mock(),
        errors=SimpleNamespace(CommError=ConnectionError, UsageError=PermissionError),
    )
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setattr("experiment_tracking.credentials", lambda: None)
    return module, run


def test_offline_fallback_resume_axes_and_failure(sdk, tmp_path):
    module, run = sdk
    cfg = WandbConfig(enable=True, group="experiment")
    tracker = Tracker(tmp_path, cfg, "train")
    assert module.init.call_args.kwargs["mode"] == "offline"
    assert tracker.state["fallback_reason"]
    tracker.log({"train": {"loss": 2}}, update=25)
    tracker.log({"val/probe": {"loss": 1}}, update=25)
    assert [call.kwargs["step"] for call in run.log.call_args_list] == [0, 1]
    assert run.log.call_args.args[0]["train/update"] == 25
    tracker.finish(failed=True)
    assert (
        json.loads((tmp_path / "tracking.json").read_text())["status"]
        == "failed_or_interrupted"
    )
    resumed = Tracker(tmp_path, cfg, "train", resume=True)
    resumed.log({"train": {"loss": 1}}, update=26)
    assert resumed.state["run_id"] == tracker.state["run_id"]
    assert run.log.call_args.kwargs["step"] == 2
    with pytest.raises(ValueError, match="project"):
        Tracker(
            tmp_path,
            WandbConfig(enable=True, project="different", group="experiment"),
            "train",
            resume=True,
        )


def test_network_failure_falls_back_without_reinitializing_later(
    sdk, tmp_path, monkeypatch
):
    module, run = sdk
    monkeypatch.setattr("experiment_tracking.credentials", lambda: "test-credential")
    module.init.side_effect = [ConnectionError("network down"), run]
    tracker = Tracker(tmp_path, WandbConfig(enable=True), "collect")
    tracker.log({"collection": {"episodes_committed": 1}})
    assert module.init.call_count == 2
    assert tracker.state["mode"] == "offline"
    assert "test-credential" not in (tmp_path / "tracking.json").read_text()


def test_disabled_tracking_does_not_import_sdk_or_create_output(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = tmp_path / "unused"
    tracker = Tracker(output, WandbConfig(), "train")
    tracker.log({"loss": 0})
    tracker.finish()
    assert not output.exists()


def test_nested_yaml_cli_precedence(tmp_path):
    from policy_utils import parse_args

    path = tmp_path / "logging.yaml"
    path.write_text(
        "enable: true\nproject: research\nmode: online\ntags: [act, cube]\n"
    )
    cfg = parse_args(WandbConfig, ["--config", str(path), "--mode", "offline"])
    assert cfg.enable and cfg.project == "research" and cfg.mode == "offline"
    from dataclasses import dataclass, field

    @dataclass
    class Config:
        wandb: WandbConfig = field(default_factory=WandbConfig)

    path.write_text("wandb:\n  enable: true\n  project: research\n  mode: online\n")
    nested = parse_args(
        Config, ["--config", str(path), "--wandb.no-enable", "--wandb.mode", "offline"]
    )
    assert not nested.wandb.enable and nested.wandb.mode == "offline"


def test_report_and_video_upload_allowlist(sdk, tmp_path):
    module, run = sdk
    for name in ("report.json", "model.safetensors", "episode.npz", "video.mp4"):
        (tmp_path / name).write_bytes(b"test")
    tracker = Tracker(tmp_path, WandbConfig(enable=True, mode="offline"), "eval")
    report = {
        "train/probe": {"loss": 1, "loss_definition": "inference_mode_l1"},
        "val/rollout": {"pc_success": 50, "episodes": [{"seed": 4, "success": True}]},
    }
    evaluation_log(tracker, report, tmp_path, 1000)
    payload = run.log.call_args.args[0]
    assert payload["train/probe/loss"] == 1 and payload["val/rollout/pc_success"] == 50
    assert "val/rollout/episodes" in payload
    tracker.log({}, reports=[tmp_path / "report.json"])
    assert [
        call.args[0] for call in module.Artifact.return_value.add_file.call_args_list
    ] == [str(tmp_path / "report.json")]
    module.Video.assert_called_once()


def test_native_adapter_uses_one_run_and_disables_model_upload(sdk, tmp_path):
    module, run = sdk

    class NativeLogger:
        def log_policy(self, path):
            assert self.cfg.disable_artifact

    trainer = SimpleNamespace(WandBLogger=NativeLogger)
    record = {
        "config": {},
        "dataset": {},
        "metadata_sha256": {},
        "camera_map": {},
        "versions": {},
    }
    getter = install_native_logging(
        trainer, {"wandb": {"enable": True, "mode": "offline"}}, record
    )
    cfg = SimpleNamespace(output_dir=tmp_path, wandb=SimpleNamespace(), resume=False)
    logger = trainer.WandBLogger(cfg)
    logger.log_dict({"loss": 2, "lr": 0.001}, 25)
    getter().log({"val/probe": {"loss": 1}}, update=25)
    logger.log_policy(tmp_path)
    assert module.init.call_count == 1 and cfg.wandb.run_id == "testid"
    assert run.log.call_args_list[0].args[0]["train/loss"] == 2


def test_real_offline_sdk_journal_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = WandbConfig(enable=True, mode="offline", group="offline-validation")
    tracker = Tracker(tmp_path, cfg, "train", {"validation": True})
    tracker.log({"train": {"loss": 2}}, update=1)
    tracker.log({"val/probe": {"loss": 1}}, update=1)
    identity = tracker.state["run_id"]
    tracker.finish()
    resumed = Tracker(tmp_path, cfg, "train", resume=True)
    resumed.log({"train": {"loss": 1}}, update=2)
    resumed.finish()
    state = json.loads((tmp_path / "tracking.json").read_text())
    assert state["run_id"] == identity and state["last_event"] == 2
    assert len(state["segments"]) == 2
    assert list(tmp_path.rglob("*.wandb"))


def test_pipeline_rejects_pid_reuse_and_allows_dead_owner(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run-ogbench-training.py"
    spec = importlib.util.spec_from_file_location("pipeline_tracking_test", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"python\0unrelated.py\0")
    assert not module.pipeline_alive({"pid": 1})
    monkeypatch.setattr(
        Path, "read_bytes", lambda _: b"python\0" + str(script).encode() + b"\0"
    )
    assert module.pipeline_alive({"pid": 1})


def test_corrupt_checkpoint_is_preserved_and_skipped(tmp_path):
    import numpy as np
    from policy_utils import checkpoint_step, latest_checkpoint
    from safetensors.numpy import save_file

    for step in (1, 2):
        checkpoint = tmp_path / "checkpoints" / f"{step:06d}"
        model, state = checkpoint / "pretrained_model", checkpoint / "training_state"
        model.mkdir(parents=True)
        state.mkdir()
        for path in (
            model / "config.json",
            model / "train_config.json",
            state / "optimizer_param_groups.json",
        ):
            path.write_text("{}")
        (state / "training_step.json").write_text(json.dumps({"step": step}))
        for path in (
            model / "model.safetensors",
            state / "optimizer_state.safetensors",
            state / "rng_state.safetensors",
        ):
            save_file({"value": np.ones(1, dtype=np.float32)}, str(path))
    broken = tmp_path / "checkpoints/000002/training_state/training_step.json"
    broken.write_bytes(b"\0" * 65)
    chosen = latest_checkpoint(tmp_path)
    assert checkpoint_step(chosen) == 1
    assert broken.read_bytes() == b"\0" * 65


def test_episode_callback_acknowledges_only_committed_archive(tmp_path):
    pytest.importorskip("ogbench_mjwarp")
    import numpy as np
    from ogbench_mjwarp.recording import ArchiveWriter, EpisodeBuffer

    acknowledgments = []

    def on_commit(row):
        assert (tmp_path / row["archive"]).is_file()
        assert (tmp_path / "episode-000000.json").is_file()
        acknowledgments.append(row)

    writer = ArchiveWriter(on_commit=on_commit)
    buffer = EpisodeBuffer(tmp_path, 0, {})
    buffer.add(None, np.zeros(18), np.zeros(5), False, True, True, 0)
    writer.submit(buffer, "failure", "timeout", {})
    writer.close()
    assert len(acknowledgments) == 1


def test_background_pipeline_freezes_nested_settings(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run-ogbench-training.py"
    spec = importlib.util.spec_from_file_location("pipeline_settings_test", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    process = Mock(pid=123)
    launch = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    cfg = module.Config(
        runs=tmp_path / "runs",
        experiment="test-experiment",
        background=True,
        wandb=WandbConfig(enable=True, mode="offline"),
    )
    module.run(cfg)
    settings_path = tmp_path / "runs/test-experiment/pipeline-config.json"
    settings = json.loads(settings_path.read_text())
    assert (
        not settings["background"] and settings["wandb"]["group"] == "test-experiment"
    )
    assert settings["raw"] == str(tmp_path / "raw/test-experiment")
    assert str(settings_path) in launch.call_args.args[0]
    from policy_utils import parse_args

    parsed = parse_args(module.Config, ["--config", str(settings_path)])
    assert parsed.wandb.mode == "offline" and parsed.experiment == "test-experiment"
