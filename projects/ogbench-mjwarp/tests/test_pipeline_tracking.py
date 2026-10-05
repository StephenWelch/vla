import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from vla_tools.tracking import WandbConfig


def test_pipeline_rejects_pid_reuse_and_allows_dead_owner(monkeypatch):
    from ogbench_mjwarp import pipeline as module

    script = Path(module.__file__)
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"python\0unrelated.py\0")
    assert not module.pipeline_alive({"pid": 1})
    monkeypatch.setattr(
        Path, "read_bytes", lambda _: b"python\0" + str(script).encode() + b"\0"
    )
    assert module.pipeline_alive({"pid": 1})
    monkeypatch.setattr(
        Path, "read_bytes", lambda _: b"python\0-m\0ogbench_mjwarp.pipeline\0"
    )
    assert module.pipeline_alive({"pid": 1})


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
    from ogbench_mjwarp import pipeline as module

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
    from vla_tools.config import parse_args

    parsed = parse_args(module.Config, ["--config", str(settings_path)])
    assert parsed.wandb.mode == "offline" and parsed.experiment == "test-experiment"
