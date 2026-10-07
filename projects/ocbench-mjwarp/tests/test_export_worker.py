import subprocess
import sys
import time

import pytest
from ocbench_mjwarp.export_worker import wait_for_progress


def test_watchdog_kills_and_reaps_stuck_worker(tmp_path):
    pid = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        f"import os,time;open({str(pid)!r},'w').write(str(os.getpid()));time.sleep(60)",
    ]
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        wait_for_progress(command, tmp_path / "progress", 0.5)
    assert time.monotonic() - started < 5
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_watchdog_progress_and_errors(tmp_path):
    progress = tmp_path / "progress"
    command = [
        sys.executable,
        "-c",
        f"import pathlib,time;p=pathlib.Path({str(progress)!r});\nfor i in range(8): p.write_text(str(i));time.sleep(.1)",
    ]
    wait_for_progress(command, progress, 0.5)
    with pytest.raises(subprocess.CalledProcessError):
        wait_for_progress([sys.executable, "-c", "raise SystemExit(3)"], progress, 1)


def test_supervisor_retries_without_removing_checkpoint(tmp_path, monkeypatch):
    import json
    from dataclasses import dataclass
    from pathlib import Path

    from ocbench_mjwarp import export_worker

    @dataclass
    class Config:
        output: Path = tmp_path / "dataset"
        worker_timeout_seconds: float = 1
        worker_retries: int = 1

    config = Config()
    config.output.mkdir()
    checkpoint = config.output / "checkpoint.json"
    checkpoint.write_text('{"episodes": [1, 2]}')
    attempts = []

    def worker(command, progress, timeout):
        attempts.append(command)
        if len(attempts) == 1:
            raise TimeoutError("injected driver hang")
        (tmp_path / "dataset.worker-result.json").write_text('{"episodes": 3}')

    monkeypatch.setattr(export_worker, "wait_for_progress", worker)
    assert export_worker.supervise(config) == {"episodes": 3}
    assert len(attempts) == 2
    assert json.loads(checkpoint.read_text()) == {"episodes": [1, 2]}
