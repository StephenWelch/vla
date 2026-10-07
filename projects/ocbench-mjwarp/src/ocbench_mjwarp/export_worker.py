"""Process boundary for CUDA/NVENC calls that cannot be cancelled by Python."""

import json
import subprocess
import sys
import time
from dataclasses import asdict

from vla_tools.tracking import write_json


def wait_for_progress(command, progress, timeout):
    """Kill and reap a stuck worker; a thread timeout cannot stop driver calls."""
    previous = None
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            signature = progress.stat().st_mtime_ns if progress.exists() else None
            if signature != previous:
                previous = signature
                deadline = time.monotonic() + timeout
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Export worker made no progress for {timeout:g} seconds"
                )
            time.sleep(min(1.0, timeout / 10))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            process.kill()  # SIGKILL bypasses blocked Python executor/driver cleanup.
        process.wait(timeout=10)


def supervise(config):
    if config.worker_timeout_seconds <= 0 or config.worker_retries < 0:
        raise ValueError("Positive worker timeout and nonnegative retries required")
    spec = config.output.with_name(config.output.name + ".worker.json")
    result = config.output.with_name(config.output.name + ".worker-result.json")
    write_json(spec, asdict(config))
    command = [
        sys.executable,
        "-m",
        "ocbench_mjwarp.export_worker",
        "--config",
        str(spec),
    ]
    for attempt in range(config.worker_retries + 1):
        try:
            wait_for_progress(
                command,
                config.output / "worker-state.json",
                config.worker_timeout_seconds,
            )
            return json.loads(result.read_text())
        except TimeoutError as error:
            write_json(
                config.output.with_name(config.output.name + ".worker-status.json"),
                {"status": "timed_out", "attempt": attempt + 1, "error": str(error)},
            )
            if attempt == config.worker_retries:
                raise
            checkpoint = config.output / "checkpoint.json"
            if not checkpoint.exists() and config.output.exists():
                marker = config.output / "INCOMPLETE.json"
                if (
                    not marker.exists()
                    or json.loads(marker.read_text())["episodes"] != 0
                ):
                    raise RuntimeError(
                        "No durable checkpoint; refusing an unsafe restart"
                    ) from error
                # Preserve an empty/unfinished first batch, then retry a clean writer.
                config.output.rename(
                    config.output.with_name(
                        config.output.name + f".timeout-{time.time_ns()}"
                    )
                )
            print(
                f"Restarting export worker from its durable checkpoint ({attempt + 1}/{config.worker_retries})",
                flush=True,
            )


if __name__ == "__main__":
    from vla_tools.config import parse_args

    from .dataset import ExportConfig
    from .materialize import materialize

    config = parse_args(ExportConfig)
    result = materialize(config)
    write_json(
        config.output.with_name(config.output.name + ".worker-result.json"), result
    )
