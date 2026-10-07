"""Serial export ablations on a fixed, length-stratified Hub subset."""

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from vla_tools.config import parse_args
from vla_tools.tracking import write_json


@dataclass
class BenchmarkConfig:
    source: Path
    output: Path
    episodes_per_split: int = 10
    repeats: int = 3
    wait_for_queue: Path | None = None
    verify_gpu: bool = True


def select_subset(rows, count):
    selected = []
    if count < 2:
        raise ValueError("At least two episodes per split required")
    for split in ("train", "val"):
        candidates = sorted(
            (r for r in rows if r.get("dataset_split") == split),
            key=lambda r: (r["length"], r["episode_id"]),
        )
        if len(candidates) < count:
            raise ValueError(f"Need {count} {split} episodes")
        selected.extend(
            candidates[round(i * (len(candidates) - 1) / (count - 1))]
            for i in range(count)
        )
    return selected


def run(config):
    import psutil

    from .benchmark_eval import wait_for_training

    if config.repeats < 1:
        raise ValueError("Positive repeat count required")
    config.output.mkdir(parents=True, exist_ok=True)
    path = config.output / "benchmark.json"
    spec = config.output / "config.json"
    requested = json.loads(json.dumps(asdict(config), default=str))
    if spec.exists() and json.loads(spec.read_text()) != requested:
        raise FileExistsError("Benchmark configuration differs; use a fresh output")
    write_json(spec, requested)
    report = json.loads(path.read_text()) if path.exists() else {"cases": []}

    def status(value):
        report["status"] = value
        write_json(path, report)

    def idle():
        wait_for_training(None)
        return not any(
            "ocbench_mjwarp.benchmark_eval" in (p.info["cmdline"] or [])
            for p in psutil.process_iter(["cmdline"])
        )

    try:
        status("waiting_for_training")
        wait_for_training(config.wait_for_queue)
        status("waiting_for_eval")
        while not idle():
            time.sleep(30)
        from .dataset import ExportConfig
        from .episodes import records, select_episodes

        status("checking")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        repo = Path(__file__).resolve().parents[4]
        if config.verify_gpu:
            with (config.output / "gpu-tests.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-m",
                        "gpu",
                        "projects/ocbench-mjwarp/tests/test_gpu_export.py",
                        "projects/ocbench-mjwarp/tests/test_materialize.py",
                        "projects/ocbench-mjwarp/tests/test_hub.py",
                        "projects/ocbench-mjwarp/tests/test_video_codec.py",
                        "projects/ocbench-mjwarp/tests/test_eval_pipeline.py",
                    ],
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
        subset = select_subset(
            select_episodes(config.source), config.episodes_per_split
        )
        source = config.output / "source"
        (source / "raw").mkdir(parents=True, exist_ok=True)
        existing = records(source)
        if existing and existing != sorted(subset, key=lambda r: r["archive"]):
            raise FileExistsError("Benchmark subset changed")
        for row in subset:
            target = source / "raw" / row["archive"]
            if not target.exists():
                target.symlink_to((config.source / "raw" / row["archive"]).resolve())
            write_json(target.with_suffix(".json"), row)
        report["episode_ids"] = [r["episode_id"] for r in subset]
        report["frames"] = sum(r["length"] for r in subset)
        base = ExportConfig(
            source=source,
            output=config.output / "unused",
            successes=None,
            reuse_simulation=False,
            episode_order="source",
            overlap_commits=False,
        )
        reuse = replace(base, reuse_simulation=True)
        length = replace(reuse, episode_order="length")
        overlap = replace(length, overlap_commits=True)
        variants = {
            "baseline": base,
            "reuse": reuse,
            "length": length,
            "overlap": overlap,
            "320x240": replace(overlap, image_size=(240, 320)),
        }
        status("running")
        for repeat in range(config.repeats):
            for name, variant in variants.items():
                if any(
                    c["variant"] == name and c["repeat"] == repeat
                    for c in report["cases"]
                ):
                    continue
                if not idle():
                    raise RuntimeError("Evaluation started during export benchmark")
                if psutil.virtual_memory().available < 8 * 2**30:
                    raise RuntimeError("Need 8 GiB available host memory for benchmark")
                cfg = replace(variant, output=config.output / f"{name}-{repeat}")
                filename = config.output / f"{name}-{repeat}.json"
                write_json(filename, asdict(cfg))
                result_path = cfg.output.with_name(
                    cfg.output.name + ".worker-result.json"
                )
                started = time.perf_counter()
                peak_rss = 0
                # nvidia-smi includes Warp and NVENC allocations omitted by Torch's allocator.
                peak_gpu = 0
                with filename.with_suffix(".log").open("w") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "ocbench_mjwarp.cli",
                            "export",
                            "--config",
                            str(filename),
                        ],
                        cwd=repo,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        while child.poll() is None:
                            try:
                                process = psutil.Process(child.pid)
                                peak_rss = max(
                                    peak_rss,
                                    sum(
                                        p.memory_info().rss
                                        for p in [
                                            process,
                                            *process.children(recursive=True),
                                        ]
                                    ),
                                )
                            except psutil.NoSuchProcess:
                                pass
                            memory = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                                check=True,
                            )
                            peak_gpu = max(
                                peak_gpu, max(int(v) for v in memory.stdout.split())
                            )
                            time.sleep(1)
                        if child.returncode:
                            raise subprocess.CalledProcessError(
                                child.returncode, child.args
                            )
                    finally:
                        if child.poll() is None:
                            try:
                                descendants = psutil.Process(child.pid).children(
                                    recursive=True
                                )
                            except psutil.NoSuchProcess:
                                descendants = []
                            for descendant in descendants:
                                try:
                                    descendant.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            child.kill()
                            psutil.wait_procs(descendants, timeout=5)
                        child.wait()
                if cfg.output.with_name(
                    cfg.output.name + ".worker-status.json"
                ).exists():
                    raise RuntimeError(
                        "Export restarted during benchmark; use a fresh output for comparable timing"
                    )
                result = json.loads(result_path.read_text())
                if "seconds" not in result:
                    raise RuntimeError(
                        "Completed export has no fresh timing; use a fresh benchmark output"
                    )
                report["cases"].append(
                    {
                        "variant": name,
                        "repeat": repeat,
                        **result,
                        "paired_frames_per_second": result["frames"]
                        / result["seconds"],
                        "process_seconds": time.perf_counter() - started,
                        "peak_host_rss_bytes": peak_rss,
                        "peak_device_memory_mib": peak_gpu,
                    }
                )
                write_json(path, report)
        report["median_paired_frames_per_second"] = {
            name: statistics.median(
                c["paired_frames_per_second"]
                for c in report["cases"]
                if c["variant"] == name
            )
            for name in variants
        }
        report["timing_note"] = (
            "Stage durations overlap; wall time is authoritative. Device memory includes other resident GPU processes."
        )
        status("complete")
    except BaseException as error:
        report["error"] = str(error)
        status("failed_or_interrupted")
        raise


if __name__ == "__main__":
    run(parse_args(BenchmarkConfig))
