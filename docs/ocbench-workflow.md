> Historical experiment log. Earlier benchmark commands, flat comparison configs, and OGBench references below describe the implementations used for those measurements; they are not current entrypoints. See the [OCBench README](../projects/ocbench-mjwarp/README.md) for the consolidated workflow. The experimental GPU runner and rendered-video re-exporter were removed; `benchmark_render` now measures the production exporter.

# OCBench collection and ACT workflow

The native `block-double-task2-v0` controller replaces the planner for the new stacking collection. Its 50 Hz actions, randomization, retries and native stopping conditions are preserved. Existing OGBench planners and checkpoints remain available separately. OCBench is pinned to `e2cd2f72110b66bd65afab1b855d81ebc73aeacc`; dependencies are locked in the new simulator project.

## OCBench installation

OCBench is a regular installed library, pinned by Git commit in `projects/ocbench-mjwarp/pyproject.toml` and `uv.lock`. The wheel includes the native controllers, MJWarp kernels and model assets. Run `bash projects/ocbench-mjwarp/scripts/setup-wsl.sh` for a fresh environment; a separate upstream checkout is not required.

The existing shared training environment was converted from an editable checkout to the same pinned package without changing its other dependencies:

```bash
uv pip install --python /mnt/e/vla-ogbench/venv/bin/python --no-deps \
  'ocbench @ git+https://github.com/seohongpark/ocbench.git@e2cd2f72110b66bd65afab1b855d81ebc73aeacc'
```

The upstream checkout is retained for reference. Runtime imports resolve to `site-packages`; our integration stays in `ocbench_mjwarp`. Change the dependency pin, lockfile and provenance `COMMIT` together when upgrading upstream, then recheck action, camera and replay compatibility.

Installation verification: all 148 package files matched the checkout (allowing CRLF line endings), environment reset and the exported rendering profile matched, and all four contract tests passed, including native/instrumented GPU physics parity. Existing training was left running.

## Data contract

Each attempt retains N actions and N+1 states, including the terminal state before upstream parking. Metadata includes reset/oracle seeds, train/validation assignment, native success/health, completion reason, contact peaks, numerical/capacity validity, stable-stack diagnostics, native timing distributions and sampled keyframe plans. Plans include world poses, gripper targets, tangents, mistake flags, RNG counters and retry counts. These are planned motion annotations, not measurements of actual grasp contact points.

Exported successes require native success, native workspace health, finite physics, no capacity overflow and robot/environment penetration within 1 mm for non-pad contacts and 3 mm overall. Valid native failures use the same checks. Solver/line-search iteration-budget flags are retained separately and do not alone invalidate an attempt. The contact audit covers simulator collision geometry at physics substeps; it does not certify visual mesh, self-collision or continuous-time collision freedom. Stable released stacks are a separate diagnostic and are not required: native episodes can terminate while the gripper still holds a block.

Actions are seven normalized joint/gripper deltas with native scales, not the existing OGBench absolute targets. Dataset/checkpoint camera and action profiles must match. Policy inputs contain 18 proprioceptive values and two RGB images; privileged object state stays in replay/annotations. Images are 640x480, with native OCBench front and wrist optics. Rendering replays stored states in GPU batches, updates ray-tracing bounds, and feeds bounded queues directly into concurrent LeRobot video encoders. The writer commits encoded episodes without decoding or re-encoding; preview videos are byte copies.

## Verification

The 32-attempt integration pilot (reset seeds 82000-82031) produced 26 native successes, 23 contact-audited successes, three valid failures and six contact-invalid attempts. This pilot preceded addition of explicit capacity-overflow recording; production records that evidence as well. See [collection W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/i989lbt9).

- All 32 archives passed N+1-state and non-parked terminal-state checks.
- Eleven usable attempts were rendered; front/wrist samples were visually inspected at initial, intermediate and final states.
- Four successes (7,536 frames) and one failure (2,500 frames) exported and loaded through LeRobot. Successes split into three train and one validation episode.
- A two-update ACT smoke run completed separate train/validation loss probes and one three-step rollout per split. Checkpoint reload and standalone evaluation also passed. [Training W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/cr9me5mo). This verifies integration, not learned task competence.
- Shared training/contact/config/tracking regressions passed 49 tests; OCBench contract and native/instrumented GPU parity passed four. Rendering/CLI tests passed 18 initially; two resume tests exposed tuple/list config serialization and passed after canonicalizing the saved-run comparison. Ruff and lockfile checks passed.

Pilot artifacts are under `C:\Users\steph\data\vla\ocbench-integration-pilot`. The compatible existing WSL environment `/mnt/e/vla-ogbench/venv` was used for these runs to avoid duplicating large GPU dependencies on the nearly full E: drive. The new setup script creates a separate environment for a fresh installation. PyAV is used for video writing and decoding; an optional TorchCodec shared-library warning does not prevent this path.

## Production recipe

```bash
MUJOCO_GL=egl vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml
```

The recipe uses 500 attempts, 32 worlds, reset seeds 83000-83499 and oracle seeds 93000-93499. Every fifth attempt is validation, assigned before outcome filtering. All raw attempts remain; only audited native successes/failures are rendered/exported. The pipeline estimates storage before rendering, checks a 4 GiB reserve between batches, and verifies exported datasets load.

ACT trains from scratch for 20,000 updates on audited native successes, with batch size eight and 40-step action chunks. Normalization statistics use training episodes only. Loss probes run every 1,000 updates; policy rollouts run every 2,000 on ten fixed train and ten fixed validation reset states, up to 2,500 steps each. Native success, audited success, contact/physics validity, truncation and stable-stack diagnostics are separate metrics. Evaluations use the policy without oracle recovery. Checkpoint retention keeps latest and best validation checkpoints.

The full pipeline was launched under a hidden Windows WSL host process: [collection W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/o3voj764). Its first 32 attempts produced 28 native successes, 23 audited successes, three valid failures and six invalid attempts. Output is `C:\Users\steph\data\vla\ocbench\stack-500`; `status.json`, `pipeline.log`, `pipeline.stderr.log` and `tracking.json` report progress. ACT starts automatically after rendering/export. A launch is not a completed collection or training result.

Generation resumes committed attempts with the original batch layout; completed exports are reused. Standalone `render` remains available for viewing. Export uses the production direct encoder; compatible partial exports resume from durable checkpoints. Interrupted training resumes through `vla-ocbench-train --config <run>/train.json --resume <checkpoint>/pretrained_model`; the pipeline does not silently restart or overwrite a training run.

## Direct rendering benchmark

A matched test of 16 episodes, sampling 64 states each with two 640x480 cameras and H.264 CRF 18 encoding, measured 124 paired-camera frames/s at four worlds, 103 at eight and 97 at sixteen. Initialization was excluded; other workloads were active, and this short test is not a full-export speedup measurement. Larger batches increased encoder backpressure, so the default remains four worlds, with one encoder thread per camera and eight queued frames. Override `render_batch_size`, `encoder_threads` and `encoder_queue_size` in the pipeline YAML. Reproduce with `python -m ocbench_mjwarp.benchmark_render --source <collection> --output <benchmark>`.

The direct encoder handoff is pinned to LeRobot 0.6.1. Tests check frame counts, timestamps, episode boundaries, image statistics and camera identity after dataset reload. Raw attempts and previously rendered videos are retained.

The complete-episode direct-export check wrote two episodes (3,501 frames) in 32.5 seconds. Reload checks decoded every frame in both cameras and verified raw action/state alignment, timestamps and finite image statistics. Five encoder/contract/native-parity tests passed. Production resumed with batch size four; `pipeline-direct.log` and `pipeline-direct.stderr.log` record this launch.

### NVENC sweep

Production was stopped with 72 successes committed; its partial export and all raw attempts are retained. The matched sweep used 32 episodes, 128 sampled frames each, two 640x480 views, GOP 2, and the same bounded queues/statistics path. NVENC used constant QP 18; CPU x264 used CRF 18. Those settings are not equal-quality targets.

| Worlds | NVENC p1 paired frames/s | NVENC p3 paired frames/s |
| --- | ---: | ---: |
| 1 | 98.4 | 91.4 |
| 2 | 109.8 | 106.2 |
| 4 | 97.4 | 94.5 |
| 8, 16, 32 | Encoder initialization failed | Encoder initialization failed |

The four-world CPU control achieved 181.2 paired frames/s. Encoded totals were 33.5 MB for CPU, 71.5 MB for p1 and 60.3 MB for p3. Timings include encoding/drain but exclude simulator initialization and warmup. Hardware acceleration did not improve this short-clip path. Larger NVENC batches failed inside `avcodec_open2`; each world requires two concurrent sessions. This establishes tested working/failing batch sizes, not the exact driver session limit.

LeRobot 0.6.1 rejects named NVENC presets during numeric option validation; the benchmark resolves names using PyAV's codec option enumeration. Reproduce with `python -m ocbench_mjwarp.benchmark_render --source <collection> --output <fresh-output> --codec h264_nvenc --preset p1 --episodes 32 --frames 128 --batch-sizes 1 2 4 8 16 32`. Reports retain failures rather than treating them as throughput results. Artifacts are in `C:\Users\steph\data\vla\ocbench\nvenc-sweep`.

A longer-clip control used four episodes resampled to 2,048 frames each (8,192 paired frames). CPU at four worlds achieved 139.3 paired frames/s; NVENC p1 at two worlds achieved 131.6. Including simulation setup, wall times were effectively tied: 65.49 versus 65.43 seconds. Files totaled 64.6 versus 130.4 MB. NVENC reduced queue waiting (7.77 to 4.12 seconds), but replay/render time increased (49.02 to 53.93 seconds). These measurements show no end-to-end advantage for the current host-frame LeRobot encoding path; they do not measure a GPU-resident NVENC pipeline. Keep CPU encoding pending a different optimization. Production remains paused after the benchmark.

### GPU-resident export pilot

`ocbench_mjwarp.gpu_export` is an opt-in experiment, not the production exporter. It uploads trajectories once, exposes borrowed packed GPU images from the shared renderer, and feeds CUDA tensors directly to PyNvVideoCodec 2.2.3. Only the same 120x160 RGB samples used by LeRobot image statistics cross to CPU; encoded H.264 packets are remuxed into MP4 and committed with the native LeRobot writer. Two worlds use four bounded encoder sessions. This pilot does not decouple larger rendering batches from encoder concurrency or overlap rendering with asynchronous GPU encoding.

Two complete production episodes (3,587 paired-camera frames, 640x480 at 50 Hz) produced:

| Path | Export seconds | Paired frames/s | Video MB |
| --- | ---: | ---: | ---: |
| Existing replay + x264 | 22.18 | 161.7 | 33.3 |
| GPU replay + x264 | 21.88 | 163.9 | 33.3 |
| GPU replay + direct NVENC | 29.08 | 123.4 | 61.3 |
| CUDA graph replay + x264 | 31.00 | 115.7 | 33.3 |
| CUDA graph replay + direct NVENC | 42.13 | 85.1 | 61.3 |
| Existing replay + x264 veryfast | 25.01 | 143.4 | 33.9 |

Times include rendering, statistics, encoding, remux, episode commit and dataset finalization; setup and post-export validation are excluded. These are single small runs, not confidence intervals. The 1.4% GPU replay difference is too small to establish a gain. NVENC QP 18 is not x264 CRF 18; the larger GPU-encoded videos also had higher sampled PSNR. None of these synchronous-pilot results justifies replacing the current exporter.

All paths passed pixel-exact replay checks at initial/middle/final states, complete video decoding and timestamp checks, dataset reload and sampled action/state alignment. Minimum sampled decoded-image PSNR was 39.2 dB or higher. Direct-GPU and baseline image statistics were identical for both cameras. Full reports/configurations/datasets are under `C:\Users\steph\data\vla\ocbench\gpu-pilot`. Production remains paused, retaining the earlier 72 committed successes and all raw attempts.

Install the optional `gpu-video` extra to reproduce (the pilot used `/mnt/e/vla-ogbench/venv` with `PyNvVideoCodec==2.2.3`):

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.gpu_export \
  --source <collection> --output <fresh-pilot-directory> --mode device --worlds 2
```

Use `--mode host` for the existing path, `--mode replay` for GPU replay with CPU encoding, `--graph` for CUDA capture, or `--mode host --cpu-preset veryfast` for the software preset control. `--max-frames 16` provides a short smoke test. PyNvVideoCodec 2.2.3 requires uppercase preset names and returns packet dictionaries, unlike the bytes-returning examples in parts of its online documentation.

Ten GPU replay/shared-renderer regression tests passed (three native rasterizer visibility tests were deselected); Ruff and lockfile checks passed.

### Asynchronous GPU export pilot

`--mode async --buffer-frames 2` overlaps GPU replay/rendering with encoding, statistics and compressed-packet writes. Each world owns one worker and two persistent NVENC sessions on an independent CUDA stream. A bounded ring copies the renderer's reusable image storage, transfers only statistics samples to pinned CPU buffers, and records a readiness event. A slot cannot be reused until every consuming worker has finished its input stream and statistics update. Per-world queues preserve frame order; slow workers apply backpressure and failures propagate to the producer. The main thread commits finished episodes through LeRobot after workers flush/remux their videos.

This prototype supports one to four worlds (two to eight encoder sessions). Rendering batch size and session count remain coupled; it does not yet multiplex chunks across a smaller encoder pool. Full-resolution frames stay on GPU. At two worlds, a two-frame ring uses 9.4 MiB of GPU snapshots plus 0.44 MiB of pinned host samples, excluding trajectories and internal renderer/encoder allocations.

On the same two complete episodes (3,587 paired frames):

| Path | Export seconds | Paired frames/s |
| --- | ---: | ---: |
| CPU baseline | 23.96 | 149.7 |
| Synchronous GPU | 30.59 | 117.2 |
| Async GPU, two buffered frames | 18.59 | 193.0 |
| Async GPU, eight buffered frames | 18.50 | 193.9 |
| CPU baseline repeated afterward | 23.16 | 154.9 |

Async reduced elapsed export time by approximately 20–23% relative to the CPU controls and 39% relative to synchronous GPU encoding in this pilot. Increasing buffer depth did not materially improve throughput. Both async outputs were byte-for-byte identical to the synchronous GPU MP4s, and all dataset statistics matched exactly. NVENC output remained larger than the CPU baseline at the tested settings (61.3 versus 33.3 MB); QP 18 and CRF 18 are not equal-quality targets. These are small pilot measurements, not a production throughput guarantee.

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.gpu_export \
  --source <collection> --output <fresh-pilot-directory> \
  --mode async --worlds 2 --buffer-frames 2
```

The same options can be supplied through `--config <yaml>`. Artifacts are under `C:\Users\steph\data\vla\ocbench\async-pilot`. Production remains paused; the partial production dataset is preserved.

A matched four-world comparison exported 5,151 paired frames in 36.63 seconds with the CPU baseline and 22.85 seconds with async GPU encoding (140.6 versus 225.4 paired frames/s). That is 1.60x throughput, or 37.6% less export time, on this small workload. All video frames/timestamps decoded correctly, sampled action/state/image checks passed, and aggregate statistics matched the CPU dataset exactly. Encoded size was 43.9 MB for CPU and 80.8 MB for NVENC. The two-frame ring occupied 18.75 MiB of GPU snapshots and 0.88 MiB of pinned samples at four worlds.

Six targeted tests passed: delayed-consumer buffer ownership at depths one and three, injected encoder failure/cleanup, GPU replay parity with/without CUDA graphs, and native LeRobot episode handoff. Ruff passed. `async-pilot/comparison.json` contains the matched reports and byte/statistics comparisons. Four-world async is a promising production candidate; this prototype is still opt-in, preserves the eight-session ceiling, and has not been run across the full collection.

### Async statistics investigation

Profiling the same four episodes (5,151 paired frames) found that native image statistics consumed 30.68 seconds of summed worker time, versus 12.81 seconds in NVENC calls. Workers overlap, so these are not additive wall-clock costs. CUDA event intervals include launch gaps and contention, not just kernel execution. The export took 21.84 seconds, including 12.93 seconds of producer backpressure.

The opt-in `--image-stats uint8` path replaces the adaptive 5,000-bin image histogram with exact 256-bin RGB counts on CPU. It samples the same pixels, accumulates moments in float64, and computes exact linear-interpolated quantiles at episode completion. Memory stays bounded independently of episode length. Rendering, GPU buffer ownership, resolution, encoding settings and the LeRobot writer are unchanged. Moving statistics to GPU is deferred: this simpler change already removes most of their cost.

| Statistics | Export seconds | Paired frames/s |
| --- | ---: | ---: |
| Native, profiled | 21.84 | 235.9 |
| uint8 histogram, profiled | 12.90 | 399.4 |
| uint8 histogram, unprofiled repeat | 13.09 | 393.5 |
| Native, unprofiled control repeated afterward | 21.85 | 235.8 |

Summed statistics worker time fell to 4.61 seconds and producer backpressure to 4.61 seconds. These matched pilots show approximately 1.7x throughput over native statistics; they do not establish full-collection throughput. The earlier CPU baseline took 36.63 seconds on these episodes. All eight MP4 files were byte-identical between the profiled paths (80.8 MB total). Full video decoding, exact timestamps, pixel replay parity and sampled action/state alignment passed. Normalized aggregate statistics differed by at most 0.00011: the new quantiles use exact discrete counts rather than LeRobot's adaptive approximation, and moment accumulation has different numerical precision. Unit tests compare moments and quantiles directly against NumPy, including constant channels and multiple updates.

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.gpu_export \
  --source <collection> --output <fresh-pilot-directory> \
  --mode async --worlds 4 --buffer-frames 2 --image-stats uint8
```

Add `--profile` for worker encode/write/statistics timing and render/copy event intervals. This option adds measurement overhead. Reports and datasets are under `C:\Users\steph\data\vla\ocbench\async-investigation`; `comparison.json` records video hashes and statistics differences. The native statistics backend remains the default and production remains paused. Remaining measured costs are encoding, compressed-packet writes and rendering; temporal batching and renderer/session decoupling remain untested follow-ups.

Nine targeted tests passed: NumPy statistics parity/input validation, delayed-consumer ownership and profiled events, injected failure cleanup, GPU replay parity with/without graphs, and native LeRobot episode handoff. Ruff lint/format checks passed.

### Temporal rendering and packet buffering

The async pilot now accepts `--render-batch-frames`: each render call places several consecutive timestamps into separate MJWarp worlds, ordered by timestamp and then episode. Encoder concurrency stays at two sessions per episode (eight for four episodes). Each worker consumes its temporal chunk in order before releasing the ring slot. Final-state padding is rendered but never encoded as extra frames. `--buffer-frames` counts ring slots; each slot now holds an entire temporal batch. CUDA graph replay remains restricted to one timestamp per render.

`--write-buffer-bytes` controls each camera's bounded compressed-packet buffer. Increasing it from 8 KiB to 1 MiB reduced summed worker write time from 4.39 to 0.34 seconds in the matched profile. Flush/remux time remains included in the export measurement. `--gop` separately controls NVENC keyframe spacing; B-frames remain disabled and QP stays at 18.

The same four full episodes (5,151 paired frames, two 640x480 views, uint8 statistics) produced:

| Path | Export seconds | Paired frames/s | Video MB |
| --- | ---: | ---: | ---: |
| Control, 8 KiB writes | 13.44 | 383.2 | 80.8 |
| 1 MiB writes | 12.45 | 413.6 | 80.8 |
| 1 MiB writes, two timestamps/render | 10.38 | 496.5 | 80.8 |
| 1 MiB writes, four timestamps/render | 9.98 | 516.1 | 80.8 |
| Four timestamps/render, unprofiled repeat | 9.95 | 517.6 | 80.8 |
| Eight timestamps/render, unprofiled | 10.46 | 492.5 | 80.8 |
| Control, unprofiled repeat afterward | 13.10 | 393.3 | 80.8 |
| GOP 8, four timestamps/render, unprofiled | 9.80 | 525.4 | 37.8 |
| GOP 50, one timestamp/render | 11.79 | 437.0 | 25.5 |

Four timestamps per render with 1 MiB writes improved throughput by 32% over the unprofiled control, reducing export time by 24%. Rendering event intervals fell from 6.72 to 3.73 seconds in the profiled runs; these include CPU launch gaps and GPU contention. Eight timestamps did not improve throughput. The four-timestamp, two-slot ring uses 75 MiB of GPU snapshots and 3.52 MiB of pinned samples, plus up to 8 MiB of compressed-packet buffering and separate simulator/encoder allocations. Timings include render, encoding, statistics, flush/remux, dataset commits and finalization; setup and validation are excluded. These are small pilot results, not full-collection guarantees.

Use the unchanged GOP 2 for the recommended pilot:

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.gpu_export \
  --source <collection> --output <fresh-pilot-directory> \
  --mode async --worlds 4 --image-stats uint8 \
  --render-batch-frames 4 --buffer-frames 2 --write-buffer-bytes 1048576
```

All options support the existing Tyro/YAML interface. Artifacts, exact configurations and sequential sweep scripts are in `C:\Users\steph\data\vla\ocbench\async-throughput`. Production remains paused and its exporter defaults are unchanged.

Every GOP 2 candidate produced byte-identical MP4s to the control, and all candidates produced identical dataset statistics. All runs passed complete video decoding/timestamp checks, pixel-exact replay checks, dataset reload and sampled action/state alignment. Longer GOPs change the encoded video; sampled image-quality checks passed, but they are not bit-identical alternatives.

A separate downstream check used LeRobot's actual PyAV loader, one CPU thread, and the same 64 random single-frame queries on the first front-camera episode, repeated three times in alternating configuration order. Median totals were 0.637 seconds for GOP 2, 0.770 for GOP 8, and 2.036 for GOP 50. GOP 8 saved about 53% of video storage but made these reads 21% slower for only a small export gain; GOP 50 made reads 3.2x slower. Keep GOP 2 for the throughput-oriented training workflow. This is a cached-file PyAV microbenchmark, not a measurement of full training or other decoder backends. `comparison.json` records the hashes, statistics comparisons, run reports and read timings.

Thirteen targeted tests passed, covering temporal pixel parity (including padded tails and nonsequential replay), ordered encoding with unequal episode lengths, bounded-buffer reuse, injected worker failures, exact statistics, graph replay and native LeRobot episode handoff. Ruff lint and format checks passed. The tests emitted the existing MJWarp MULTICCD warnings for unsupported contact pairs; these export paths restore states and render rather than simulate contacts.

### Production defaults and restart

Production `export` and the pipeline now default to `encoder_backend: async`, four episodes per batch, `render_batch_frames: 4`, `buffer_frames: 2`, 1 MiB packet buffers, exact uint8 statistics and GOP 2. Both success and valid-failure exports use the shared GPU replay/encoder implementation. Native LeRobot commits, previews, replay states and episode randomization/audit annotations are preserved. The final partial batch is supported. Encoding settings are stored in manifests and progress reports; an existing complete export is reused only when its selection, action profile and encoding settings match. Use `--encoder-backend cpu` for the software fallback. The WSL setup script now installs the `gpu-video` extra.

Fourteen targeted tests passed, including a production export across two batches with unequal episode lengths and a partial final batch, full decoding/timestamps, replay/preview metadata, dataset reload and completed-export compatibility checks. Ruff and shell syntax checks passed.

The 500-attempt production export was restarted on 2026-10-06 using these defaults. The interrupted 72-success CPU dataset is retained at `stack-500/datasets/successes-interrupted-cpu-20261006T171023Z`; the fresh export rebuilds all 352 audited successes and 46 valid failures from raw archives. `stack-500/resume-export.json` sets `train_steps: 0` for this export-only restart. The previous pipeline configuration is archived alongside it. `async-launch.json` records the process and log; `pipeline-async.log`, `status.json` and each dataset's `materialization.json` track progress. This launch does not imply the full export has completed.

### Encoder hang recovery

The first production GPU run stopped advancing after 156 success episodes. A captured Python stack showed an encoding worker blocked inside `PyNvVideoCodec.CreateEncoder` while the main thread waited in `ThreadPoolExecutor.shutdown(wait=True)`. The future timeout did not cancel the driver call, so cleanup could wait indefinitely. The driver eventually returned; interpreter teardown then aborted in PyArrow while trying to flush buffered episode metadata. The action/state parquet remained readable, but episode metadata lacked a valid footer. This identifies the blocked call and cleanup failure; it does not establish the underlying NVIDIA driver defect.

Encoder initialization is now serialized on the owning workers; frame encoding remains concurrent. Production async export runs in a supervised child process. After 180 seconds without a stage/progress update, the supervisor kills and reaps the worker, then retries from the last durable checkpoint (up to two retries). `worker_timeout_seconds` and `worker_retries` are configurable through CLI/YAML. Killing the process bypasses thread cleanup that cannot cancel a native driver call. Ordinary worker errors fail visibly rather than silently retrying invalid data.

Each completed batch now finalizes both data and episode-metadata parquet writers before atomically writing `checkpoint.json`. The next batch uses LeRobot's native recording-resume API. Resumption checks the selection, source provenance, action profile, encoding settings, episode count and frame count; mismatches are rejected instead of overwriting committed data. Encoder scratch from an uncommitted batch is disposable. Interruptions during a partial metadata commit are not silently rolled back: count/checkpoint mismatches require inspection. The watchdog's stage marker is `worker-state.json`; timeout details are written to `<dataset>.worker-status.json`.

Validation: 16 regression tests passed, followed by all three watchdog tests including a newly added retry/preservation test (17 distinct tests total). The production integration test interrupts between batches, reopens the checkpoint and verifies that already committed MP4s remain byte-identical. Separate stress validation completed 48 consecutive batches / 384 encoder session creations and closures. Ruff checks passed. Stack evidence and tests are under `async-throughput`; the stress artifacts are under `/home/stephen/data/vla/ocbench/hang-recovery`.

Production was restarted at 18:28 UTC on 2026-10-06 with `resume-export-recovery.json`, training disabled. Both earlier partial datasets are retained, including `datasets/successes-interrupted-gpu-156-20261006T182825Z`. The old 156-episode output cannot safely be appended because its metadata was not durably finalized; this restart rebuilds from raw archives. New datasets are stored at `/home/stephen/data/vla/ocbench/stack-500/datasets` via the new `dataset_root` pipeline option, avoiding limited Windows-disk space. The original `stack-500/datasets/successes` and `failures` paths link to these locations in WSL. `recovery-launch.json`, `pipeline-recovery.log`, `status.json` and dataset progress/checkpoint files track this run. A restart is not proof that the whole collection is complete.

Post-restart check: the fresh exporter reached a durable 20-episode checkpoint. The first batch's data parquet (5,151 frames) and metadata parquet (four episodes) both had readable footers and matched the checkpoint. This confirms the new checkpoint path in the live production run; the full collection was still exporting at this check.

### ACT success-only versus all-data comparison

The completed production export contains 352 successes (483,796 frames) and 46 physically valid failures (115,000 frames). Two ACT policies use the same recorded holdout, assigned before outcome filtering:

| Policy | Training episodes | Shared validation episodes |
| --- | ---: | ---: |
| Successes only | 278 successes | 74 successes + 8 valid failures |
| All audited data | 278 successes + 38 valid failures | 74 successes + 8 valid failures |

The 102 physically invalid raw attempts are outside this comparison. Both policies use 20,000 optimizer updates, batch size eight, seed 1000, 40-step action chunks and the same cameras/architecture. Loss probes run every 1,000 updates on fixed episode-balanced samples. Rollouts run every 2,000 updates on identical ten train and ten validation seeds, with a 2,500-step limit. Training and validation reset seeds are disjoint. Normalization is fitted separately from each arm's training episodes only; native normalized loss values therefore use each arm's own scale. Held-out task-success metrics use the same reset states and task criteria. Best/latest checkpoint retention is unchanged.

`ocbench_mjwarp.compare` prepares a native LeRobot aggregate without re-encoding, preserves source IDs, outcomes, split labels, randomization annotations and replay states, writes two training configurations, then trains sequentially. For this pinned LeRobot 0.6.1 integration, its module-local video-copy dependency is temporarily replaced with hardlink creation while concatenation is disabled. Immutable MP4s share storage with the sources; the aggregate must be on the same filesystem. Parquet index remapping uses LeRobot's implementation. The success-only configuration includes failure episodes solely in validation, never in its training sampler or normalization statistics.

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.compare \
  --config configs/ocbench/act-success-vs-all.yaml
```

Use fresh output paths for a new experiment. `--prepare-only` prepares without training; `--reuse-prepared` starts the existing prepared queue after checking its configuration. Shared trainer options `train_eval_seeds` and `val_eval_seeds` accept explicit CLI/YAML lists and validate their split membership. Changing these lists during checkpoint resume is rejected.

The comparison was launched on 2026-10-06. Dataset/configs, `comparison.json`, holdout hash, `validation.json`, queue status and `queue.log` are under `/home/stephen/data/vla/ocbench/act-success-vs-all-20261006`. Checkpoints and training logs are under `/mnt/e/vla-ocbench/act-success-vs-all-20261006` to distribute storage across drives. W&B project is `vla-ocbench`, group `act-success-vs-all-20261006`, run names `act-successes` and `act-all`. The second run is queued until the first finishes; a launch is not a completed training result.

Validation checked all 398 manifest episodes / 598,796 frames, equal holdouts, disjoint train/validation seeds, hardlinked video identity, boundary action/state/camera alignment across the merged sources and both training plans. Twenty-two split/training/checkpoint regression tests passed, including YAML roundtripping of explicit evaluation seeds. Ruff passed.

Startup verification: the successes-only run passed 125 optimizer updates, and its live sampler contains exactly 278 training episodes / 379,766 frames with no failures. Its live validation split hashes to the frozen shared holdout. W&B: [act-successes](https://wandb.ai/rlgoats/vla-ocbench/runs/91hk56zy). The all-data run remains queued and receives its run URL when it starts.

### Wrist-camera geometry investigation (2026-10-06)

The collection run's `examples/success/000001-wrist` is exported episode 1, raw episode 2, seed 83002. Its curved, see-through arm surfaces are reproduced in the saved preview. The upstream wrist camera is attached to `ur5e/wrist_3_link` at local position `(0, 0, -0.1)` with a 75-degree vertical field of view. That point can enter other arm links as the wrist rotates. In this example it is 20.4 mm inside the wrist-2 collision capsule at 4 seconds and 32.4 mm inside at 12 seconds. MJWarp enables backface culling by default, so a viewpoint inside the mesh can see through its outward-facing surfaces.

A CPU kinematics scan sampled every fifth raw episode and every 100th frame: the camera center was inside a robot capsule collision proxy in 390 of 1,629 poses across 100 episodes. These are geometric screening results, including invalid raw attempts, not a dataset-wide rate of visibly corrupted frames. Moving the camera to `(0, 0.08, -0.1)` produced zero capsule intersections in this sample, with 15.8 mm minimum clearance. This is a candidate mount, not a validated fix: visual meshes, the gripper, full trajectories, framing and other tasks still need checking. Larger offsets tested were not uniformly safer.

The model also specifies a 70 mm near plane. Lowering it to 5 mm changes native MuJoCo rasterization, but does not fix a camera embedded in geometry. In the installed MJWarp renderer, perspective rays start at the camera center and `znear` is used to construct ray directions; the primary intersection code does not discard near hits. Changing that value alone therefore does not address the exported artifacts.

Diagnostic scripts, `mount-scan.json`, and original/candidate contact sheets are under `/home/stephen/data/vla/ocbench/camera-audit`. `example-000001.jpg` compares the exact encoded W&B preview with native MuJoCo renders using the candidate mount and 5 mm near plane; lighting and renderer differences are intentional, so it is not a pixel-parity test. Production camera poses, datasets, training and evaluation remain unchanged. A future mount change needs a new rendering profile, re-rendered observations and matching training/evaluation cameras; existing raw simulation trajectories can be reused.

### ImageNet-normalization restart (2026-10-06)

The dataset-normalized ACT comparison was stopped at the user's request, including its queued all-data run. Its checkpoints, metrics and W&B history remain under `act-success-vs-all-20261006`. Fresh success-only and all-data policies now use ImageNet RGB mean/std with the same cameras, action representation, training episodes, shared holdout, evaluation seeds and 20,000-update budgets. The purpose is to isolate normalization; the wrist-camera mount has not changed.

The shared trainer defaults ACT to `dataset.use_imagenet_stats=true`. The split hook preserves the native factory's fixed image mean/std while computing state/action statistics exclusively from training episodes. Other policy defaults are unchanged; an explicit `image_normalization: dataset` YAML setting retains the previous ACT behavior. Checkpoints save the resulting preprocessing statistics for evaluation.

`configs/ocbench/act-imagenet.yaml` defines the restart. Prepared configs, validation record, launch and queue log are under `/home/stephen/data/vla/ocbench/act-imagenet-20261006`; success-only training artifacts are under `/mnt/e/vla-ocbench/act-imagenet-20261006/successes`. The prepared all-data config writes to `/home/stephen/data/vla/ocbench/act-imagenet-20261006/training-all` to distribute checkpoint storage across drives; this per-arm path override is recorded in `comparison.json`. Both arms run sequentially in W&B group `act-imagenet-20261006`. Holdout hash, training IDs, evaluation seeds and source manifest hashes match the stopped comparison. Videos are hardlinked without re-rendering. Twenty-four regression tests passed, including image-stat preservation and exclusion of held-out state/action statistics; Ruff passed.

### Stable-stack diagnostic and action replay

The stable-stack diagnostic now computes vertical separation from the sum of the two block geom half-heights instead of assuming 4 cm blocks. OCBench's 6 cm blocks are handled correctly. Synthetic MuJoCo tests cover both sizes, unsupported gaps, unreleased grippers, motion and insufficient history. Native collection termination and success filtering are unchanged. Existing saved diagnostic annotations are not rewritten, and already-running Python processes may retain the earlier imported diagnostic.

```bash
MUJOCO_GL=egl ocbench-mjwarp replay-check \
  --source /path/to/collection --output /path/to/fresh-replay-report \
  --episodes 0 2 3 --atol 0.001
```

The command supports the usual YAML configuration and CLI overrides. It restores recorded initial simulator states, sends the recorded normalized actions through `OCBenchVectorEnv.step`, and retains the normal observation/rendering path. It compares pre-action proprioception, post-action simulator fields, episode lengths and native/audited outcomes. `report.json` records maximum absolute errors by field and the first mismatch; `progress.json` tracks long checks. A mismatch exits nonzero. The tolerance is a numerical screening threshold in each field's native units, not a physical equivalence certificate. This checks simulator/action integration, not policy normalization or encoded-video pixel parity. It does not overwrite source data or existing reports.

The first three-world pilot on raw episodes 0, 2 and 3 is saved at `/home/stephen/data/vla/ocbench/replay-check-20261006/report.json`. Episodes 2 and 3 recovered both native and audited success; episode 0 did not recover native success within its recorded 2,159 steps. All three exceeded the strict 1e-3 trajectory threshold, first in joint velocity at frames 225, 409 and 286 respectively. Maximum qpos errors were 0.1638, 0.00698 and 0.0000785. The delayed divergence warrants investigation of contact sensitivity, batch-dependent numerics and simulator state not included in the archive; the pilot does not establish a root cause. It is a failed replay check, not proof that policy evaluation is equivalent to generation. Training was left running. Four regression tests passed, including intentionally corrupted replay actions; Ruff passed.

#### Replay divergence investigation

Follow-up scripts and JSON reports are under `/home/stephen/data/vla/ocbench/replay-investigation`. Production physics, collection, datasets and training were not changed.

- `controls.json`: restoring each recorded pre-action state and applying the native action-to-control conversion reproduced all 2,159 saved actuator-control vectors of episode 0 exactly (maximum absolute error zero). Selected single-step checks showed small position errors, including 3.18e-5 at frame 225, while velocity discrepancies could exceed the strict replay threshold. This rules out a scale/index/control-conversion mismatch for this episode; it is not a full dataset or policy-processor audit.
- `probe.json`: replay without rendering or the evaluation wrapper still diverged. Trials used one, three and the original 32 worlds, and included both native reset and explicit state restoration. Recorded initial state fields matched native reset exactly. Two otherwise identical restored single-world runs reached maximum qpos errors of 0.1356 and 0.000197 over 500 actions. Matching the original batch size did not remove divergence.
- `twins.json`: two fresh single-world simulators received identical recorded actions. Initial qpos, qvel, ctrl, mocap state, time, qacc, qacc_warmstart, applied forces, actuator state and equality activation all matched. Their velocities differed by 1.86e-9 after the first action, exceeded 1e-3 at frame 225, and differed by as much as 0.1456 over 500 actions. The maximum qpos difference was 0.000806. This reproduces nondeterminism independently of archive restoration and demonstrates amplification during constrained dynamics; the exact nondeterministic kernel was not isolated.
- `twins-20.json`: increasing only the diagnostic simulators' solver iterations from the upstream default six to twenty did not eliminate divergence (maximum twin qpos difference 0.0480 over 500 actions). This single pilot is not a statistical comparison of solver quality and does not justify changing production solver settings.

The evidence supports native floating-point/constraint-solver nondeterminism and sensitive contact dynamics as a replay limitation. Missing archived solver state is not required to reproduce it, though these checks do not prove that the archives contain every state needed for arbitrary mid-episode resumption. The generator recomputes actions from current state; replaying its frozen action sequence removes that feedback. Failed open-loop replay therefore does not alone establish an evaluation-wrapper defect or explain the learned policies' low success. Keep strict trajectory checks separate from actuator-conversion parity and repeated outcome-recovery measurements; do not silently relax tolerances until the replay is declared successful.

### 24-step open-loop comparison

At the user's request, the 40-step ImageNet run and its queued all-data run were stopped and preserved. `configs/ocbench/act-chunk24.yaml` starts fresh ACT policies with `policy.chunk_size=24` and `policy.n_action_steps=24`. Temporal ensembling remains disabled: each prediction executes open-loop for 0.48 seconds at 50 Hz. The comparison CLI now exposes `chunk_size` through Tyro/YAML, and uses it for both prediction and execution. Its default was subsequently changed to 25 for future runs; the active 24-step recipe and historical 40-step recipes remain explicit.

New comparison metadata and queue logs are under `/home/stephen/data/vla/ocbench/act-chunk24-20261006`. The success-only policy writes to `/mnt/e/vla-ocbench/act-chunk24-20261006/successes`; the prepared all-data config writes to the comparison root's `training-all` directory to distribute storage. Training IDs, shared holdout, evaluation seeds, images, ImageNet normalization and 20,000-update budgets match the previous run. The second policy remains queued until the first completes.

### Comparison with OCBench's reference training

Inspected the clean installed source checkout at commit `e2cd2f7`, particularly `impls/commands.sh`, `impls/agents/bc.py`, `impls/main.py`, `impls/envs/env_utils.py`, `impls/utils/datasets.py`, `impls/utils/evaluation_mjwarp.py` and `ocbench/envs/block_env.py`.

| Setting | Reference `block-double-task2-v0` BC command | Our current ACT comparison |
| --- | --- | --- |
| Observations | Proprioception plus privileged block positions, quaternions and yaw features | Two RGB cameras plus 18D proprioception; no object state |
| Model/loss | JAX residual MLP, eight 4096-wide layers, L2 BC | ACT, pretrained ResNet18, L1 plus VAE KL |
| Prediction/execution | 25 actions, fully open-loop | 24 actions, fully open-loop |
| Updates / batch / learning rate | 2,000,000 / 1,024 / 1e-4 | 20,000 / 8 / 1e-5 |
| Action normalization | 1st/99th-percentile scaling, normalized clipping at ±5; gripper retains ±1 | Training-only mean/std, including gripper |
| Episode-tail targets | Repeat terminal action and include it in BC loss | Padded actions masked from ACT loss |
| Evaluation | 500 episodes, native success | 10 train + 10 held-out resets per checkpoint, native and contact-audited success separately |

The reference data-generation command requests 30,000 attempts for this task; our collection contains 500 attempts. Reference BC uses successful trajectories; ours additionally applies physical audits. Its total sampled chunk-start budget is 12,800 times larger, although that is not a compute-equivalence comparison between the architectures. These reference commands do not establish results for our visual ACT setting.

The repository also supports visual agents: its encoders scale uint8 RGB by 255 and initialize their own network parameters, rather than use our ImageNet-pretrained ResNet pipeline. Keep ImageNet preprocessing for our chosen backbone. The inspected command file does not supply a matching visual `block-double-task2-v0` training recipe. The useful follow-up comparisons are action normalization and controlled state-based/visual baselines; the reference already supports the requested approximately half-second open-loop execution scheme.

### Future budget and success filtering

The selected future budget is 100,000 updates at batch eight, with 25-step prediction and open-loop execution. `configs/ocbench/act-chunk25.yaml` records this recipe with loss probes every 1,000 updates and rollouts every 5,000. The comparison defaults now use 100,000 updates and 25-step chunks. This does not restart or extend the active, explicitly configured 24-step/20,000-update comparison.

Success filtering is episode-level: `native_success and physical_valid`. Physical validity requires healthy, finite simulation without disallowed overflow and maximum robot/environment penetration within 1 mm for non-pad geometry and 3 mm overall. Native success is the upstream block-position criterion; released stable stacking is diagnostic only. Whole successful trajectories are retained, including missed grasps, retries and recovery actions before the first native success signal. Of the 352 exported successes, 143 have a recorded plan with `num_pick_retries > 0`: 106 of the 278 training successes and 37 of the 74 held-out successes. These counts describe recorded pick-retry annotations, not manually verified regrasp events. Failed actions inside an eventually successful episode are not trimmed or masked from behavior cloning.

Future comparisons now default to `exclude_pick_retries: true`: the success-only training arm requires every recorded plan's `num_pick_retries` to be zero. Missing or incomplete retry annotations are rejected. This excludes whole episodes, not individual actions; it is a recorded-pick-retry filter rather than a guarantee against every physical regrasp. The current dataset supplies 172 eligible training successes (106 excluded). The all-data arm retains 316 training episodes, and both arms retain the same 82 held-out episodes, including retries. Training normalization uses the selected training subset. `comparison.json` records the filter and excluded aggregate episode IDs. Existing exports and prepared runs are unchanged; historical recipes explicitly disable this new filter.

### Future action representation and normalization

The 25-step recipe and new comparison defaults enable `absolute_gripper` and `percentile_normalization`. Existing recipes explicitly disable both. Training builds an in-memory action view before chunk sampling: the six arm coordinates remain native normalized deltas, while the gripper target is `clip(state[16] + 0.12 * clip(action[6], -1, 1), 0, 1)`. This reconstructs the commanded opening, not the next measured joint position or a binary open/close intention. Source parquet, videos and raw archives stay unchanged. Evaluation applies the predicted absolute opening directly to the gripper actuators, with native arm control and physics.

Each arm fits exact 1st/99th percentiles for state and action from its selected training frames only; validation uses those same statistics. LeRobot's QUANTILES processors map those bounds to [-1, 1] without clipping outliers. The absolute gripper uses fixed physical [0, 1] bounds, mapped to [-1, 1]; near-constant feature ranges below 1e-6 use a unit interval to avoid epsilon amplification. Images keep ImageNet normalization and 640×480 resolution. Statistics are saved in `train_stats.json` and checkpoint processors; the versioned action profile records the gripper semantics. Resume rejects changes to either setting. These changes do not restart existing training.

Validation: nine targeted tests passed for retry filtering, train-only statistics, per-frame action conversion, absolute actuator clamping, checkpoint profiles and image normalization. A real LeRobot dataset check transformed 1,428 training frames and 2,159 validation frames without modifying source files. Reconstructed gripper actuator commands matched all 2,159 archived commands in raw episode 0 exactly. No new policy training or full absolute-control rollout has been run yet.

### Relaunch: 25-step percentile/absolute-gripper comparison

On 2026-10-06, the 24-step queue was stopped at approximately 8,425 updates and its saved checkpoints were preserved. The new comparison uses `configs/ocbench/act-chunk25.yaml`: 100,000 updates per arm, batch eight, 25-step open-loop execution, no recorded pick retries in success-only training, training-only percentile state/action statistics and absolute gripper commands. Rollouts run every 5,000 updates, loss probes every 1,000. Cameras remain 640×480 with ImageNet normalization; AMP, resizing and compilation are not enabled.

Prepared metadata, launch record, validation and queue log are under `/home/stephen/data/vla/ocbench/act-chunk25-100k`. Both arms write checkpoints beneath its `training` directory because E: had only 608 MB free. The shared 82-episode holdout hash and held-out rollout seeds match the preceding run; training counts are 172 successes and 316 all-data episodes. The success-only arm runs first, followed automatically by the all-data arm. W&B group is `act-chunk25-100k`. A native GPU simulator step verified direct absolute-gripper actuator control before launch.

Success-only W&B run: https://wandb.ai/rlgoats/vla-ocbench/runs/52suvpbo. The live loader reports 172 training episodes / 211,230 frames, and saved statistics confirm percentile state/action bounds, the fixed absolute-gripper range and ImageNet RGB statistics.

### Demonstration parity rerun with absolute gripper and percentile processors

Artifacts: `/home/stephen/data/vla/ocbench/parity-rerun-20261006` (`summary.json`, per-mode reports, `controls.json`, `images.json`, `images.jpg` and reproducible audit scripts). Replayed the original raw episodes 0, 2 and 3: episode 0 is held out and has a recorded retry; 2 and 3 are retry-free training episodes. Processor files were copied from the 15,000-update checkpoint before retention could remove them. Training continued during diagnostics.

| Replay mode | Native/audited successes | Strict trajectory passes (1e-3 tolerance) |
| --- | --- | --- |
| Native delta actions | 2/3 | 0/3 |
| Absolute-gripper targets | 3/3 | 0/3 |
| Absolute targets through saved 25-step percentile processors | 3/3 | 0/3 |

Across all 4,204 saved pre-action states, native and absolute-gripper conversion reproduced archived actuator controls exactly. Saved processor round trips differed by at most 1.788e-7 in actions and 3.815e-6 in actuator controls. All three modes still diverged from recorded trajectories; native episode 0 again lost success. These small trials do not establish comparative success rates. They support action/control conversion parity and confirm the existing simulator replay limitation; they do not explain the learned policy's low success. The diagnostic was extended with `--absolute-gripper` and `--checkpoint` (saved processor round trip, without policy predictions). Six regression tests and Ruff passed.

For both cameras at frames 0, 100, 300 and 600 of all three episodes, rendering restored archived states matched exported images closely: 24 views, mean absolute pixel difference 0.933–1.222 on the 0–255 scale. Visual inspection of paired frames found matching framing and object placement, consistent with lossy video differences. Restored proprioception differed by up to 9.356e-4; the test does not claim bitwise state/image identity. Wrist-camera arm occlusion persists in both views.

Normalization statistics are pooled over all selected training frames: one q01/q99 pair per state/action coordinate, broadcast across batches and chunk timesteps. They are not fitted separately per temporal position. Each comparison arm fits its own training statistics, and its validation/evaluation uses those saved values.

### Policy failure investigation at 15k updates

Pinned the 15,000-update success-only model and processor files under `/home/stephen/data/vla/ocbench/policy-investigation-20261006/checkpoint-015000`. `audit.py`, `report.json`, `teacher.npz`, `rollout.npz`, `rollout-analysis.json`, `replan1/` and `trajectories.png` record a controlled diagnostic on raw episodes 0 (held out), 2 and 3 (retry-free training). Production training and its 25-step execution setting were unchanged.

- Sampled every 25 frames with full 25-action chunks. The real LeRobot loader's transformed targets matched raw-archive-derived targets exactly (maximum error zero). Both training and evaluation convert RGB to [0, 1] before saved ImageNet normalization. The native ACT queue was instrumented: predictions occurred at 0, 25, 50, … with no unexpected resets.
- On demonstration observations, native arm-action MAE was 0.02724 across the three episodes versus 0.06668 for zero arm motion. Per-episode arm errors were 0.03826, 0.01593 and 0.01463. Gripper MAE was 0.04826. Images matter: blank normalized images increased arm error to 0.03380 and gripper error to 0.11658; within-batch image permutation increased them to 0.03018 and 0.07217. The blank-image test is out of distribution, and permutation mostly compares nearby frames, so these are sensitivity checks rather than attribution proofs.
- Autonomous 25-step execution achieved 0/3 native successes. Joint trajectories exceeded 0.05 rad deviation from the demonstrations at steps 36, 33 and 23, respectively. Neither block moved; minimum pinch-to-block-center distances were 0.154, 0.193 and 0.200 m. Joint 6 eventually reached approximately +2π in all three worlds. First-chunk predictions from exported versus freshly rendered initial observations differed by approximately 0.0024 MAE in native action units, consistent with the small input-image differences already measured.
- A diagnostic using the same model but executing only one action per prediction also achieved 0/3, with no block movement and joint 6 reaching +2π. Minimum distances were 0.345, 0.228 and 0.178 m. Replanning alone did not resolve this small pilot; it does not establish that all execution horizons are equivalent.
- 25 of the 172 selected successful training episodes still contain `is_mistake` plan annotations (upstream calls these potential high-level mistakes). This differs from pick retries and was not excluded by the requested retry filter. No labels or training selection were changed in this investigation.

The demonstrated failure is approach/control drift before contact, not rejection of a completed stack by the success audit. Target alignment, action inversion and tested observation processing show no clear integration defect. Prediction errors, distribution shift and ambiguity in randomized demonstrations remain plausible contributors; their individual effects are not established. A controlled tiny-dataset overfit check and an absolute-arm-target comparison are more informative next experiments than changing the success threshold or only shortening execution.

### Clean first-attempt successes: absolute versus relative arm actions

`configs/ocbench/act-clean-actions.yaml` launches the requested action-representation comparison. Here “clean, zero-shot” is an annotation-based filter: native success and physical validity, with every recorded plan having both `num_pick_retries == 0` and `is_mistake == 0`. Missing required annotations fail preparation. This leaves 147 training episodes / 183,596 frames; it is not a manual guarantee of humanlike motion or stable released stacking. The existing 82-episode holdout remains unchanged, including failures/retries, for comparable validation losses and rollout evaluation. No held-out episodes enter training statistics.

Both policies use exactly the same clean training IDs, shared holdout and reset seeds, ImageNet-normalized 640×480 images, training-only 1st/99th-percentile statistics, 100k updates, batch eight, seed 1000 and 25-step open-loop execution. Loss probes run every 1k and rollouts every 5k. Both use absolute gripper position. Only the six arm coordinates differ: normalized native deltas versus absolute commanded joint angles in radians. The latter are reconstructed as current recorded joint position plus scaled/clipped delta, then clipped to native actuator limits; these are not next measured positions. Execution applies absolute targets directly to position actuators. Profiles record the representation and limits; the simulator checks those limits against the native model. Normalized losses across action representations are not directly equivalent physical errors; use matched rollout success as the primary comparison.

Preparation and validation are under `/home/stephen/data/vla/ocbench/act-clean-actions-20261006`, with `comparison.json`, `clean-selection.json`, `validation.json`, prepared configs, `launch.json` and `queue.log`. Model artifacts go under `training/absolute` and `training/relative`. The absolute run starts first; relative follows automatically. The old 25-step success/all-data queue was stopped at the last observed update 25,000 and its checkpoints preserved.

Preflight artifacts are under `/home/stephen/data/vla/ocbench/absolute-arm-parity-20261006`: all 4,204 actuator vectors in raw episodes 0, 2 and 3 matched exactly after reconstructing absolute targets. GPU execution passed, and full absolute-action demonstration replay recovered 3/3 audited successes. Strict trajectory equality still failed, so this is not a claim of deterministic replay. Thirteen targeted regression tests passed.

Absolute-action W&B run: https://wandb.ai/rlgoats/vla-ocbench/runs/zo4fsgh2. The relative-action run is queued in the same `act-clean-actions-20261006` group.

## ACT at 320×240 with AMP

`configs/ocbench/act-clean-actions-320-amp.yaml` repeats the clean absolute-versus-relative comparison with `image_size: [240, 320]` (height, width), `use_amp: true` and `amp_dtype: bfloat16`. Both arms retain the same 147 training episodes, 82 held-out episodes, evaluation seeds, 100k updates, batch size eight and 25-step open-loop chunks. ImageNet and percentile normalization remain unchanged.

The saved LeRobot processor resizes both cameras with antialiasing before normalization during training, loss probes and rollouts. Source videos and simulator camera profiles remain 640×480; video decoding is still full resolution. Checkpoint processors carry the resize and convert predicted actions to float32 before unnormalization; `precision.json` records the evaluation autocast dtype. Resume rejects changes to resolution or precision. AMP checks the trainer's actual precision and rejects nonfinite losses/gradient norms. Native W&B metrics include throughput and peak GPU memory.

```bash
MUJOCO_GL=egl python -m ocbench_mjwarp.compare \
  --config configs/ocbench/act-clean-actions-320-amp.yaml
```

The full-resolution comparison was stopped at the user’s request to replace it with this pair. Use `--reuse-prepared` if the comparison dataset/configs have already been prepared. This comparison measures the combined resolution/precision change, not separate causal effects of each.

Preflight: 75 updates including checkpoint resume passed with finite BF16 losses/gradients, both loss probes, five-step train/validation rollouts and a fresh-process checkpoint evaluation. Twenty distinct regression tests passed, including serialized resizing and float32 action unnormalization. The preflight exposed and fixed BF16-to-NumPy conversion in rollout actions. These short rollouts verify integration, not learned success. Reports are in `/home/stephen/data/vla/ocbench/act-clean-actions-320-amp-20261006/validation.json`; [preflight W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/11zpvd32).

Replacement training: [absolute ACT](https://wandb.ai/rlgoats/vla-ocbench/runs/5y066q7m) is running, with relative ACT queued in the same comparison. Startup passed 225 updates with finite BF16 losses and gradients. The old pair stopped at 15k observed absolute updates; its relative arm never started. Existing checkpoints were retained. Disposable uv download caches were pruned to leave space for the new checkpoints.

### TorchCodec decoding

OCBench training/comparisons now default to `video_backend: torchcodec`; `pyav` remains an explicit fallback. The selected backend is used by both the native training factory and the custom train/validation split loaders. This is LeRobot's cached CPU TorchCodec decoder, not GPU/NVDEC decoding. Source videos remain 640×480 and policy resizing remains 320×240.

WSL requires FFmpeg shared libraries (`sudo apt-get install ffmpeg`); the setup script checks the TorchCodec import. The shared environment uses TorchCodec 0.11.1 with FFmpeg 6.1.1. Thirty-six sampled frames across six front/wrist videos matched PyAV exactly, including timestamp checks; eleven split-loader/hook tests passed. Decoder selection can change on resume without changing actions, normalization, or data splits.

The active absolute ACT run switches at its 5,000-update checkpoint under the same W&B identity; the queued relative run also uses TorchCodec. `decoder-transition.json` and `torchcodec-parity.json` in the comparison directory record the transition and checks. Sparse multi-frame decode timings in the parity report are diagnostic, not an end-to-end training benchmark.

Update profiling (`update-profile.json`/`.txt`): a synthetic GPU-resident batch with the current ACT shapes/checkpoint took ~70 ms/update without profiling. The five-update trace recorded 10,340 CUDA runtime kernel launches and ~63 ms of operator-attributed GPU kernel time in total. This points to host dispatch/autograd/launch overhead at batch eight; profiler CPU timings are inflated and these measurements are not an isolated end-to-end benchmark. The installed ACT also requests attention weights that it discards, and synchronizes loss scalars inside forward. Compilation/CUDA graphs and avoiding unused attention weights are candidates to benchmark; neither was enabled in this run.

## Workflow consolidation (2026-10-06)

Removed the OGBench package, launchers and recipes. Its validation records are retained under `docs/archive/ogbench`; external datasets/checkpoints and the running ACT comparison were left untouched. Shared code no longer imports simulator projects. OCBench owns rendering, contact diagnostics, action views and rollout construction; LIBERO retains its independent environment and commands.

Production export is the sole dataset writer, with CPU fallback and the existing bounded queues, watchdog and durable checkpoints. The experimental runner and rendered-video re-export path are removed. `python -m ocbench_mjwarp.benchmark_render --source <raw> --output <fresh-output> --limit 2` measures the production exporter, including initialization, commit time and peak allocated GPU memory.

`prepare-dataset` combines audited exports once, links immutable videos/replays where possible, and records source fingerprints and the canonical split. Comparisons reference `--dataset`, select episode IDs, and retain a shared unfiltered holdout. Existing combined comparison datasets work directly. Preparation reuse rejects incomplete output or changed provenance.

Pipeline/comparison recipes now embed `training`; native overrides are a mapping and `action_mode` replaces the arm/gripper booleans. One adapter installs preprocessing, optimizer logging and checkpoint evaluation around the native LeRobot loop. The worker constructs a typed native config directly. Modern saved specs/checkpoints remain readable through one compatibility conversion; pre-image-normalization checkpoints are unsupported. Supported current resume and queued relative-arm settings were checked against the real experiment without launching or altering it.

Validation: 82 shared/OCBench tests passed, including GPU replay, async ownership/failure handling, native aggregation, export reload and retry recovery; all seven LIBERO tests passed in its separate environment. A final 28-test pass covered the moved GPU primitives, dataset preparation, and typed-override protection. A separate small ACT model trained for two updates with BF16 AMP, resumed for one update, saved/reloaded its processors, and completed train/validation loss probes and one-step simulator evaluations. Standalone checkpoint reload/evaluation and a two-frame recorded-action check with absolute-gripper target conversion also passed. The recorded check confirmed the removed OGBench module is not importable. This smoke test verifies integration, not policy quality or throughput.

Runtime Python shrank from approximately 5,642 to 5,258 lines across shared utilities and OCBench (including comments/blank lines), separately from removing approximately 9,126 OGBench runtime lines. Shared regression tests moved out of the removed package; preparation/configuration/resume tests were added.

## Dataset browsing and WSL graphics

Rerun 0.38.1 directly opens LeRobot v3 directories, with each episode represented by a separate recording. Switch recordings in the viewer without restarting. This differs from `lerobot-dataset-viz --episode-index ...`, which loads one episode. The isolated `uvx` command avoids changing the training environment or its older Rerun dependency constraint.

```bash
DATASET=/home/stephen/data/vla/ocbench/act-clean-actions-320-amp-20261006/dataset
uvx --from 'rerun-sdk==0.38.1' rerun "$DATASET"
```

This opens all 398 stored episodes, not just the 147 clean training episodes and 82 held-out episodes selected by the current comparison. The authoritative selections are in `training/absolute/split.json` beneath that experiment directory. Training examples include indices 1, 2, 5, 8, 9, 10 and 14; validation examples include 0, 4, 6, 11, 15 and 19. Indices refer to the combined dataset, not raw collection attempt IDs.

The viewer displays stored 640×480 images and native delta actions. Training resizes images to 320×240 and applies its configured action conversion and normalization in memory; those transformations are not applied by the standalone dataset viewer.

For accelerated display, serve from WSL and open the viewer in Windows Chrome or Edge:

```bash
uvx --from 'rerun-sdk==0.38.1' rerun --web-viewer --renderer webgpu "$DATASET"
```

Use the URL printed by Rerun in the Windows browser, even if automatic browser launch from WSL fails. The default web port is 9090. Enable “Use graphics acceleration when available” in browser settings and restart the browser if needed. Check `chrome://gpu` or `edge://gpu` to verify WebGPU is hardware accelerated. `--renderer webgl` is a fallback if WebGPU is unavailable. GPU rendering and hardware video decoding are separate: Rerun's decoder defaults to `auto`; `--video-decoder prefer_hardware` is only a preference and can fail if no compatible decoder exists.

For a native WSLg window, Rerun supports `--renderer gl` as an alternative to its default Vulkan backend. WSLg can accelerate OpenGL through Mesa's D3D12 driver. Keep the Windows GPU driver current; CUDA availability alone does not confirm accelerated viewer rendering. If X11 startup reports missing `libxkbcommon-x11.so.0`, install `libxkbcommon-x11-0`. For adapter diagnostics, install `mesa-utils` and run `glxinfo -B`; a renderer such as `D3D12 (NVIDIA ...)` indicates the GPU path, while `llvmpipe` indicates software rendering. `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` can select the NVIDIA adapter when Mesa D3D12 is available. For a Wayland-specific startup failure, try `env -u WAYLAND_DISPLAY ...` after installing the X11 library.

Local checks on 2026-10-06 found `/dev/dxg`, the WSL D3D12 libraries and Mesa's D3D12 driver. Rerun 0.38.1's CLI flags were verified. A headless OpenGL probe selected llvmpipe and failed to create its render context; the X11 probe stopped at the missing `libxkbcommon-x11` library. These checks do not establish native WSLg acceleration or browser acceleration; verify the actual adapter as above. No graphics drivers, global environment settings or training dependencies were changed.

References: [Rerun LeRobot loading](https://rerun.io/docs/howto/logging-and-ingestion/lerobot), [Rerun graphics options](https://rerun.io/docs/reference/cli), [Rerun troubleshooting](https://rerun.io/docs/getting-started/install-rerun/troubleshooting), [WSLg GPU support](https://github.com/microsoft/wslg#opengl-accelerated-rendering).

## Single-trajectory fitting diagnostic (2026-10-07)

The interrupted absolute/relative comparison remains stopped. A separate fresh ACT run uses combined-dataset episode 1 (raw attempt 2, reset seed 83002): 1,428 frames at 50 Hz, native/audited success, and zero recorded pick retries or mistakes. This selection does not assert visibility or released stable completion; its archived stable-stack diagnostic is false.

Recipe: `configs/ocbench/act-overfit-episode001.yaml`. Run artifacts: `/home/stephen/data/vla/ocbench/act-overfit-episode001-20261007`; the resolved config, source episode metadata, process launch and log are retained there. Budget is 5,000 updates at batch eight, from scratch, preserving absolute arm/gripper targets, 25-step open-loop execution, 320×240 images, ImageNet vision normalization, BF16 AMP and TorchCodec. State/action quantiles are fitted only on the selected trajectory. Native ACT optimizer and KL settings remain unchanged.

The trainer now supports an explicit `overfit` mode. It rejects multiple episodes, nonzero validation fraction, and (for OCBench) episodes that are held out, unsuccessful, physically invalid, missing retry/mistake annotations, or annotated with retries/mistakes. It uses the existing dataset/action/preprocessor/checkpoint path. Loss probes cover all 1,428 frames every 250 updates; same-reset simulator evaluation runs every 1,000 updates and at completion. Reports are labeled `single_episode_overfit`, contain no validation partition metrics, and retain the latest checkpoint. Memorization loss and same-reset rollout outcomes are diagnostic, not held-out performance.

Before launch, 32 focused configuration, normalization, split, checkpoint and comparison tests passed, including a single-episode checkpoint probe with an empty validation partition and preservation of the existing train/validation behavior. The new run does not resume or modify the interrupted comparison.

W&B: https://wandb.ai/rlgoats/vla-ocbench/runs/6k0lf4wz. Live startup confirmed exactly one training episode, 1,428 frames, and an empty validation partition. The first 250-update checkpoint/probe completed: inference-mode normalized L1 was 0.08316 across all 1,428 frames. This is an early fit measurement; the first same-reset rollout is scheduled at 1,000 updates, so no rollout-success claim is made yet. Eight additional clean-selection/CPU-metric tests passed. The prior comparison's 55,000-update checkpoint passed the resumable-state integrity check and remains untouched.

### Relative-action single-trajectory overfit (2026-10-07)

Launched `configs/ocbench/act-overfit-relative-episode001.yaml` alongside the absolute-action pilot. The only modeling change is `action_mode: absolute_gripper`: relative arm actions, absolute gripper opening. Episode 1, seed 1000, reset seed 83002, 5,000 updates, batch 8, 25-step open-loop chunks, 320×240 images, BF16 AMP, TorchCodec, ImageNet image normalization and train-only percentile normalization match the absolute run. Both evaluate the same training trajectory/reset; neither provides held-out validation. Normalized probe losses use different action-space statistics, so compare rollout behavior as well as loss.

Artifacts: `/home/stephen/data/vla/ocbench/act-overfit-relative-episode001-20261007` (`train.yaml`, `episode.json`, `launch.json`, `run.log`, and `training/`). W&B group: `single-trajectory-overfit`; run name: `act-relative-episode001`. The W&B layout cleanup is deferred until after this launch.

Relative run: https://wandb.ai/rlgoats/vla-ocbench/runs/ddv5arjf. Absolute reference: https://wandb.ai/rlgoats/vla-ocbench/runs/6k0lf4wz.
