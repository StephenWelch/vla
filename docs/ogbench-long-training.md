# OGBench longer training

ACT and SmolVLA share the same annotated dataset, scene split, training-only normalization statistics, and fixed rollout suites. The default recipes run 20,000 updates each, loss probes every 1,000 updates, and simulator evaluation every 2,000 updates. This workflow uses the isolated OGBench environment and LeRobot 0.6.1.

## Run the pipeline

From the repository root in WSL, after running the OGBench setup script:

```bash
export OGBENCH_DATA_ROOT=/mnt/e/vla-ogbench
source "$OGBENCH_DATA_ROOT/venv/bin/activate"
export MUJOCO_GL=egl
vla-ogbench-pipeline --runs "$OGBENCH_DATA_ROOT/runs" --experiment cube-wandb-01 --background
cat "$OGBENCH_DATA_ROOT/runs/long-training-status.json"
tail -f "$OGBENCH_DATA_ROOT/runs/cube-wandb-01/pipeline.log"
```

The detached WSL process collects demonstrations, trains ACT from scratch, and then fine-tunes the existing SmolVLA base checkpoint. It survives closing the initiating terminal, but requires WSL and the machine to remain running. Omit `--background` for a foreground run. The command refuses to start another pipeline while its recorded PID is alive.

Collection targets at least 100 successful independent cube-single task 1 scenes, with two attempted trajectory variants per reset. It processes 20 scenes per round with up to 32 concurrent episodes by default for the 32 GB GPU, and stops after at most 1,000 attempted scenes. Override concurrency with `--generation-batch-size`; batch size is an episode count, not a VRAM allocation target. Existing rounds retain their recorded planner settings on resume; new rounds use the requested batch size. Only successful, contact-valid episodes are exported. Round completion can exceed the target slightly; failures and their diagnostics remain in the raw archives. Raw collection is restartable. Generation retains the existing path, grasp, timing, order, and post-IK joint-target randomization annotations. Its compact CEM budget matches the diversity pilot: eight candidates, horizon eight, two iterations, and joint-target noise of 0.01 radians.

Default locations:

- Raw demonstrations: `E:/vla-ogbench/raw/<experiment>`
- Annotated LeRobot dataset: `E:/vla-ogbench/datasets/<experiment>`
- Policies: `E:/vla-ogbench/runs/<experiment>/act` and `smolvla`

Pipeline paths and the scene target accept Tyro flags or YAML through `--config`. Each pipeline saves its resolved `pipeline-config.json`; rerun with this file to reuse collection and resume available policy checkpoints. Pass an existing complete v2 dataset through `--dataset` to skip generation. New pipelines enable [W&B tracking](ogbench-wandb.md); the individual `*-long.yaml` recipes below retain disabled logging. A failed pipeline stops before subsequent phases. Legacy datasets are unsupported; historical raw archives, checkpoints, and reports remain available.

Generation renders front and wrist together before each action, using execution worlds only. CEM candidate worlds have no renderer. Two writing threads compress completed archives with at most four outstanding writes; metadata is committed after its archive. Backpressure bounds memory. The v2 dataset manifest and each policy checkpoint retain camera settings, model identity, image resolution, and renderer versions; evaluation rejects mismatches. State-only v2 recordings can be rendered later with `ogbench-mjwarp rerender --source RAW --output NEW_RAW --batch-size 32`, then exported. The MuJoCo WSLg viewer remains available for state playback.

New pipelines reuse finished execution slots, batch CPU state transfers, and export completed rounds concurrently through one dataset writer. Export streams video directly with blocking encoder queues; metadata and raw episode ordering remain deterministic. The [W&B workflow](ogbench-wandb.md) documents controls, timing reports, and baseline comparisons.

## Train or resume one policy

After collection:

```powershell
.\scripts\train-ogbench.ps1 -Config configs/ogbench/train-ogbench-act-long.yaml
.\scripts\train-ogbench.ps1 -Config configs/ogbench/train-ogbench-smolvla-long.yaml
```

The launcher accepts YAML defaults followed by Tyro overrides. For example, `--steps 5000 --output /mnt/e/vla-ogbench/runs/act-short-v2` creates a shorter fresh run. Batch size defaults to eight and action chunks to 16. New collection renders 640×480 front/wrist RGB, matching SO-101 wrist camera capture; existing v2 datasets retain their original 32×32 diagnostic images. Training and evaluation derive camera shapes from the dataset and checkpoint rendering profile.

Resume with the same recipe/output and `--resume /mnt/e/vla-ogbench/runs/act-long-v2/checkpoints/last/pretrained_model`. `steps` means the total target update count, not additional updates. Resume restores LeRobot optimizer, scheduler, RNG, and episode sampler state. Dataset metadata, split, architecture overrides, batch size, and evaluation budgets must match. Use the latest checkpoint; branching from an older best checkpoint requires a fresh run. Optimizer metrics from updates lost after the resumed checkpoint are discarded. Extending the total budget uses LeRobot's scheduler behavior; it is not equivalent to having chosen that larger budget initially.

## What train and validation mean

The split uses seed 1000 and holds out 20% of independent scenes. Episodes sharing a reset seed or saved initial-state fingerprint stay together, including aliases connecting multiple seeds to one state. The split manifest retains episode IDs, original metadata, scene fingerprints, reset seeds, randomization factors, and the source manifest hash. A task with fewer than two independent scenes is rejected: the original two-trajectory diversity pilot cannot supply validation.

Statistics are aggregated from training episodes' Parquet statistics, including quantiles; validation statistics never enter normalization. Both splits and saved processors use those training statistics. No dataset metadata files are overwritten. The run saves `split.json` and `train_stats.json` for inspection.

Metrics are separated as follows:

| Output | Meaning |
| --- | --- |
| `optimizer_metrics.jsonl`, `train/optimizer_loss` | Training objective observed every 25 updates; ACT also reports L1 and KL |
| `metrics.jsonl`, `train/probe` | Inference-mode loss on a fixed episode-balanced training probe |
| `metrics.jsonl`, `val/probe` | Matching inference-mode loss on held-out scenes |
| `metrics.jsonl`, `train/rollout` | Fixed reset suite drawn from training scenes |
| `metrics.jsonl`, `val/rollout` | Fixed reset suite drawn from held-out scenes |

Loss probes use up to 1,024 frames per split, without augmentation. Partial batches are weighted by sample count, or valid action timesteps for ACT. ACT's inference-mode L1 excludes its training VAE/KL objective; SmolVLA reports flow-matching loss with fixed probe RNG seeds. Loss values should be compared within an architecture, not directly between ACT and SmolVLA.

Rollout suites use the first ten unique sorted reset seeds from each split, or all available seeds when fewer exist. Explicit reset lists run in GPU batches of up to five worlds. Each rollout has a 250-step horizon at 20 Hz. Reports contain task success, contact-valid success, contact/physics validity, truncation, penetration peaks, and individual reset metadata. One side-by-side camera video is saved per split/evaluation under `eval/<step>/<split>/task-1/videos`. Contact tolerances remain 1 mm for nonpad contacts and 3 mm for all robot contacts.

Evaluation runs synchronously between updates on the current policy. It restores policy mode, action queues, and Python/NumPy/Torch RNG state afterward. Native optimization and checkpointing remain upstream; local hooks are version-checked against LeRobot 0.6.1. Evaluation also runs at the final checkpoint even when the total budget is not a multiple of its interval.

Only the latest resumable checkpoint and best validation checkpoint are retained. `best.json` records the winner: highest contact-valid validation success, then lowest validation probe loss. At completion the latest checkpoint is the final checkpoint. Per-step reports remain under `metrics/` even after old weights are removed. Validation guides model selection; an independent test suite is still needed for final performance claims.

## Batched rendering validation

The v2 integration passed 119 tests, Ruff, PowerShell syntax checks, and scoped legacy-deletion checks. GPU tests cover scene and puzzle colors, wrist motion, unchanged physics, pre-action image alignment, padded batches, replay pixel equality, orphan-archive recovery, bounded writes, and rendering-profile rejection. Seven v1 datasets were removed; the [removal inventory](../projects/ogbench-mjwarp/validation/legacy-dataset-removal.json) records their paths. Historical raw archives and policy checkpoints remain.

Fresh collection produced four successful, contact-valid demonstrations from two resets, totaling 505 frames. ACT and SmolVLA each trained for two updates and resumed to three, with separate train/validation loss probes and batched simulator evaluations. Standalone evaluation passed for both policies, as did ACT evaluation through the native LeRobot CLI. All eleven evaluation videos decoded as two 64x32 frames. These short checks verify the pipeline, not task performance. See the [machine-readable results](../projects/ogbench-mjwarp/validation/batched-rendering.json).

On the RTX 5090, the warmed renderer produced 64 images (32 worlds, front/wrist, 32x32) in 0.79 ms per batch; initial context creation/rendering took 0.31 seconds. The warmed planner/physics batch took about 0.36 seconds, with about 2.8 GiB total device usage during the benchmark. These timings exclude host skill/contact work and archive writing. [Benchmark details](../projects/ogbench-mjwarp/validation/batched-rendering-benchmark.json) record the configuration. The fresh background v2 pipeline starts with collection and then trains ACT and SmolVLA for 20,000 updates each; consult the live status/log for progress.

## Archived v1 implementation validation

The integration was exercised on an RTX 5090 in the WSL policy environment. The smoke dataset has four successful, contact-valid demonstrations from two independent resets (20000 and 20001), split into two training and two validation episodes. Both architectures completed 26-update runs with two loss/rollout checks, then resumed to update 28. Smoke rollouts were limited to two steps to validate the interface and video pipeline; their success scores do not measure task performance.

Regression checks cover scene aliases, same-scene rejection, training-only statistics, partial-batch loss weighting, explicit reset mapping, RNG/mode/action-queue restoration, duplicate report prevention, and best/latest checkpoint retention. Machine-readable smoke results are in `projects/ogbench-mjwarp/validation/periodic-training.json`. Longer-run progress is reported separately in `long-training-status.json`; smoke validation is not evidence that the 20,000-update runs have completed.

Validation passed 113 tests, including the existing GPU simulator tests, plus Ruff and PowerShell syntax checks. All twelve smoke-evaluation videos decoded as two 64x32 frames. The default background pipeline was started on 2026-10-04; its initial phase is collection. Consult the live status/log for subsequent progress.
