> Historical report: the OGBench package and its commands have been removed. See the [current workspace guide](../README.md) for OCBench and LIBERO.

# OGBench experiment tracking

New collection → ACT → SmolVLA pipelines log to the `vla-ogbench` W&B project. Each experiment groups collection, both policies, and subsequent evaluations. Local reports and checkpoints remain the source for resume and checkpoint selection.

## Run

Scene task 1 uses `configs/ogbench/scene-open-act.yaml` with the root pipeline's `--config` flag. It collects at least 100 successful independent resets of `scene-v0`, task 1, in rounds of 32, then trains ACT from scratch for 20,000 updates. The instruction is “Open the drawer and window. Leave the cube in place.” Numeric goals remain in planner metadata and success checks. Rendered front/wrist views are 640×480 with renderer revision 2.

The Scene recipe uses one randomized trajectory per reset, an 80/20 split by reset/state identity, batch size 8, and 16-action chunks. Loss probes run every 1,000 updates; train/validation rollouts run every 2,000 updates with up to 10 episodes per split and 750 steps per episode. Latest and best validation checkpoints are retained. Joint targets, handle grasps, waypoint positions/yaw, and timing are randomized and annotated. Raw recordings live under the repository's `outputs/scene-open-act-20261005/raw`; datasets and checkpoints stay on E: to spread storage use.

A [Scene smoke run](https://wandb.ai/rlgoats/vla-ogbench/runs/vgx95p04) verified two ACT updates, separate train/validation losses, and both Scene rollout suites. Its two-step rollouts check integration only. The initial collection check had three contact-valid successes in four attempts; the failed attempt was excluded from its exported dataset.

From WSL, after logging into W&B in the isolated OGBench environment:

```bash
export OGBENCH_DATA_ROOT=/mnt/e/vla-ogbench
source "$OGBENCH_DATA_ROOT/venv/bin/activate"
MUJOCO_GL=egl vla-ogbench-pipeline --runs "$OGBENCH_DATA_ROOT/runs" --experiment cube-wandb-01 --dataset "$OGBENCH_DATA_ROOT/datasets/cube-training-v2" --background
tail -f "$OGBENCH_DATA_ROOT/runs/cube-wandb-01/pipeline.log"
```

An existing complete v2 dataset is reused; omit `--dataset` to collect into a new experiment directory. Omit `--experiment` for a unique generated name. Outputs live under `runs/<experiment>/{collect,act,smolvla}`; `pipeline-config.json` freezes the full configuration. Rerun with that file to resume the same pipeline. A stale PID after reboot does not prevent restarting.

For fresh 640×480 collection followed by ACT only, use `--policies act --size 480 640` and omit `--dataset`. `--policies act smolvla` trains both architectures. Each new experiment preserves previous datasets and runs.

Generation refills finished slots and batches host transfers. A single spawned exporter processes completed rounds while the next round generates, with at most two queued round paths. It streams H.264 without temporary PNGs and blocks on full encoder queues; frames are never intentionally dropped. Writer or encoder failure stops the pipeline and leaves partial datasets marked incomplete. Retry incomplete export in a fresh dataset directory, reusing completed raw recordings from the same implementation/configuration.

Tyro/YAML exposes `refill_slots`, `batched_cpu`, `streaming_encoding`, `overlap_export`, `encoder_queue_size` (30 frames per camera), `encoder_threads` (two per camera), and `export_queue_capacity` (two rounds). Disable these optimizations with their `--no-*` flags for comparison. Local round summaries contain rendering, host-transfer, CPU-validation, planner, archive-wait, utilization, and Torch memory metrics; `export_metrics.json` separates source waiting from active export time. W&B receives progress and final reports through the parent process.

Benchmark identical-episode export and baseline/optimized eight-scene collection with `scripts/benchmark-ogbench-pipeline.py --source RAW_ROUND --output NEW_OUTPUT`. Add `--wait-for-training TRAIN_OUTPUT` to wait for the existing training experiment to stop first. Results, contact diagnostics, and train/validation counts are saved in `benchmark.json`; Torch memory figures exclude allocations owned directly by Warp.

On the RTX 5090, direct encoding exported the same eight episodes/1,017 frames in 11.7 seconds versus 35.1 seconds with PNG staging (3.0× faster). The two-round, eight-scene pipeline took 227 seconds versus 311 seconds (27% less time). The optimized run produced 16 successes versus 15, with all 16 attempts contact-valid in both runs. These are single-run measurements with the same seeds and planner budgets; scheduling can change numerical trajectories and outcomes.

Regression checks cover 150 tests, including queue saturation, encoder/worker failures, slot reuse, RNG resets, per-episode timeouts, resume, and image alignment. Every benchmark video decoded at 640×480 with exactly one frame per action; state, action, and annotation columns matched between the identical-episode export cases. See the [validation report](../projects/ogbench-mjwarp/validation/pipeline-optimization.json) and [W&B benchmark](https://wandb.ai/rlgoats/vla-ogbench/runs/28whp3ke). The current ACT job was stopped at the user's request; its update-7,000 checkpoint remains available.

For individual policies:

```powershell
.\scripts\train-ogbench.ps1 -Config configs/ogbench/train-ogbench-act-wandb.yaml
.\scripts\train-ogbench.ps1 -Config configs/ogbench/train-ogbench-smolvla-wandb.yaml
```

The PowerShell training/evaluation launchers forward the Windows `.netrc` path when present. Credentials stay in memory and are excluded from saved configurations. YAML or Tyro flags override `wandb.enable`, `project`, `entity`, `mode`, `group`, `name`, and `tags`. Examples: `--wandb.group cube-comparison`, `--wandb.mode offline`, or `--wandb.no-enable`. The original `*-long.yaml` recipes retain disabled logging.

Enable standalone evaluation with `--wandb.enable`; its group defaults to the checkpoint's saved group. Use the matching dataset/rendering profile, as in the [training workflow](ogbench-long-training.md).

## Metrics and uploads

Renderer revision 2 fixes stale ray-tracing bounds, excludes invisible goal markers, and uses a 3.5 mm near plane for wrist close-ups. Evaluation videos show front on the left and wrist on the right, each at 640×480. The wrist camera follows the gripper, so a cube outside its field of view or behind the fingers can still be absent.

Earlier dataset images and evaluation videos are affected by the renderer bug, including `cube-act-640x480-20261005`. Their policies learned from faulty images and need fresh data and training. Collection, export, training, and evaluation enforce the rendering profile; old files are preserved. The earlier frame-count/alignment checks did not detect incorrect scene visibility. New tests compare geometry IDs against native MuJoCo across cube, scene, and puzzle tasks, multiple worlds, moved objects, and wrist close-ups.

The [corrected diagnostic video](https://wandb.ai/rlgoats/vla-ogbench/runs/suetzq5x) renders one saved demonstration with the fixes; it is not a new policy evaluation. Eight reference views at 640×480 agree with native MuJoCo on over 99.99% of geometry IDs. See the [audit report](../projects/ogbench-mjwarp/validation/rendering-audit.json).

Native optimizer metrics and periodic `train/probe`, `val/probe`, `train/rollout`, and `val/rollout` metrics share one training run. Plot training metrics against `train/update`: SDK event indices also include evaluations and are not optimizer updates. Collection reports committed episodes, frames, outcomes, planner progress, and episode tables containing randomization factors and contact diagnostics.

Small provenance, split, normalization, rendering, evaluation, and checkpoint-selection reports are uploaded along with selected existing evaluation videos. Raw demonstrations, dataset camera videos, model weights, and optimizer state are not uploaded. Local files remain available when logging is disabled.

Missing credentials or an initialization connection failure falls back to offline recording with a warning. `tracking.json` records the actual mode, stable run ID, URL when online, and offline segments. Resume retains the run ID. Sync offline segments in their recorded order using the commands printed on exit:

```text
wandb sync --append --id RUN_ID /path/to/offline-run-FIRST
wandb sync --append --id RUN_ID /path/to/offline-run-NEXT
```

## Validation and interrupted runs

Offline SDK checks exercised dataset reuse, ACT and SmolVLA training through update two then resume to three, periodic train/validation probes and simulator evaluations, and standalone ACT evaluation. These short rollouts test plumbing, not policy performance. An [online smoke run](https://wandb.ai/rlgoats/vla-ogbench/runs/l26g4uli) verified authentication and uploaded dataset reports and episode tables. Results are recorded in [wandb.json](../projects/ogbench-mjwarp/validation/wandb.json).

Validation passed 130 tests, Ruff, and PowerShell syntax checks, including offline fallback, stable resume identity, upload selection, and incomplete-checkpoint recovery.

The system restart interrupted the previous long pipeline; its local status now records the interruption. ACT completed 20,000 updates. SmolVLA's update-5,000 save is incomplete; update 4,000 is the latest validated resumable checkpoint, and its training configuration passed a resume dry run. Incomplete saves are preserved and skipped when selecting the latest checkpoint. If `experiment.json` was not flushed before interruption, resume uses the durable experiment sidecar. Historical metric import is not automatic. The long run has not been restarted.
