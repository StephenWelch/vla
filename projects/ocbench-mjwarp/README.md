## OCBench workflow

Native GPU demonstrations for `block-double-task2-v0` (stack anywhere), deferred batched cameras, LeRobot export and ACT training/evaluation. Native termination and randomization are preserved. Physical audits filter exports; released stable-stack status is a separate diagnostic.

```bash
sudo apt-get install ffmpeg
bash projects/ocbench-mjwarp/scripts/setup-wsl.sh
source ~/.venvs/vla-ocbench/bin/activate
export MUJOCO_GL=egl
vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml
```

The recipe retains 500 attempts and trains ACT on physically audited native successes. Native failures passing the same audits are exported separately. Every fifth attempt is assigned to validation before filtering. W&B logs to `vla-ocbench`; use `VLA_WANDB_NETRC_PATH` for an existing Windows netrc from WSL.

```bash
ocbench-mjwarp generate --output outputs/ocbench --episodes 32
ocbench-mjwarp export --source outputs/ocbench --output outputs/ocbench/dataset
MUJOCO_GL=glfw ocbench-mjwarp view --root outputs/ocbench --episode 0
ocbench-mjwarp replay-check --source outputs/ocbench --output outputs/replay-check --episodes 0 2 3
```

Actions are normalized joint/gripper deltas at 50 Hz. OGBench absolute-target checkpoints are incompatible. Front/wrist images default to 640×480; the policy state contains only proprioception. Raw archives retain N+1 simulator states, native outcomes, substep robot/environment penetration peaks, sampled plans and retry history. These numerical checks do not certify visual mesh or continuous-time collision freedom.

Export defaults to async NVENC with four episodes per batch, four timestamps per render, exact uint8 image statistics, 1 MiB packet buffers and GOP 2. Exports reuse render contexts, group episodes by length and overlap one batch commit with rendering. Videos go directly into LeRobot with bounded buffers and no decode/re-encode pass. Source IDs and splits are preserved; new LeRobot episode indices follow length order. Use `--encoder-backend cpu` for the software fallback. Standalone `render` is available for viewing. Generation and completed exports are reusable with the same configuration. Exports checkpoint after each batch and resume compatible partial datasets. A process watchdog restarts stalled GPU workers; older partial exports without checkpoints require a fresh output. Datasets, checkpoints and full videos remain outside Git. See the [planner comparison](../../docs/ocbench-planner-comparison.md).

The setup script installs OCBench as a regular package from a pinned upstream commit, including its simulator assets; no separate OCBench checkout is needed. It also installs the `gpu-video` extra required by the default exporter. Benchmarks and tuning options are recorded in the [workflow guide](../../docs/ocbench-workflow.md#temporal-rendering-and-packet-buffering).

Training and loss validation use TorchCodec CPU video decoding by default. Override `video_backend: pyav` in YAML (or `--video-backend pyav`) for the fallback. Policy resizing remains separate from full-resolution video decoding.

Prepare an audited combined dataset once, then reuse it across comparisons:

```bash
ocbench-mjwarp prepare-dataset --datasets outputs/stack/datasets --output outputs/stack/datasets/all
python -m ocbench_mjwarp.compare --config configs/ocbench/act-clean-actions-320-amp.yaml --dataset outputs/stack/datasets/all --output outputs/comparison --training-root outputs/comparison/training
python -m ocbench_mjwarp.benchmark_render --source outputs/stack --output outputs/export-benchmark --limit 2
```

Existing combined comparison datasets can be passed directly with `--dataset`. Comparisons retain the same unfiltered holdout and select training IDs without copying videos. Clean successes exclude pick retries and mistakes; missing annotations are rejected.

Pipeline and comparison recipes embed `training`; standalone `vla-ocbench-train` uses the same fields at the top level. Set `action_mode` to `delta`, `absolute_gripper` (delta arm), or `absolute` (joint-angle targets). Native overrides are a mapping, e.g. `overrides: {policy.chunk_size: 25, policy.n_action_steps: 25}` for open-loop 25-step chunks. Checkpoints from the image-normalization change onward remain resumable from their saved training specs. Older checkpoints and removed OGBench commands are unsupported.

Recorded-frame checks use `ocbench-mjwarp evaluate-recorded --dataset <dataset> --checkpoint <pretrained_model> --output <report.json>` and compare targets in the checkpoint's action representation. These are prediction checks, not simulator success evaluations.

## Browse episodes with Rerun

Open the whole LeRobot dataset to switch episodes in Rerun's recordings list without restarting. Run the newer viewer in an isolated environment; it does not need LeRobot's older `viz` extra:

```bash
uvx --from 'rerun-sdk==0.38.1' rerun /path/to/dataset
```

For graphics acceleration in a Windows browser while the data stays in WSL:

```bash
uvx --from 'rerun-sdk==0.38.1' rerun --web-viewer --renderer webgpu /path/to/dataset
```

Open the URL printed by Rerun in Windows Chrome or Edge (default web port 9090). Enable browser graphics acceleration and check `chrome://gpu` or `edge://gpu` for hardware-accelerated WebGPU. See [dataset browsing and WSL graphics](../../docs/ocbench-workflow.md#dataset-browsing-and-wsl-graphics) for the current dataset path, split selection and native WSLg troubleshooting.

## Single-trajectory diagnostic

`vla-ocbench-train --config configs/ocbench/act-overfit-episode001.yaml` fits one clean training episode from scratch. Explicit `overfit: true` requires exactly one selected episode and `validation_fraction: 0`. Statistics, loss probes and reset-seed rollouts use only that episode; reports contain training metrics and no validation scores. The recipe uses absolute actions, 25-step open-loop chunks, 320×240 images, BF16 AMP and 5,000 updates. Override `--output` for a fresh run.

For `lerobot-dataset-viz --mode distant`, add `--display-compressed-images`: the default 1 GiB Rerun server buffer can evict early uncompressed camera frames while action curves remain visible. See [viewer instructions](../../docs/ocbench-workflow.md#dataset-browsing-and-wsl-graphics) for the full command. This affects visualization only.

Training defaults to native relative arm and gripper actions (`action_mode: delta`). ACT experiments can opt into `per_timestep_normalization: true` together with `percentile_normalization: true` to fit action bounds separately at each chunk offset. Use this project's training/evaluation entrypoints to restore the chunk-decoding adapter when loading these checkpoints. The batch-32, 5k-update recipes are `configs/ocbench/act-overfit-per-timestep-{absolute,relative}.yaml`; see the workflow notes for details.

Evaluation streams videos through a bounded queue. The opt-in ACT evaluator uses GPU actions and temporal rendering: `vla-ocbench-eval --checkpoint <pretrained_model> --dataset <dataset> --output <output> --rollout-backend chunked --render-batch-frames 4`. `auto` retains the LeRobot reference until the serial parity/throughput sweep passes. See [evaluation batching](../../docs/ocbench-workflow.md#batched-act-evaluation) for configuration and benchmarking.

Import the official OCBench stacking release with `vla-ocbench-pipeline --config configs/ocbench/import-stack.yaml`. This recipe keeps 100 train and 20 validation episodes of all outcomes, renders through the shared exporter, and leaves training disabled. Official splits and source provenance are preserved; imported successes have unknown contact audits and regrasp counts. See [Hub import](../../docs/ocbench-workflow.md#official-ocbench-hub-import). Test direct 320×240 rendering with `configs/ocbench/import-stack-320.yaml`; evaluation uses the dataset’s recorded resolution. Run `ocbench-mjwarp benchmark-export --config configs/ocbench/benchmark-export.yaml` for the serial export comparison.
