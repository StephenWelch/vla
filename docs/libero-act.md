# Single-task LIBERO ACT

The pilot uses LIBERO-Goal task 0: **open the middle drawer of the cabinet**.
It trains ACT on the compact `lerobot/libero` video dataset, selecting episodes
by exact task language rather than assuming the dataset task index matches the
simulator ID. The Hub revision resolves to a commit and is recorded before training.

Run from the repository root in WSL using the isolated LIBERO environment:

```bash
export LIBERO_DATA_ROOT=/mnt/e/vla-libero
bash projects/libero/scripts/setup-wsl.sh
source "$LIBERO_DATA_ROOT/venv/bin/activate"
MUJOCO_GL=egl vla-libero-train --config configs/libero/libero-drawer-act.yaml
```

LIBERO's EGL probe needs a compiler and development libraries. On Ubuntu,
install `build-essential cmake libegl1-mesa-dev libgl1-mesa-dev libx11-dev` first.
Download simulator assets with `scripts/download-libero.py` if needed.
New data, LIBERO configuration, checkpoints and logs stay under `outputs/`.
CLI flags override YAML, for example `--steps 40000 --output outputs/libero-drawer-act-long`.
`--dry-run` validates settings without downloads or simulator imports.

The default budget is 20,000 updates, batch 8, 16-action chunks, two 256×256 RGB
cameras, and relative 7D end-effector actions. ACT uses its standard ImageNet
ResNet initialization; it does not load a pretrained ACT policy. The last 20%
of the selected task's episodes are reserved for validation. State/action
normalization uses only training episodes; images use fixed ImageNet statistics.

The compact dataset's episode metadata omits task labels and per-episode
statistics. The wrapper recovers language through frame `task_index` and
`meta/tasks.parquet`, then passes explicit train/validation IDs to native
LeRobot loaders and computes state/action statistics from training frames.
ACT training, held-out loss evaluation, simulator rollouts and policy
processors all use LeRobot 0.6.1 implementations.

Training logs loss every 100 updates, validation loss every 1,000, and ten
simulator rollouts every 2,000. After training, the latest checkpoint is tested
for 20 episodes with seed 2027, separate from seed 1000 used during training.
Only the latest resumable checkpoint is retained to limit disk usage.

This uses LIBERO's standard fixed initialization bank. Different evaluation
seeds do **not** guarantee different initial states, and these simulator tests
are not a held-out-scene or out-of-distribution benchmark. The demonstration
train/validation split is disjoint by episode, not by verified initial state.

W&B logs training and validation metrics under `train/` and `eval/`, and final
test metrics and videos under `test/`, in project `rlgoats/vla-libero`.
The run URL is saved in `outputs/libero-drawer-act/tracking.json`.
The adjacent `.experiment.json` records the dataset commit, selected episode
IDs, budgets, metadata hashes and final test results. Native test metrics are
also saved in `test/eval_info.json`, with console output in `test.log`.

## WSL run (2026-10-05)

Launched [libero-drawer-act-20261005](https://wandb.ai/rlgoats/vla-libero/runs/o3ixueoj)
using `configs/libero-drawer-act-wsl.yaml`, with 34 training demonstrations
(4,783 frames) and 9 validation demonstrations. The pinned dataset commit is
`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.

This run stores the dataset cache on the WSL filesystem at
`/home/stephen/vla-libero/cache`, and checkpoints, tracking identity and reports
under `E:\vla-libero\runs\libero-drawer-act-20261005`. Its adjacent `.train.log`
and `.pid` record the detached launch. The run completed all 20,000 updates
and the final test on 2026-10-05 at 03:42 local time. Final validation loss was
0.3397; the final periodic evaluation succeeded in 9/10 episodes, and the
separate-seed final test succeeded in 18/20 episodes (90%). These evaluations
use the standard initialization bank, with the overlap caveat described above.
The final model and saved processors are under
`checkpoints/020000/pretrained_model`, and test videos are under `test/videos/`.

Before launch, a two-update smoke test completed native validation loss
(0.7270), checkpoint save/reload, one periodic rollout and one final test
rollout, including rendered videos. Both rollouts had zero successes; the
test establishes workflow compatibility, not policy quality. Its files are
in `E:\vla-libero\runs\libero-drawer-smoke2-20261005`.

Four local contract tests and Ruff checks pass. Installed simulator versions
are `hf-libero==0.1.4`, `robosuite==1.4.0` and `mujoco==3.8.1`, with CUDA Torch
2.11.0 and LeRobot 0.6.1. LIBERO meshes resolve through a package-local assets
symlink to the existing E: assets because hf-libero ignores the YAML assets
path when resolving meshes.

## Held-out demonstration rollouts

The final ACT checkpoint succeeded from **9/9 held-out validation episode
reset states (100%)**, evaluated on 2026-10-05. Metrics, all nine videos, the
episode table and a provenance report were appended to the same W&B run under
`val/rollout/` and `val/episode_<id>`.

This evaluation uses `scripts/evaluate-libero-validation.py`. It matches the
complete actions and initial observations of all 43 selected LeRobot episodes
to `clip-rt/modified_libero_hdf5`, pinned at
`6a6659f8ac7d580fd594173a0e3abf880c843130`. The source file is
`libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5`.
Source demonstrations are disjoint across the training/validation split.

The regenerated HDF5 keeps the original simulator reset in `states[0]`, while
its first observation follows ten zero-motion settling steps. Evaluation
restores each matched reset and repeats those ten steps before ACT takes
control, following the [regeneration procedure](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py).
Restored policy-state observations differ from the validation observations by
at most 2.3842e-7. No validation reset matches a training reset within absolute
tolerance 1e-6, comparing the simulator-state vectors excluding time.

Each validation episode gets one closed-loop rollout, capped at 300 actions,
using the native LeRobot evaluator and saved policy processors. These are
held-out demonstrations and reset states for one familiar task; nine episodes
do not establish generalization to new tasks or substantially different scenes.

The report is `E:\vla-libero\runs\libero-drawer-act-20261005\validation-episodes\report.json`,
with videos in the adjacent `videos/` directory. Three additional tests cover
full-sequence matching, rejection of ambiguous matches, frame sorting and
initial-state alias detection. A prior reset-verification diagnostic is kept
under `validation-rollouts/`; it is not included in these results.

To evaluate another completed experiment, run in WSL:

```bash
python scripts/evaluate-libero-validation.py --experiment /path/to/run.experiment.json
```

The source repository/file defaults are specific to this drawer task. Override
them for other tasks. Existing evaluation output is protected from overwriting.
