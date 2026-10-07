## Workspace

VLA research with LeRobot datasets, ACT/SmolVLA training, held-out evaluation, and W&B metrics. OCBench and LIBERO live on `main`; SO-101 work retains its original layout on [`feature/so101`](https://github.com/StephenWelch/vla/tree/feature/so101).

| Directory | Purpose |
| --- | --- |
| `projects/ocbench-mjwarp` | Native GPU demonstrations, batched video export, quality audits and ACT workflow |
| `projects/libero` | Single-task ACT training and evaluation from held-out demonstration states |
| `packages/vla-tools` | YAML/Tyro configuration, training integration, checkpoints and W&B logging |
| `configs/ocbench`, `configs/libero` | Reproducible experiment recipes |
| `docs` | Experiment reports and workflow notes |

Each simulator has its own `pyproject.toml`, `uv.lock`, and environment. OCBench uses MuJoCo 3.14; LIBERO uses 3.8.1. Keep their environments separate. The root environment contains development tooling only.

## Setup

In Ubuntu/WSL with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/):

```bash
sudo apt-get install ffmpeg
bash projects/ocbench-mjwarp/scripts/setup-wsl.sh
bash projects/libero/scripts/setup-wsl.sh
```

See the [OCBench workflow](projects/ocbench-mjwarp/README.md) and [LIBERO workflow](projects/libero/README.md) for storage, environments and simulator assets.

## Demonstration generation

Activate the OCBench environment, then run:

```bash
ocbench-mjwarp generate --output outputs/stack --episodes 32
ocbench-mjwarp export --source outputs/stack --output outputs/stack/datasets/successes
ocbench-mjwarp export --source outputs/stack --output outputs/stack/datasets/failures --no-successes
ocbench-mjwarp prepare-dataset --datasets outputs/stack/datasets --output outputs/stack/datasets/all
MUJOCO_GL=glfw ocbench-mjwarp view --root outputs/stack --episode 0
```

Use `MUJOCO_GL=egl` for generation/evaluation and `MUJOCO_GL=glfw` for the WSLg viewer. Raw episodes retain sampled plans and contact diagnostics; exports retain provenance and use batched MJWarp cameras with direct video encoding.

## Training and evaluation

In the appropriate environment, from the repository root:

```bash
vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml
python -m ocbench_mjwarp.compare --config configs/ocbench/act-clean-actions-320-amp.yaml --dataset outputs/stack/datasets/all --output outputs/comparison --training-root outputs/comparison/training
vla-libero-train --config configs/libero/libero-drawer-act.yaml
vla-libero-eval-validation --experiment <run>.experiment.json
```

Typed defaults are overridden by YAML, then explicit CLI flags. Repeat `--config` to layer local recipes. Pipeline/comparison settings embed the shared `training` configuration; standalone training uses those fields directly. Native LeRobot overrides are a mapping, for example `overrides: {policy.chunk_size: 25, policy.n_action_steps: 25}`.

W&B records losses, periodic simulator metrics, selected videos and provenance. Datasets, model weights, optimizer states, credentials and full videos stay outside Git. OGBench code and launchers have been removed; historical reports remain in `docs`.

## Development

Run each project's tests in its own environment, including shared utility tests:

```bash
python -m pytest projects/ocbench-mjwarp/tests packages/vla-tools/tests
python -m pytest projects/libero/tests packages/vla-tools/tests
uv run ruff check packages projects scripts
```

GPU and dataset tests are marked in the OCBench suite. See [OCBench notes](docs/ocbench-workflow.md) and [LIBERO results](docs/libero-act.md).
