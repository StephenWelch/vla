## Workspace

VLA research with LeRobot datasets, ACT/SmolVLA training, held-out evaluation, and W&B metrics. LIBERO, OGBench and OCBench live on `main`; SO-101 collection, rollout, and system identification retain their original layout on [`feature/so101`](https://github.com/StephenWelch/vla/tree/feature/so101).

| Directory | Purpose |
| --- | --- |
| `projects/ogbench-mjwarp` | GPU-parallel demonstration planning, batched rendering, export, training and evaluation |
| `projects/ocbench-mjwarp` | Native OCBench GPU demonstrations, quality audits and ACT workflow |
| `projects/libero` | Single-task ACT training and evaluation from held-out demonstration states |
| `packages/vla-tools` | YAML/Tyro configuration, checkpoints, policy loading and W&B logging |
| `configs/ogbench`, `configs/libero` | Reproducible experiment recipes |
| `docs` | Experiment reports and workflow notes |

Each simulator has its own `pyproject.toml`, `uv.lock`, and environment. OGBench uses MuJoCo 3.14; LIBERO uses 3.8.1. Do not install both into one environment. The root environment contains development tooling only.

## Setup

In Ubuntu/WSL with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/):

```bash
bash projects/ogbench-mjwarp/scripts/setup-wsl.sh
bash projects/libero/scripts/setup-wsl.sh
```

Set `OGBENCH_DATA_ROOT` and `LIBERO_DATA_ROOT` to choose storage; the corresponding `OGBENCH_ENVIRONMENT` and `LIBERO_ENVIRONMENT` variables override environment locations. LIBERO simulator assets are downloaded by `scripts/download-libero.py`; see the [LIBERO workflow](projects/libero/README.md). Docker setup remains available through `scripts/setup-ogbench.ps1` and `scripts/setup-libero.ps1`.

## Demonstration generation

Activate the OGBench environment, then run:

```bash
ogbench-mjwarp list-tasks
ogbench-mjwarp generate --env scene-v0 --task-ids 1 --episodes 2 --output outputs/scene/raw
ogbench-mjwarp export --source outputs/scene/raw --output outputs/scene/dataset --repo-id local/scene
ogbench-mjwarp view --root outputs/scene/raw --episode 0
```

Use `MUJOCO_GL=egl` for generation/evaluation and `MUJOCO_GL=glfw` for the WSLg viewer. Raw episodes retain randomization factors and contact diagnostics; exports retain provenance and use batched MJWarp camera views. See the [OGBench guide](projects/ogbench-mjwarp/README.md).

For native OCBench stacking, run `vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml` in its simulator environment. See the [OCBench guide](projects/ocbench-mjwarp/README.md) for setup and the 500-attempt ACT workflow.

## Training and evaluation

In the appropriate environment, from the repository root:

```bash
vla-ogbench-pipeline --config configs/ogbench/scene-open-act.yaml
vla-ogbench-train --config configs/ogbench/train-ogbench-act-wandb.yaml
vla-ogbench-eval --config configs/ogbench/eval-ogbench-act.yaml
vla-libero-train --config configs/libero/libero-drawer-act.yaml
vla-libero-eval-validation --experiment <run>.experiment.json
```

Typed defaults are overridden by YAML, then explicit CLI flags. Repeat `--config` to layer an ignored `configs/local/*.yaml` file over a recipe. Published paths are relative to the current directory or use OmegaConf environment interpolation. Existing flat recipe paths and root Python launchers remain compatibility aliases.

W&B records losses, periodic simulator metrics, selected videos and provenance. Datasets, model weights, optimizer states, credentials and full videos stay outside Git. See [OGBench training](docs/ogbench-long-training.md), [W&B logging](docs/ogbench-wandb.md) and [LIBERO results](docs/libero-act.md).

## Development

Run each project's tests in its own environment, including shared utility tests:

```bash
python -m pytest projects/ogbench-mjwarp/tests packages/vla-tools/tests
python -m pytest projects/libero/tests packages/vla-tools/tests
uv run ruff check packages projects scripts
```

GPU and dataset tests are marked in the OGBench suite. The [repository migration notes](docs/monorepo.md) record branch preservation, compatibility and validation.
