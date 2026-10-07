> Historical report: the OGBench package and its commands have been removed. See the [current workspace guide](../README.md) for OCBench and LIBERO.

## Repository organization

`main` contains the LIBERO and OGBench simulation stack, with shared configuration, policy/checkpoint utilities and W&B logging in `packages/vla-tools`. Simulator dependencies resolve independently; there is no root uv workspace combining their incompatible MuJoCo versions.

`feature/so101` preserves the original hardware layout, assets, calibration/collection scripts, training helpers and system identification work. It retains the original local Git history. `main` starts from the remote's existing history. Hardware work can be checked out beside simulation work with:

```bash
git worktree add ../vla-so101 feature/so101
```

The original mixed source state is retained locally on `archive/pre-monorepo`, with a source and patch backup under ignored `outputs/monorepo-migration/`. The original `master` branch is also retained. These preservation branches are not published. Existing datasets, checkpoints, caches, calibration files and evaluation videos were not relocated or migrated.

## Compatibility

Root Python launchers delegate to installed package modules. OGBench keeps its `ogbench-mjwarp` command and PowerShell generation/viewer launchers; new training and evaluation commands use package entry points. Subprocesses use `python -m`, so they do not depend on a sibling script or a modified `sys.path`.

Recipes are grouped under `configs/ogbench` and `configs/libero`; flat paths remain aliases for older commands. Paths use environment interpolation or relative defaults. Multiple `--config` arguments merge in order, followed by CLI overrides. Use ignored `configs/local/` for host-specific storage and credentials supplied through the environment or credential files.

OGBench's dataset/action/rendering contracts and LIBERO's checkpoint processors remain unchanged. Historical experiment reports retain their original paths and source hashes as evidence; these are not installation defaults. Full experiment artifacts are excluded from Git.

## Validation

Validation on the existing WSL environments:

- OGBench and shared utilities: 157 tests passed, including GPU physics, contact, batched rendering and dataset checks; the added configuration tests passed separately.
- LIBERO and shared utilities: 17 tests passed.
- SO-101 camera alignment, dataset curation and system identification dispatch: 10 tests passed without operating hardware.
- The packaged LIBERO evaluator matched all 43 existing demonstrations (34 train, 9 validation), with no validation initial state aliases in training. The existing ACT checkpoint loaded through the shared policy loader and produced finite seven-dimensional actions.
- Ruff and both project lockfile checks passed. OGBench built as a source distribution and wheel, including its pipeline recipes.

Docker build definitions use the monorepo root as context so both projects can install the shared package. Docker images were not rebuilt during this migration. No new full training run or benchmark evaluation was started.
