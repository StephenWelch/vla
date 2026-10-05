## LIBERO

Single-task ACT training through LeRobot 0.6.1, with episode-level train/validation splits, training-only normalization, periodic loss/evaluation, and W&B logging. Held-out evaluation matches entire action sequences to source HDF5 demonstrations, verifies the initial observations, and restores the matched simulator states.

From the repository root in WSL:

```bash
export LIBERO_DATA_ROOT="$HOME/vla-libero"
bash projects/libero/scripts/setup-wsl.sh
source "$LIBERO_DATA_ROOT/venv/bin/activate"
python scripts/download-libero.py "$LIBERO_DATA_ROOT"
MUJOCO_GL=egl vla-libero-train --config configs/libero/libero-drawer-act.yaml
MUJOCO_GL=egl vla-libero-eval-validation --experiment outputs/libero-drawer-act.experiment.json
```

The downloader fetches the simulator assets and the optional LIBERO SmolVLA checkpoint. ACT trains from scratch. Set `--assets` to the downloaded `libero-assets` directory; the default recipe uses `LIBERO_DATA_ROOT`. `--dry-run` prepares and validates the training configuration without training.

The CLI accepts YAML through `--config`, with explicit flags taking precedence. Store machine-specific overrides in ignored `configs/local/`. Run output includes the resolved dataset revision, episode IDs, training configuration, metrics and checkpoint paths. Keep these outputs and downloaded data outside Git.

This environment pins MuJoCo 3.8.1 and must remain separate from OGBench's MuJoCo 3.14 environment. See the [ACT experiment report](../../docs/libero-act.md) for the drawer task, recorded results and held-out evaluation protocol.
