## SO-101 workspace

Hardware collection, training, rollout, and system identification live on `feature/so101`. LIBERO and OGBench development lives on `main`. This branch keeps the original hardware layout and environment; install it with `uv sync`.

## Hardware

### Ports
```
uv run lerobot-find-ports
```
Left arm: `COM4`
Right arm: `COM3`

```
uv run lerobot-find-cameras
```
Left camera: `1`
Right camera: `0`

### Calibration
```
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<name>
```

### Left-arm SmolVLA rollout

Install dependencies with `uv sync`, then calibrate the left follower while supporting the arm:

```powershell
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=COM4 --robot.id=left
```

With a SmolVLA checkpoint trained for this six-joint, one-camera setup, run a five-second rollout:

```powershell
.\scripts\rollout-left-so101.ps1 -PolicyPath <checkpoint-directory> -Task 'your training task'
```

The rollout script uses camera `1` and checks the checkpoint and calibration before moving the arm. The local PushT and LIBERO checkpoints are incompatible.

The base checkpoint can be tested without moving the arm using `uv run python scripts\shadow-smolvla-base.py`. Its unadapted actions are not suitable for direct SO-101 rollout.

### Left-arm lighter demonstrations

Use Telegrip's `config.windows-left.yaml` with `--task "Grasp the green lighter by its body and lift it at least 3 cm for one second."` to record one left-wrist-camera episode at a time. Mark five lighter positions A-E and collect 50 clean successes plus 10 successful recoveries; a success with a flagged mistake is a recovery. Failures and unmarked episodes stay in the raw dataset but are excluded from training.

Episode start and stop move to home outside recording. Left X stows and disengages; press it again during the move to cut torque and keep a partial unmarked episode. Align the VR robot model with the left thumbstick to use operator-world control. With the arm disengaged, calibrate the A-E placement zones from the desktop and position each zone with the right thumbstick in VR.

Record a separate 10-20 second diagnostic episode in `E:\vla-smolvla\datasets\wrist_timing` with Telegrip's left-arm config while gently varying left wrist roll in front of a stationary textured scene. Estimate camera lag and align new recordings before curation:

```powershell
uv run python scripts\estimate-camera-lag.py --source E:\vla-smolvla\datasets\wrist_timing --episode 0 --output E:\vla-smolvla\camera-lag.json
uv run python scripts\align-lighter-dataset.py --source E:\vla-smolvla\datasets\lighter_left_raw --output E:\vla-smolvla\datasets\lighter_left_aligned --calibration E:\vla-smolvla\camera-lag.json
uv run python scripts\curate-lighter-dataset.py --source E:\vla-smolvla\datasets\lighter_left_aligned --output E:\vla-smolvla\datasets\lighter_left_curated
.\scripts\train-lighter-smolvla.ps1
uv run python scripts\validate-lighter-smolvla.py --checkpoint <checkpoint-pretrained-model-directory>
```

The curation step makes a fixed 48-episode training set and 12-episode validation set, retaining recovery and position labels in `manifest.json`. Training uses the local SmolVLA base checkpoint, wrist camera input `camera2`, and PyAV. If GPU memory is insufficient, rerun in a new output directory with `-BatchSize 4`. Inspect validation errors before any supervised hardware rollout.

### Setting motor IDs
```
uv tool install git+https://github.com/Enigma-Incorporated/servotools.git
```

### SO-101 system identification

The vendored model `assets/mjcf/so101_new_calib.xml` (TRS Onshape export) has unmeasured dynamics: actuator gains derived from a servo P=16 heuristic, stock `armature`/`damping`/`frictionloss`, and no identified link inertias. The sysid routine identifies all of them (plus actuator delay) from a multisine excitation recording. STS3215 servos have no torque telemetry, so the fit is the tracking error between measured encoders and a MuJoCo rollout driven by the recorded position commands, with the servo PD law as the actuator model.

Collect on the left arm (Windows, ~5-6 min, support the arm first):

```powershell
uv run python scripts\collect-so101-sysid.py --port COM4 --id left
uv run python scripts\collect-so101-sysid.py --dry-run --seed 42   # preview the trajectory, no motors
```

The recording lands in `outputs/so101_sysid/<stamp>/` (`data.csv` at 500 Hz: commands, positions, velocities, temperatures; `meta.json`: trajectory definition, calibration, servo PD gains).

Fit in Docker (mjbatch ships Linux wheels only; the image builds itself from `docker/sysid.Dockerfile` on first use, via `scripts/run-so101-sysid.sh` in a WSL 2 shell or directly in PowerShell):

```powershell
docker run --rm -v ${PWD}:/work -w /work so101-sysid:latest python scripts/run-so101-sysid.py --data outputs/so101_sysid/<stamp>
```

Outputs go to `outputs/so101_sysid/run-<stamp>/`: `params_x_0.yaml` / `params_x_hat.yaml` (nominal vs identified), `confidence.pkl` + per-parameter 95% intervals in `summary.txt`, identified MJCF, `report.html`, and a held-out-window RMSE comparison (nominal vs identified). The wide CIs are the honest answer to which parameters the recording actually identifies.

The fit is self-testable without hardware: `--synthetic` perturbs the sim truth (masses x1.3, CoM shifts, kp x0.7, kv x1.5, doubled frictionloss, 10 ms delay), records it through the same data path, and exits non-zero unless every injected truth value lands inside its 95% confidence interval.

```
scripts/run-so101-sysid.sh --synthetic
```
