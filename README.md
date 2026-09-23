## Workspace

Open `vla.code-workspace` for this project and sibling `../telegrip` on `feature/data-collection`. They keep separate Git histories and Python environments. Telegrip uses the published LeRobot 0.6.1 package and an environment at `E:\vla-telegrip\venv`.

For Telegrip development here, run `.\scripts\setup-dev.ps1 -EnvironmentPath E:\vla-telegrip\venv -CachePath E:\vla-telegrip\uv-cache -Test` from `../telegrip`.

## Eval

### LIBERO

LIBERO is a Linux MuJoCo benchmark. This setup runs the [LIBERO-trained SmolVLA checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero) in Docker Desktop's WSL 2 GPU engine. The benchmark uses a simulated 7D end-effector controller, not an SO-101 arm.

Docker Desktop's disk image on this machine is at `E:\DockerDesktopWSL\disk\docker_data.vhdx`, moved through **Settings → Resources → Advanced → Disk image location**. Keep the WSL 2 backend enabled. Docker's [WSL 2 instructions](https://docs.docker.com/desktop/features/wsl/) describe the disk image setting and GPU requirements.

From PowerShell in this directory:

```powershell
.\scripts\setup-libero.ps1
.\scripts\eval-libero.ps1
```

The setup script downloads the pinned model and simulator assets into `E:\vla-libero`, builds `vla-smolvla-libero:0.6.1`, and verifies GPU visibility inside the image. The default evaluation runs one episode of Spatial task 0 with seed 1000. Find its metrics and video under `outputs/libero/smoke/`. To run the [four-suite, 400-episode protocol](https://huggingface.co/docs/lerobot/libero), use:

```powershell
.\scripts\eval-libero.ps1 -FullBenchmark
```

That run writes to `outputs/libero/full/` and may take many hours. Both scripts accept `-DataRoot` and `-Image` overrides. The image uses LeRobot 0.6.1, EGL rendering, and a pinned GPU base; model and simulator assets are mounted from E: to keep the image small. The one-episode smoke run on this machine completed successfully; its 100% score is one trial, not a benchmark estimate.

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
