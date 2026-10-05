## Workspace

Open `vla.code-workspace` for this project and sibling `../telegrip` on `feature/data-collection`. They keep separate Git histories and Python environments. Telegrip uses the published LeRobot 0.6.1 package and an environment at `E:\vla-telegrip\venv`.

For Telegrip development here, run `.\scripts\setup-dev.ps1 -EnvironmentPath E:\vla-telegrip\venv -CachePath E:\vla-telegrip\uv-cache -Test` from `../telegrip`.

## Demonstration generation

[OGBench + MJWarp](projects/ogbench-mjwarp/README.md) generates manipulation demonstrations with front/wrist RGB, labeled outcomes, and randomization annotations. From this directory, use the same Docker/WSL workflow as LIBERO; data defaults to `E:\vla-ogbench`.

```powershell
.\scripts\setup-ogbench.ps1
.\scripts\run-ogbench.ps1 generate --env cube-single-v0 --task-ids 1 --episodes 8 --seed 2026 --config /configs/diverse.yaml --planner.candidates 8 --planner.horizon 8 --planner.iterations 2 --output /data/raw/diverse-cube
.\scripts\run-ogbench.ps1 export --source /data/raw/diverse-cube --output /data/datasets/diverse-cube --outcome success --require-contact-valid
.\scripts\view-ogbench.ps1 -Root E:\vla-ogbench\datasets\diverse-cube -Episode 0
```

The root launchers accept `-DataRoot`; Docker launchers also accept `-Image`. `/data` inside Docker maps to that data root, while the WSLg viewer accepts Windows paths. Simulator dependencies remain in the subproject. Exported datasets use LeRobot 0.6.1, shared with training and Telegrip.

New OGBench episodes render both views at 640×480, matching SO-101 wrist camera capture. `--size 480 640` overrides height/width; existing datasets retain their recorded resolution.

## Training

Train ACT or the local SmolVLA base checkpoint in the WSL policy environment:

```powershell
.\scripts\train-ogbench.ps1 -Config configs/train-ogbench-act-wandb.yaml
.\scripts\train-ogbench.ps1 -Config configs/train-ogbench-smolvla-wandb.yaml
```

The Tyro CLI uses YAML defaults through OmegaConf; explicit flags override them. `--dataset`, `--policy`, and `--output` accept other paths; `--batch-size 4` reduces memory use. The recipe shares the base checkpoint and model cache at `E:\vla-smolvla` with hardware training, and writes runs under `E:\vla-ogbench\runs`. The launcher maps front/wrist to camera1/camera2, binds state/action dimensions to the dataset, and saves configuration, camera mapping, native training arguments, and dataset provenance hashes in `experiment.json`. Randomization annotations stay in the dataset manifest and are excluded from policy inputs.

The lighter training script uses the same launcher with its wrist camera mapped to camera2. OGBench's native five-element actions and SO-101's six joint targets remain separate dataset contracts. Hold out entire reset seeds/scenarios when comparing VLA architectures; variants of one reset should stay in the same split. The one-step OGBench training check verifies data/model compatibility, not policy performance.

OGBench v2 datasets use inline batched MJWarp cameras; checkpoints retain their rendering profile for evaluation. Legacy datasets are unsupported. The [archived pilot report](docs/ogbench-policy-pilots.md) records earlier training and the cancelled pi0.5 attempt. `train-smolvla.py` remains compatible with existing commands.

Evaluate these checkpoints in the batched OGBench simulator with `.\scripts\eval-ogbench.ps1 -Config configs/eval-ogbench-act.yaml` or `configs/eval-ogbench-smolvla.yaml`. The [pilot report](docs/ogbench-policy-pilots.md) records closed-loop results and WSL setup. YAML/Tyro overrides control reset seeds, episode count, batch size, and horizon; evaluation saves LeRobot metrics, contact diagnostics, and rollout videos.

For longer ACT/SmolVLA runs, the [training workflow](docs/ogbench-long-training.md) collects independent cube scenes and provides periodic train/validation losses, simulator evaluations, videos, and checkpoint resume. The [W&B workflow](docs/ogbench-wandb.md) groups collection, training, and evaluation metrics with provenance and selected videos; local reports and checkpoints remain available.

For cloud training:

```
uv tool install "skypilot[lambda]" --with wandb
sky check lambda
```

## Eval

### LIBERO

For single-task ACT training with validation loss, periodic simulator evaluation, and W&B logging, see [the LIBERO ACT workflow](docs/libero-act.md) and `configs/libero-drawer-act.yaml`.

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
