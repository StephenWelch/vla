## OGBench demonstrations

Generate manipulation attempts with [OGBench](https://github.com/seohongpark/ogbench), batched [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) physics, and sampling-based MPC. Export RGB, proprioception, native actions, and task instructions to LeRobot v3. Successes and failures are retained with outcome labels.

This subproject has its own dependencies and lockfile. Keep it separate from the LIBERO environment. Generation requires Linux and an NVIDIA GPU; Docker Desktop's WSL 2 engine and Ubuntu under WSL are supported.

The [workspace workflow](../../README.md#demonstration-generation) exposes setup, generation/export, and WSLg playback through root scripts. Fresh v2 datasets feed the shared training launcher for ACT and SmolVLA in WSL. The [training workflow](../../docs/ogbench-long-training.md) provides periodic train/validation probes and simulator evaluations. Legacy datasets and the sequential OpenGL recorder are retired; historical raw archives remain viewable.

## Setup

From PowerShell in this directory, with Docker Desktop running:

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1 list-tasks
```

The image uses the same pinned GPU base as the repository's LIBERO workflow. Data and compilation caches default to `E:\vla-ogbench`; both scripts accept `-DataRoot`, and Docker scripts accept `-Image`.

Alternatively, from Ubuntu/WSL with `uv` installed:

```bash
bash scripts/setup-wsl.sh
export MUJOCO_GL=egl
export PATH=/mnt/e/vla-ogbench/venv/bin:$PATH
ogbench-mjwarp list-tasks
```

Set `OGBENCH_DATA_ROOT` and `OGBENCH_ENVIRONMENT` to override the WSL locations. Native Linux can use `uv sync --frozen --extra dataset --extra dev` and `uv run ogbench-mjwarp doctor`. The first run compiles collision kernels and can take several minutes.

## Generate and export

```powershell
.\scripts\run.ps1 generate --env cube-single-v0 --task-ids 1 --episodes 2 --config /configs/smoke.yaml --output /data/raw/cube-smoke
.\scripts\run.ps1 export --source /data/raw/cube-smoke --output /data/datasets/cube-smoke --repo-id local/cube-smoke
.\scripts\run.ps1 inspect --root /data/datasets/cube-smoke --chunk-length 16
.\scripts\run.ps1 replay --root /data/datasets/cube-smoke --episode 0 --video /data/cube-replay.mp4
```

Use the same arguments with `ogbench-mjwarp` in Linux, replacing `/data` with your data directory. To exercise failure recording, generate a separate run with `--max-steps 2`.

`list-tasks` lists the pinned upstream cube variants, scene, and puzzle sizes and their task IDs. Generation defaults to cycling tasks 1–5. Repeating the same command resumes completed episodes; a directory cannot be reused with a different configuration. Incomplete episodes are regenerated. Counts refer to attempts, not guaranteed successes. Export defaults to all attempts; `--outcome success` or `--outcome failure` filters episodes.

Pass multiple directories to `export --source` to combine task runs into one dataset. Source episode IDs are preserved in the manifest and remapped to unique LeRobot episode indices. All sources must share image size and frame rate.

New episodes render front and wrist at 640×480, matching the SO-101 wrist camera capture size. Use `--size 480 640` (height, width), YAML `size: [480, 640]`, or `--size 32` for square diagnostic images. Export and evaluation retain the recorded dimensions; existing datasets and checkpoints keep their original resolution.

Generation refills finished episode slots and batches CPU state transfers by default. `--no-refill-slots` and `--no-batched-cpu` retain the comparison paths. Export streams H.264 directly with blocking backpressure; `--no-streaming-encoding` enables PNG staging. `--encoder-queue-size` defaults to 30 frames per camera and `--encoder-threads` to two per camera. Raw archives remain available for replay and audit.

The default planner runs eight episodes concurrently, with 128 candidates, a 16-step horizon, three cross-entropy iterations, and 10% elites. The Tyro CLI exposes every planner field through `--planner.*` flags, including `--planner.episodes` (concurrent worlds), `--planner.candidates`, `--planner.horizon`, and `--planner.iterations`. Use `--help` on any command to see its typed options. `configs/smoke.yaml` provides a smaller initial run. Lower the batch size and candidate count if GPU memory is insufficient.

Every command accepts `--config PATH`. OmegaConf loads ordinary YAML (including interpolations); defaults are overridden by YAML, then explicit CLI flags. YAML keys match command fields, with planner settings nested under `planner`. Unknown fields and invalid values fail during typed parsing.

```yaml
# configs/smoke.yaml
planner:
  episodes: 1
  candidates: 8
  horizon: 8
  iterations: 1
```

```bash
uv run ogbench-mjwarp generate --config configs/smoke.yaml --output /data/raw/cube --planner.candidates 32
```

YAML can also set `env`, `output`, `task_ids`, and other command fields. Paths are relative to the process working directory. Existing recorded planner metadata remains compatible.

## Trajectory diversity

`configs/diverse.yaml` enables feasible manipulation-order choices, cube grasp symmetries, free-space waypoint offsets, segment timing variation, and post-IK joint offsets. Interaction positions remain anchored; timing floors preserve grasp and handle-motion durations. Experimental handle half-turn grasps are optional through `--randomization.handle-grasps` and remain subject to contact rejection. All additional factors default off.

`--episodes` counts attempts. `--randomization.variants-per-reset 2` generates pairs sharing the same initial state and goal; task cycling advances once per pair. `--seed` controls environment resets, while `--randomization.seed` controls trajectory variation and defaults to `--seed`. Ordering, grasp, path, timing, oracle, CEM, and joint-target draws use separate reproducible streams, independent of batching and resume.

```powershell
.\scripts\run.ps1 generate --env cube-double-v0 --episodes 10 --config /configs/diverse.yaml --output /data/raw/diverse
.\scripts\run.ps1 diversity --source /data/raw/diverse --output /data/diversity.json
.\scripts\run.ps1 export --source /data/raw/diverse --output /data/datasets/diverse --diverse-per-task 2
.\scripts\run.ps1 ablate --env cube-double-v0 --task-id 2 --variants 4 --output /data/ablations
```

`diversity` compares successful, contact-valid trajectories sharing a reset-state fingerprint. Its relative distance combines phase-aligned joint/effector paths, skill order, and duration; it is not a calibrated quality score. Older recordings without sufficient simulator state use seed-only grouping. `export --diverse-per-task N` selects up to N successful, contact-valid recordings per environment/task using deterministic farthest-point selection. `ablate` compares baseline, each factor, and combined settings on matching resets, saving outcomes, CPU contact audits, diversity, and generation time; it skips RGB by default.

Run configuration and versioned episode `randomization` records describe enabled factors, methods, units, bounds, separate seeds, initial state, and sampled joint offsets. Each skill records eligible ordering choices, selected targets, grasp symmetry, path offsets, requested/applied durations, upstream and modified keyframes, and reference frames clipped to workspace bounds. Raw metadata and the exported `manifest.json` preserve these records, including failed attempts when exported. Legacy provenance is marked unavailable.

Per-frame `annotation.skill_id`, `annotation.phase_id`, `annotation.route_id`, `annotation.reference`, and `annotation.available` are exported as numeric LeRobot features. Skill records map phase IDs to names. References are absolute world-frame xyz, yaw, and gripper opening before CEM refinement; realized references and simulator controller targets are retained in raw/replay archives. These annotations support analysis and filtering; policy inputs remain `observation.*` and `action`. Legacy frame IDs use -1 and availability is false.

## Contact quality

Task success alone does not establish demonstration quality. Generation now rejects MPC candidates with excessive robot/environment penetration, checking every physics substep and the integrated endpoint. Contacts on finger pads are allowed; arm/gripper-link contacts use a stricter limit. Gripper links may press buttons. Defaults are 1 mm for non-pad contacts and 3 mm for all robot/environment contacts; override `--planner.max-nonpad-penetration`, `--planner.max-penetration`, and `--planner.contact-weight` through Tyro or YAML.

Execution violations terminate with `contact_violation` rather than success. Generation also checks contacts on CPU-restored observation states and before accepting a terminal success. Raw episode metadata and exported manifests include `contact_quality` with peak penetration and thresholds. Drawer and window skills allow more time to establish a grasp and translate; window approaches use greater clearance. These are heuristic checks on upstream collision proxies, not a guarantee of physical realism or visual-mesh clearance; force limits and self-collisions are not audited.

Audit raw rollouts or exported datasets:

```powershell
.\scripts\run.ps1 audit-contacts --root /data/raw/diverse-cube --episode 1 --output /data/contact-audit.json
```

The CPU audit checks recorded states, not unrecorded substep peaks. Historical raw successes have no contact-quality guarantee; retain them for inspection and generate fresh v2 demonstrations for training.

`validate` runs every registered goal and audits each recording independently on the CPU. A goal counts as covered only when a successful attempt passes both the GPU substep checks and the CPU audit. `validation.json` includes per-episode contact results and lists goals without a contact-valid success. Increase `--attempts` to check more seeds. Validation skips RGB recording by default; add `--record-images` to retain images. Image-free runs remain viewable but cannot be exported to LeRobot. `generate` records images by default.

For VLA data selection, add `--require-contact-valid` to `export` or `inspect`, or pass `require_contact_valid=True` to `load_dataset`. This excludes contact violations and legacy episodes without recorded substep evidence; combine it with `--outcome success` to select successful demonstrations.

## Planner and observations

OGBench's waypoint plans seed candidate trajectories. MPC evaluates perturbed feedback actions with MJWarp, scores reference tracking and progress toward the current skill goal, executes the best first action, and replans. The unmodified reference is always evaluated. Skills finish after release and clearance, omitting the upstream oracle's random final retreat. Cube task decomposition handles temporary placements and stacking; scene decomposition unlocks components before interaction; puzzles use a GF(2) neighbor-toggle solution followed by physical button presses. Intermediate goals allow actions such as unlocking a drawer that must end locked; episode success always uses the full task goal.

Controllers, physics, and front/wrist cameras are batched on the GPU. Task resets and skill decomposition run on the host. Each rollout has independent physics state, button state, goals, and scene lock damping. Contact/capacity overflows and invalid candidates produce labeled failures. CPU simulation supplies upstream comparisons, contact audits, and interactive playback.

Gripper commands follow the skill reference by default, which preserves closure while carrying objects. Set `gripper_noise` in the planner configuration to explore gripper commands as well.

For varied joint-angle targets after IK, use `--config /configs/diverse.yaml` or set `--planner.joint-target-noise 0.01`. Each episode samples six independent uniform offsets in `[-0.01, 0.01]` radians, held constant to avoid target jitter. CEM simulates the same offsets used during execution; joint targets respect actuator limits and gripper commands are unchanged. Episode seeds reproduce the offsets, which are saved in metadata and simulator snapshots for replay. The limit defaults to zero; contact checks still apply to randomized trajectories.

Recording finishes the current skill's release and clearance before accepting success. Reported successes are checked against upstream OGBench's task logic before recording the outcome. Dependency versions, the OGBench revision, and an integration source hash are saved with each run.

Two opaque-arm RGB views, `front` and `wrist`, are recorded at 256×256 and 20 Hz. The wrist camera is an added simulated camera, not an upstream OGBench observation. Images and the 18-element proprioceptive state precede the associated action. State order is six arm joint positions, six joint velocities, effector x/y/z, effector yaw, normalized gripper state, and gripper joint velocity. Positions use meters and angles use radians; the upstream gripper state runs from 0 (open) to 1 (closed). Debugging targets and workspace overlays are hidden. Instructions describe the task's actual goal, including metric coordinates and row-major button states.

`action` is OGBench's normalized five-element command: relative world-frame x/y/z, relative yaw, and relative gripper closure. Physical scales are 0.05 m, 0.05 m, 0.05 m, 0.3 rad, and 1.0 respectively. Positive gripper commands close the gripper. These actions describe the simulated UR5e and are not SO-101 hardware commands.

## Rollout viewer

From PowerShell, open a recorded rollout in MuJoCo through WSLg:

```powershell
.\scripts\view.ps1 -Root E:\vla-ogbench\raw\diverse-cube -Episode 0
.\scripts\view.ps1 -Root E:\vla-ogbench\raw\cube-smoke -Episode 0 --camera wrist --speed 0.5
```

Run `bash scripts/setup-wsl.sh` in Ubuntu once to install the WSL environment. The launcher accepts `-Distro` and `-DataRoot`, converts Windows paths, and selects GLFW for WSLg. It uses the WSL environment rather than the Docker image.

The WSL launcher selects Mesa's D3D12 backend and NVIDIA adapter by default to avoid CPU rendering through llvmpipe. Existing `GALLIUM_DRIVER` and `MESA_D3D12_DEFAULT_ADAPTER_NAME` settings override those defaults.

If a taskbar icon appears without a visible window and its title shows `[WARN:COPY MODE]`, retry with `-RestartWslg`. This restarts only the WSLg compositor; Linux GUI windows close, while WSL shells and Docker remain running. The launcher warns when WSLg's log reports the failed shared-memory transport.

From WSL/Linux:

```bash
bash scripts/view-wsl.sh --root /mnt/e/vla-ogbench/raw/diverse-cube --episode 0
# Or with the project environment active:
MUJOCO_GL=glfw ogbench-mjwarp view --root /mnt/e/vla-ogbench/raw/diverse-cube --episode 0
```

The viewer accepts raw rollout directories and exported datasets with replay sidecars. `--episode` selects the raw episode ID or exported episode index. Recorded simulator states are displayed directly, including terminal state and button/lock state; CUDA and LeRobot decoding are unnecessary. Drag the mouse to orbit/pan/zoom in the free camera. Use `--camera front` or `--camera wrist` for recorded camera viewpoints.

Space pauses/resumes, left/right step one frame, Home or R restarts, End shows the terminal state, +/- changes speed, and Esc closes. Playback loops by default; `--no-loop` holds the terminal state. Use `--paused` to start paused or `--seconds 10` for a timed preview. The `view` command also accepts YAML through `--config`.

## LeRobot and replay

```python
from ogbench_mjwarp.dataset import load_dataset

dataset = load_dataset("/data/datasets/cube-smoke", chunk_length=16, outcome="success")
sample = dataset[0]
# observation.images.front, observation.images.wrist, observation.state,
# action [16, 5], action_is_pad, task, and next.* outcome flags
```

The writer uses LeRobot 0.6.1's v3 dataset API and PyAV decoding. The standard dataset contains videos, state, actions, instructions, and `next.success`, `next.done`, and `next.truncated`. `next.done` marks every episode boundary, including timeouts; `next.truncated` distinguishes time limits. `manifest.json` records episode outcomes, reasons, source IDs, seeds, task IDs, and generation settings. Replay sidecars contain simulator snapshots and terminal state; they are not model inputs.

`replay` initializes from the saved state and executes the recorded actions, reporting maximum qpos drift and outcome agreement. The drift includes mixed position, angle, and quaternion coordinates. `--restore-frames` restores every recorded state before stepping, which helps inspect contact-sensitive trajectories. Exact long-horizon replay is not guaranteed across different simulator or driver versions. Uploading to the Hub is an explicit separate operation through LeRobot.

## Policy evaluation

`ogbench_mjwarp.lerobot_env` registers an `ogbench` environment with LeRobot. Its vector environment uses the same batched MJWarp physics, controller, task resets, state layout, and front/wrist cameras as generation. Finished worlds freeze until the next rollout. Success requires reaching the goal with valid physics and contacts throughout the episode; reports also retain raw task completion and penetration peaks.

From the workspace root, use `scripts/eval-ogbench.ps1 -Config configs/ogbench/eval-ogbench-act.yaml` or `configs/ogbench/eval-ogbench-smolvla.yaml`. The isolated WSL environment needs the project's `evaluation` extra. See the [policy report](../../docs/ogbench-policy-pilots.md) for setup, native LeRobot CLI usage, results, and saved videos.

## Validation

```bash
uv run --extra dataset --extra dev pytest
uv run ogbench-mjwarp benchmark --planner.episodes 1 --planner.candidates 8 --planner.horizon 8 --planner.iterations 1
uv run ogbench-mjwarp validate --output /data/validation
```

Tests cover puzzle solutions, upstream controller agreement, candidate isolation, snapshots, button transitions, scene damping, and LeRobot export/loading. `doctor --env scene-v0` and `doctor --env puzzle-3x3-v0` check other families. Benchmark output separates warmed-up planning from rendering and reports GPU memory and physics validity. Inspect `summary.json` for generation outcomes; throughput does not establish demonstration quality.

`validate` records attempts for every predefined task and writes `validation.json`. It exits nonzero if any task lacks a successful episode. Use `--envs`, `--attempts`, and `--config` to select coverage and planner budget; the default validation budget is five concurrent episodes, eight candidates, an eight-step horizon, and one optimization iteration.

See [local validation results](VALIDATION.md) for tested coverage, replay checks, and the generated example dataset.
