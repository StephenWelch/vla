# SO-101 System Identification — MuJoCo sysid + mjbatch

## Context

Every SO-101 MJCF in circulation (TRS, mujoco menagerie, community forks) carries **un-identified
dynamics**: actuator gains from a "servo P=16" heuristic (kp 998.22 / kv 2.731), stock
`frictionloss`/`armature`, no measured inertial corrections, and a defined-but-never-used backlash
class. Any sim2real attempt on this arm starts from an unknown sim bias. This plan builds the missing
sysid routine for the **left SO-101 (COM4)** using the two requested frameworks:

- **`mujoco[sysid]`** (DeepMind, python/mujoco/sysid): `TimeSeries`, `Parameter`/`ParameterDict`,
  `ModelSequences`, `build_residual_fn`, `optimize`, `save_results`, `default_report` (HTML).
- **`mjbatch`** (kevinzakka): C++ thread-pool batched simulation; `Batch(model, n_sims)` with
  per-sim model fields via `expand()`/`set_const()`. Its `examples/rizon_inertia.py` is a complete
  multisine-excitation → record → batched-Levenberg–Marquardt sysid reference.

### User decisions (locked)
| Question | Answer |
|---|---|
| Data | **Include a collection script** driving the left arm |
| Parameters | **Full set**: body inertials + joint armature/damping/frictionloss + servo kp/kv + time delay |
| Base model | **TRS `so101_new_calib.xml`** (Onshape export, 13 meshes, matches this repo's calibration workflow) |
| mjbatch role | **Framework-native `custom_rollout`** (window-batched inside `optimize`), no separate LM stage |
| Runtime | Collection on **Windows native** (COM4); runner in a **Docker container on WSL2** (mjbatch
  ships Linux wheels only; `docker-desktop` WSL dist is present on this machine) |

### Physical constraint
STS3215 servos are position-controlled with **no torque telemetry**; LeRobot records at 30 fps
(too slow). Identification inputs: commanded position `c(t)` (known) + measured `q(t)`, `q̇(t)` at
500 Hz, with the servo PD law as the actuator model (residual = tracking error). The feetech
SDK supports bulk reads fast enough for 500 Hz on all 6 motors (lerobot's per-motor `read()`
wrapper is not).

## Framework API (verified against mujoco main)

- `TimeSeries(times, data)`, `TimeSeries.from_names(times, data, model, names)` — names resolve to
  sensors / `qpos` / `qvel` / `act`; `from_control_names` for ctrl.
- `create_initial_state(model, qpos, qvel, act)` → `(n_state,)`.
- `ModelSequences(name, spec, sequence_name, initial_state, control, sensordata)` — **one spec +
  many measured sequences** (used for data windows).
- `Parameter(name, nominal, min_value, max_value, modifier)`; `ParameterDict.add`;
  `body_inertia_param(spec, model, body, InertiaType.MassIpos)` (4 coords: mass + ipos);
  `apply_pgain` / `apply_dgain` for position-actuator kp/kv.
- `build_residual_fn(models_sequences=..., build_model=..., custom_rollout=..., signal_transform=...)`
  → `fn(x, params, **overrides)` → `(residuals, pred_ts_list, meas_ts_list)`.
  **`custom_rollout`** replaces the rollout step only; it receives
  `(models, datas, control_signal, initial_states, rollout_signal_mapping, rollout_state_mapping,
  ctrl_mapping, param_dicts)` where `len(models) = n_fd × n_chunks` (one model per finite-difference
  column, repeated per chunk) and `param_dicts[c]` carries column-`c`'s values.
  Must return one `SystemTrajectory(model, control, sensordata, initial_state, state)` per sim —
  see `arrays2traj` in `_src/trajectory.py` for the exact array layouts.
- **Delay**: `signal_transform.SignalTransform()` with `.delay("qpos", delay_param)` — delays
  predicted data inside the residual (resample + delay + window pipeline).
- `optimize(initial_params, residual_fn, optimizer="mujoco"|"scipy"|"scipy_parallel_fd",
  max_iters=...)` → `(opt_params, OptimizeResult)`. `scipy_parallel_fd` = scipy least_squares +
  MuJoCo finite-difference Jacobian (each parameter perturbation = one rollout → all rollouts in a
  single residual call are batchable in one `Batch`).
- `save_results(folder, models_sequences, initial_params, opt_params, opt_result, residual_fn)` →
  `params_x_0.yaml`, `params_x_hat.yaml`, `results.pkl`, `confidence.pkl` (cov + CIs), identified XMLs.
- `default_report(models_sequences, initial_params, opt_params, residual_fn, opt_result,
  title=..., save_path=..., generate_videos=...)` → HTML.
- Install: `mujoco[sysid]` extra exists on **stable** wheels (verified PyPI 3.13.0 + 3.14.0);
  mjbatch pins `mujoco==3.13.0`, cp313 manylinux wheel exists → Docker image `python:3.13-slim`.

## Approach

### 1. `scripts/collect-so101-sysid.py` (Windows, real arm)
- Args: `--port COM4 --id left --calibration left --duration 300 --amplitude 8.0 (deg)
  --fmin 0.3 --fmax 0.9 --rate 500 --command-rate 100 --out outputs/so101_sysid/<stamp>/
  --dry-run`.
- Reuse installed `lerobot` for bring-up: `SO101Follower` config → opens port, loads the existing
  `left` calibration, torque on. Then bypass its per-motor wrapper and use `scservo_sdk`
  `PacketHandler.bulkReadTxRx` for the capture loop.
- Trajectory (rizon-style, scaled for SO-101 joint limits): per-joint multisine
  `q_j(t) = home_j + A sin(2π f_j t + φ_j)`, `A` = `--amplitude` (default 8°, all SO-101 joints
  range ≥ ~100°), `f_j` spread over `[fmin, fmax]` (0.35/0.55/0.85 Hz for the 6 joints), phases
  randomized per run and stored in meta. 2 s home dwell at start/end.
- Two threads: **command** at 100 Hz (position goals, lerobot sign-magnitude encoding as used by
  the follower); **capture** at 500 Hz (`bulkReadTxRx`: position + velocity for all 6 motors per
  packet; temperature every 10th cycle). PC timestamps via `time.perf_counter()`.
- Safety: commands clamped to 60 % of the calibrated range around home; velocity cap ~30 °/s;
  temperature watch (warn ≥ 60 °C, abort ≥ 75 °C); Ctrl-C → command home pose, then disconnect;
  `--dry-run` writes the planned trajectory CSV without touching motors.
- Output: `data.csv` (`t, q_cmd×6, qpos×6, qvel×6, temp×6`) + `meta.json` (port, motor IDs,
  calibration min/max/home, amplitude, freqs, phases, rate, command rate, duration, servo model,
  timestamp). CSV columns match what the runner consumes directly.

### 2. `docker/sysid.Dockerfile` + `scripts/run-so101-sysid.sh`
- `FROM python:3.13-slim`; `pip install "mujoco[sysid]==3.13.0" mjbatch==0.1.1 numpy scipy`.
- `docker run --rm -v <repo>:/work -w /work <image> python scripts/run-so101-sysid.py ...` —
  the repo is bind-mounted from Windows, so `outputs/so101_sysid/...` written by the container
  lands directly in the Windows tree (Docker Desktop handles the mount).
- The runner imports no lerobot/torch — minimal image.

### 3. `scripts/run-so101-sysid.py` (inside the container)
- Args: `--data <collection dir> --model <mjcf> (default: vendored assets/mjcf/so101_new_calib.xml,
  fetched from TheRobotStudio once and committed) --window 4.0 --optimizer scipy_parallel_fd
  --max-iters 200 --seed 0 --out outputs/so101_sysid/<run>/ --synthetic` (synthetic = generate
  `data.csv` from a perturbed sim instead of real data).
- **Model prep** (`MjSpec.from_file`):
  - `spec.remove_visuals()`; clear `spec.con` (dynamics-only fit, no contacts).
  - `spec.opt.timestep = 1/500` (match data rate).
  - Keep the 6 position servos as actuators; set `ctrlrange` from calibration so sim clamping
    matches the real joint limits.
  - Observed signals: `qpos` + `qvel` (state vectors via `TimeSeries.from_names(names=["qpos","qvel"])`);
    control: `q_cmd` via `TimeSeries.from_control_names`.
- **Data → windows**: drop dwells (steady-state start/end); chunk into ~4 s windows; each window →
  one measured sequence (`create_initial_state` from its first sample) inside a single
  `ModelSequences`.
- **Parameters** (`ParameterDict`, 55 total):
  - Inertia: `body_inertia_param(..., InertiaType.MassIpos)` for the 6 moving bodies
    (`shoulder`, `upper_arm`, `lower_arm`, `wrist`, `wrist_roll`, `gripper`) → 6 × 4 = 24
    (mass ±10×, ipos ±0.5 m framework defaults).
  - Joints (6): `armature`, `damping`, `frictionloss` per joint (18), custom model-stamping
    modifiers (armature from stock 0.028; damping/frictionloss from model values; bounds 0–5×).
  - Servo gains (6 actuators): `kp` via `apply_pgain`, `kv` via `apply_dgain` (12), nominal from
    model (998.22 / 2.731), bounds 0.1–10×.
  - `delay` scalar (0–0.05 s, nominal 0) registered on the `SignalTransform` (`.delay("qpos", p)`).
  - Custom `build_model(params)` replaces `apply_param_modifiers`: stamps the parameter values
    directly onto model fields (`body_mass`, `body_ipos`, `jnt_armature`, `jnt_damping`,
    `jnt_frictionloss`, `actuator_gainprm`) of a **single template MjModel** (no per-column
    recompile); returns the same model for all columns. A name → (field, index) registry built at
    setup drives both the stamping and the batch field mapping.
- **mjbatch `custom_rollout`**:
  - `Batch(template, n_fd × n_chunks)`; `expand()` each per-column field touched by the registry
    (body_mass/body_ipos for 6 bodies, jnt_armature/damping/frictionloss for 6 joints,
    actuator_gainprm for 12 gains); stamp per column from `param_dicts[c]`; `set_const()`.
  - Per sim `(c, k)`: `reset()`, qpos/qvel ← `initial_states[c,k]`, then step through
    `control_signal[c,k]` (resampled to 1/500) collecting bound `qpos`/`qvel` into
    `(n_steps, nq+nv)` state arrays; build the returned `TimeSeries`/`SystemTrajectory` exactly as
    `arrays2traj` does (sensordata times from `state[:,0]`, ctrl mapping from the pipeline).
  - Thread pool across all sims (num_threads = CPU count).
- **Fit + artifacts**: `optimize(params.copy(), residual_fn, optimizer, max_iters)`;
  `save_results(run_dir, ...)`; `default_report(..., save_path=run_dir/"report.html")`;
  console table nominal → identified ± CI; RMSE (tracking error) before/after on a held-out
  window; physical plausibility printout (masses, CoMs).
- **`--synthetic` mode**: truth model = base with injected perturbations (link masses ×1.3, CoM
  shifts up to 2 cm, kp ×0.7, kv ×1.5, extra frictionloss, 10 ms delay); "collect" data with the
  exact command/read path of the real script (sim PD loop + small encoder noise, same CSV format);
  run the fit; assert every injected truth value lies inside its 95 % CI (report per-parameter
  pass/fail). This is the first end-to-end verification and needs no hardware.

### 4. Wiring
- `pyproject.toml`: collection script needs no new deps (lerobot[feetech] + numpy already there);
  runner deps live in the Dockerfile (not the Windows venv) to avoid pulling mujoco into the torch env.
- Vendor `assets/mjcf/so101_new_calib.xml` (TRS, with the backlash class still unattached — kept
  as-is; a `--backlash` follow-up flag is out of scope, noted in README).
- `README.md`: new "System identification" subsection under Hardware (how to collect, how to fit,
  where outputs land, synthetic self-test).

## Files to modify
| File | Change |
|---|---|
| `scripts/collect-so101-sysid.py` | **new** — 500 Hz multisine data collection (Windows/COM4) |
| `scripts/run-so101-sysid.py` | **new** — MuJoCo sysid fit with mjbatch custom_rollout + report |
| `scripts/run-so101-sysid.sh` | **new** — docker run wrapper (bind-mount repo, invoke runner) |
| `docker/sysid.Dockerfile` | **new** — python:3.13-slim + mujoco[sysid]==3.13.0 + mjbatch 0.1.1 |
| `assets/mjcf/so101_new_calib.xml` | **new** — vendored TRS base model |
| `pyproject.toml` | only if collection needs a direct `scservo_sdk` dep pin (lerobot already pulls it — verify) |
| `README.md` | System identification section |

## Verification
1. **Synthetic round-trip** (`--synthetic`): injected perturbations recovered within 95 % CI for
   all 55 params; report renders; exit code 0.
2. **Collection dry-run** (`--dry-run`): trajectory CSV stats sane (bounded amplitude/frequency,
   dwell segments), no motor access.
3. **Real collection**: on the supported arm, ~5 min run; inspect CSV (no gaps, q tracks c within
   a few degrees, temps < 60 °C).
4. **Real fit**: identified inertials physically plausible (masses 20–900 g range per link, CoMs
   inside links); held-out-window RMSE lower than nominal params; `save_results` artifacts +
   `report.html` present; identified XML diff-able against the base.
5. **Spot check**: replay one held-out window open-loop with identified params — predicted qpos
   closer to measured than with nominal (the sim2real gap that started this project shrinks).

## Risks / notes
- **500 Hz capture via bulkRead** is the one unproven loop; if round-trip latency can't hold 500 Hz
  on this USB link, degrade gracefully to 250 Hz (still ≫ servo bandwidth ~16 Hz) — runner reads
  `rate` from meta.json and sets `timestep` accordingly.
- **Ill-conditioning**: full parameter set (55) from single-arm tracking data may not give tight
  CIs for every parameter (gripper coupling, kp vs mass trade-off). Mitigations: rich multisine
  (3 frequencies × 6 joints), 4 s × many windows, prior-bounded ranges; the report + CIs surface
  which params are actually identified — that's an acceptable, informative outcome.
- mjbatch `expand()` coverage of `actuator_gainprm` etc. is assumed (generic MjModel field
  expansion, per the README's field table); verified at implementation — fallback: per-column
  template models (compile is the slow part, still batched across windows).
- `opt_result.jac` from `scipy_parallel_fd` feeds `calculate_intervals` (CI computation); if a
  given optimizer path lacks it, fall back to `optimizer="mujoco"` (the notebook's default) which
  is what `save_results` was demonstrated against.
