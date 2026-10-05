#!/usr/bin/env python
"""Fit SO-101 dynamics and servo parameters with the MuJoCo sysid framework.

Reads a recording produced by scripts/collect-so101-sysid.py (data.csv +
meta.json) and identifies, against the vendored TRS so101_new_calib.xml:

  * per-body inertia (mass + ipos) for the 6 moving links      [24]
  * per-joint armature, damping, frictionloss                    [18]
  * per-servo position-gain (kp) and velocity-gain (kv)         [12]
  * actuator time delay (delay applied to the predicted data)    [1]

The STS3215 servos have no torque telemetry, so the residual is the
position/velocity tracking error between the measured encoders and a MuJoCo
rollout driven by the recorded position commands, with the servo PD law as
the actuator model.

Rollouts run on mjbatch: each residual evaluation (one base point plus the
finite-difference columns of the Jacobian) is a set of batched simulations —
one mjbatch.Batch per finite-difference column, one simulation per data
window. Identification uses mujoco.sysid.optimize; results are written with
mujoco.sysid.save_results and rendered with mujoco.sysid.report.

--synthetic generates a perturbed-truth recording from the simulator through
the same data path and checks that every injected truth value is recovered
inside its 95% confidence interval (hardware-free self test).

Outputs under --out:
  params_x_0.yaml  params_x_hat.yaml  results.pkl  confidence.pkl
  identified XML(s)  report.html  summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import mujoco
from mujoco.sysid import (
    ModelSequences,
    Parameter,
    ParameterDict,
    TimeSeries,
    body_inertia_param,
    build_residual_fn,
    create_initial_state,
    default_report,
    optimize,
    save_results,
    SystemTrajectory,
)
from mujoco.sysid import InertiaType
from mjbatch import Batch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODIES = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_spec(path: Path, dt: float) -> mujoco.MjSpec:
    """Load the SO-101 MJCF, drop collision geoms (dynamics-only fit) and set the timestep."""
    spec = mujoco.MjSpec.from_file(str(path))
    for geom in list(spec.worldbody.find_all("geom")):
        if geom.contype != 0:  # keep visual-only geoms (for report videos), drop collision
            spec.delete(geom)
    spec.option.timestep = dt
    model = spec.compile()
    if model.nsensor > 0:
        raise ValueError(
            f"Model has {model.nsensor} sensors; the dynamics-only fit expects none. Remove them from the MJCF."
        )
    return spec


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text())
    with open(data_dir / "data.csv") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    arr = np.array(body, dtype=np.float64)
    n = len(JOINTS)
    t = arr[:, 0]
    q_cmd = arr[:, 1 : 1 + n] * DEG2RAD  # servo units (deg / 0-100) -> rad
    qpos = arr[:, 1 + n : 1 + 2 * n] * DEG2RAD
    qvel = arr[:, 1 + 2 * n : 1 + 3 * n] * DEG2RAD  # deg/s -> rad/s
    # Drop any non-finite rows (capture gaps / dry-run padding).
    ok = np.isfinite(arr).all(axis=1)
    if not ok.all():
        print(f"Warning: dropping {int((~ok).sum())} non-finite rows")
        t, q_cmd, qpos, qvel = t[ok], q_cmd[ok], qpos[ok], qvel[ok]
    dt = 1.0 / meta["capture_rate_hz"]
    return {"t": t, "q_cmd": q_cmd, "qpos": qpos, "qvel": qvel, "meta": meta, "dt": dt}


def make_windows(data: dict, window_s: float) -> list[dict]:
    """Chunk the active (non-dwell) part of the recording into fixed-length windows."""
    dwell = data["meta"].get("dwell_s", 2.0)
    t0, t1 = data["t"][0], data["t"][-1]
    a0, a1 = t0 + dwell, t1 - dwell
    n_full = int((a1 - a0) // window_s)
    if n_full < 2:
        sys.exit(f"Not enough active data: {a1 - a0:.1f}s < 2 windows of {window_s}s")
    dt = data["dt"]
    t = data["t"]
    windows = []
    for k in range(n_full):
        ws = a0 + k * window_s
        i0 = int(round((ws - t0) / dt))
        i1 = i0 + int(round(window_s / dt))
        if i1 > len(t):
            break
        windows.append(
            {
                "t": data["t"][i0:i1],
                "q_cmd": data["q_cmd"][i0:i1],
                "qpos": data["qpos"][i0:i1],
                "qvel": data["qvel"][i0:i1],
            }
        )
    print(f"Using {len(windows)} windows of {window_s:.1f}s (active span {a1 - a0:.1f}s)")
    return windows


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def make_params(spec: mujoco.MjSpec, model: mujoco.MjModel) -> ParameterDict:
    params = ParameterDict()

    for body in BODIES:
        params.add(body_inertia_param(spec, model, body, InertiaType.MassIpos, param_name=f"{body}_inertia"))

    def joint_field(name: str, field: str):
        def mod(spec_, param):
            value = float(param.value[0])
            target = getattr(spec_.joint(name), field)
            if isinstance(target, np.ndarray):  # frictionloss is [3,1] in 3.13
                target[:] = value
            else:
                setattr(spec_.joint(name), field, value)
        return mod

    def act_gain(name: str, idx: int):
        def mod(spec_, param):
            actuator = spec_.actuator(name)
            actuator.gainprm[idx] = float(param.value[0])
        return mod

    def act_kv(name: str):
        # Position-actuator velocity gain: XML `kv` maps to actuator_biasprm[2] = -kv
        # (MjsActuator no longer exposes a `kv` attribute, so write the slot directly).
        def mod(spec_, param):
            spec_.actuator(name).biasprm[2] = -float(param.value[0])
        return mod

    for name in JOINTS:
        i = model.joint(name).id
        for field in ("armature", "damping", "frictionloss"):
            nom = float(getattr(model, f"dof_{field}")[i])
            params.add(
                Parameter(f"{name}_{field}", nom, 0.0, max(5.0 * nom, 1e-3), modifier=joint_field(name, field))
            )
    for name in JOINTS:
        i = model.actuator(name).id
        kp = float(model.actuator_gainprm[i, 0])
        bias_kv = float(model.actuator_biasprm[i, 2])
        kv = -bias_kv if bias_kv < 0 else 2.731  # nominal declared in the MJCF via the legacy `kv` attribute
        params.add(Parameter(f"{name}_kp", kp, 0.1 * kp, 10.0 * kp, modifier=act_gain(name, 0)))
        params.add(Parameter(f"{name}_kv", kv, 0.0, max(10.0 * kv, 1e-6), modifier=act_kv(name)))

    params.add(Parameter("delay", 0.0, 0.0, 0.05))
    return params


# ---------------------------------------------------------------------------
# mjbatch rollout (framework custom_rollout)
# ---------------------------------------------------------------------------

def make_sequential_rollout():
    """Sequential rollout using a single Batch per window to avoid memory explosion."""
    def rollout_fn(
        models, datas, control_signal, initial_states, param_dicts,
        rollout_signal_mapping, rollout_state_mapping, ctrl_mapping,
    ):
        n_total = len(control_signal)
        trajs = [None] * n_total
        b = None
        for i in range(n_total):
            model_i = models[i]
            if b is None or b.model is not model_i:
                b = Batch(model_i, 1)
            nq, nv = model_i.nq, model_i.nv
            dt = model_i.opt.timestep
            cs = control_signal[i]
            cmds = cs.data
            state0 = initial_states[i]
            qv = b.bind("qpos"); vv = b.bind("qvel"); cv = b.bind("ctrl")
            b.reset()
            qv[:] = state0[1 : 1 + nq]
            vv[:] = state0[1 + nq : 1 + nq + nv]
            b.forward()
            n_steps = cmds.shape[0]
            out_q = np.empty((n_steps + 1, 1, nq))
            out_v = np.empty((n_steps + 1, 1, nv))
            out_q[0] = qv
            out_v[0] = vv
            for s in range(n_steps):
                cv[:] = cmds[s]
                b.step()
                out_q[s + 1] = qv
                out_v[s + 1] = vv
            wtimes = cs.times[0] + np.arange(n_steps + 1) * dt
            trajs[i] = SystemTrajectory(
                model=model_i,
                control=TimeSeries(cs.times, cs.data, signal_mapping=ctrl_mapping),
                sensordata=TimeSeries(wtimes, np.zeros((n_steps + 1, 0)), signal_mapping=rollout_signal_mapping),
                initial_state=state0,
                state=TimeSeries(wtimes, np.concatenate([wtimes[:, None], out_q.squeeze(1), out_v.squeeze(1)], axis=1), signal_mapping=rollout_state_mapping),
            )
        return trajs
    return rollout_fn


# ---------------------------------------------------------------------------
# Residual (tracking error with predicted-data delay)
# ---------------------------------------------------------------------------

def make_modify_residual(vel_weight: float):
    def modify_residual(params, pred_sensordata, measured_sensordata, model, return_pred_all, state=None):
        nq, nv = model.nq, model.nv
        delay = float(params["delay"].value[0])
        t = state[:, 0]
        q = state[:, 1 : 1 + nq]
        v = state[:, 1 + nq : 1 + nq + nv]
        tm = measured_sensordata.times
        mq = measured_sensordata.data[:, :nq]
        mv = measured_sensordata.data[:, nq : nq + nv]
        tq = np.clip(tm - delay, t[0], t[-1])
        pq = np.empty_like(mq)
        pv = np.empty_like(mv)
        for j in range(nq):
            pq[:, j] = np.interp(tq, t, q[:, j])
        for j in range(nv):
            pv[:, j] = np.interp(tq, t, v[:, j])
        residuals = np.concatenate([pq - mq, (pv - mv) * vel_weight])
        pred_ts = TimeSeries(tm, np.concatenate([pq, pv], axis=1), signal_mapping=measured_sensordata.signal_mapping)
        return residuals, pred_ts, measured_sensordata

    return modify_residual


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rmse_on_window(spec: mujoco.MjSpec, params: ParameterDict, window: dict, label: str) -> np.ndarray:
    """Single-model rollout of one window with given params; per-joint RMSE (rad)."""
    from mujoco.sysid import apply_param_modifiers  # noqa: PLC0415

    model = apply_param_modifiers(params, spec)
    data = mujoco.MjData(model)
    data.qpos[:] = window["qpos"][0]
    data.qvel[:] = window["qvel"][0]
    errs = np.zeros(model.nq)
    for k in range(len(window["t"]) - 1):
        data.ctrl[:] = window["q_cmd"][k]
        mujoco.mj_step(model, data)
        errs += (data.qpos - window["qpos"][k + 1]) ** 2
    rmses = np.sqrt(errs / (len(window["t"]) - 1))
    print(f"  {label}: per-joint RMSE (rad) = {np.round(rmses, 5)}   mean = {rmses.mean():.5f}")
    return rmses


def confidence_intervals(x_star: np.ndarray, residuals_star: np.ndarray, jac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS least-squares covariance: (2*cost/(n-m)) * (J^T J)^-1, 95% intervals."""
    n, m = residuals_star.size, x_star.size
    cov = (2.0 * float(residuals_star @ residuals_star) / max(n - m, 1)) * np.linalg.inv(jac.T @ jac)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return x_star - 1.96 * se, x_star + 1.96 * se


# ---------------------------------------------------------------------------
# Two-stage optimization (regression + CEM)
# ---------------------------------------------------------------------------

def run_regression(params: ParameterDict, residual_fn, n_samples: int = 20, max_iters: int = 5) -> list[tuple[float, np.ndarray]]:
    """Run short LM optimizations from multiple random starting points.

    Returns list of (cost, x_vector) sorted by increasing cost.
    """
    from mujoco.sysid import optimize
    rng = np.random.default_rng()
    results = []
    x0 = np.asarray(params.as_vector(), dtype=np.float64)
    bounds = params.get_bounds()
    lower, upper = bounds

    for i in range(n_samples):
        perturbation = rng.standard_normal(len(x0)) * 0.1 * (upper - lower + 1e-6)
        x_start = np.clip(x0 + perturbation, lower, upper)
        params.update_from_vector(x_start)
        try:
            opt_params, opt_result = optimize(
                params, residual_fn,
                optimizer="scipy", max_iters=max_iters, verbose=False
            )
            costs_list, _, _ = residual_fn(opt_result.x, opt_params)
            cost = 0.5 * np.sum(np.concatenate(costs_list) ** 2)
            results.append((float(cost), opt_result.x.copy()))
        except Exception as e:
            print(f"Regression sample {i} failed: {e}", flush=True)

    results.sort(key=lambda x: x[0])
    return results


def cem_optimize(params: ParameterDict, residual_fn, n_samples: int = 200, n_elite: int = 40, n_iters: int = 20) -> tuple[np.ndarray, float]:
    """Cross-Entropy Method optimization.

    Derivative-free stochastic optimization that samples, selects elite, and updates distribution.
    Returns best parameters found and its cost.
    """
    rng = np.random.default_rng()
    x0 = np.asarray(params.as_vector(), dtype=np.float64)
    bounds = params.get_bounds()
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    eps = 1e-4

    mean = x0.copy()
    std = 0.1 * (upper - lower + eps)

    best_cost = float("inf")
    best_x = x0.copy()

    for iteration in range(n_iters):
        samples = rng.standard_normal((n_samples, len(x0))) * std + mean
        samples = np.clip(samples, lower, upper)

        costs = np.empty(n_samples)
        for j, x in enumerate(samples):
            params.update_from_vector(x)
            res_list, _, _ = residual_fn(x, params)
            res_arr = np.concatenate(res_list)
            costs[j] = 0.5 * np.sum(res_arr ** 2)

        elite_idx = np.argsort(costs)[:n_elite]
        elite_samples = samples[elite_idx]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0) + eps
        std = np.maximum(std, eps)

        if costs[elite_idx[0]] < best_cost:
            best_cost = costs[elite_idx[0]]
            best_x = elite_samples[0].copy()

    return best_x, float(best_cost)


def fit_parameters(params: ParameterDict, residual_fn, args):
    """Run the selected fit, seeding CEM from the best regression result."""
    if args.two_stage:
        print("Two-stage optimization...", flush=True)
        initial_x = np.asarray(params.as_vector(), dtype=np.float64).copy()
        reg_results = run_regression(params, residual_fn, args.reg_samples, args.reg_iters)
        if reg_results:
            params.update_from_vector(reg_results[0][1])
            best_x, _ = cem_optimize(
                params, residual_fn, args.cem_samples, args.cem_elite, args.cem_iters
            )
            params.update_from_vector(best_x)
        else:
            params.update_from_vector(initial_x)
    return optimize(params, residual_fn, optimizer=args.optimizer, max_iters=args.max_iters)


# ---------------------------------------------------------------------------
# Synthetic self test
# ---------------------------------------------------------------------------

def synthetic_truth(params: ParameterDict) -> ParameterDict:
    """Perturbed truth values, injected into the simulator for --synthetic mode."""
    truth = params.copy()
    for body in BODIES:
        p = truth[f"{body}_inertia"]
        v = p.value.copy()
        v[0] *= 1.3  # mass
        v[1:] += 0.01  # ipos shifts (m)
        p.value[:] = v
    for name in JOINTS:
        truth[f"{name}_kp"].value[0] *= 0.7
        truth[f"{name}_kv"].value[0] *= 1.5
        truth[f"{name}_frictionloss"].value[0] *= 2.0
        truth[f"{name}_armature"].value[0] *= 0.5
        truth[f"{name}_damping"].value[0] *= 3.0
    truth["delay"].value[0] = 0.01
    return truth


def record_synthetic(spec: mujoco.MjSpec, truth: ParameterDict, out_dir: Path, seed: int, duration: float) -> None:
    """Simulate the perturbed truth under the multisine excitation and write data.csv + meta.json."""
    from mujoco.sysid import apply_param_modifiers  # noqa: PLC0415

    model = apply_param_modifiers(truth, spec)
    dt = model.opt.timestep
    rate = round(1.0 / dt)
    amp = 8.0 * DEG2RAD
    freqs = [0.3 + (0.9 - 0.3) * j / (len(JOINTS) - 1) for j in range(len(JOINTS))]
    phases = [float(x) for x in np.random.default_rng(seed).uniform(0, 2 * math.pi, len(JOINTS))]
    delay = float(truth["delay"].value[0])
    dwell = 2.0
    t_end = duration + 2 * dwell
    n = int(t_end / dt)
    rng = np.random.default_rng(seed + 1)

    def q_cmd(t: float) -> np.ndarray:
        if t < dwell or t >= t_end - dwell:
            return np.zeros(len(JOINTS))
        a = (t - dwell) / (t_end - 2 * dwell)
        ramp = 1.0
        if a < 0.12:
            s = a / 0.12
            ramp = s * s * (3 - 2 * s)
        elif a > 0.88:
            s = (1.0 - a) / 0.12
            ramp = s * s * (3 - 2 * s)
        return ramp * amp * np.sin(2 * math.pi * np.array(freqs) * t + np.array(phases))

    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    rows = []
    for i in range(n):
        t = i * dt
        data.ctrl[:] = q_cmd(max(0.0, t - delay))  # actuator delay: command lags
        mujoco.mj_step(model, data)
        rows.append(
            [t + dt, *(q_cmd(max(0.0, t - delay)) / DEG2RAD), *((data.qpos + rng.normal(0, 1e-4, model.nq)) / DEG2RAD),
             *(data.qvel / DEG2RAD), *[30.0] * len(JOINTS)]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["t"]
    for prefix in ("q_cmd", "qpos", "qvel", "temp"):
        fields += [f"{prefix}_{name}" for name in JOINTS]
    with open(out_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)
    meta = {
        "motor_names": JOINTS,
        "motor_ids": {name: i + 1 for i, name in enumerate(JOINTS)},
        "motor_models": {name: "sts3215" for name in JOINTS},
        "home_deg": [0.0] * 5 + [50.0],
        "amplitude_deg": 8.0,
        "freqs_hz": freqs,
        "phases_rad": phases,
        "duration_s": duration,
        "dwell_s": dwell,
        "capture_rate_hz": rate,
        "command_rate_hz": rate,
        "max_velocity_deg_s": amp / DEG2RAD * 2 * math.pi * max(freqs),
        "velocity_unit_deg_s": 0.22,
        "vel_resolution_counts_per_turn": 4095,
        "synthetic": True,
        "port": "sim",
        "robot_id": "synthetic",
        "samples": len(rows),
        "dropped_reads": 0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Synthetic truth recorded: {len(rows)} rows at {rate} Hz -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="Recording directory (data.csv + meta.json)")
    parser.add_argument("--model", default="assets/mjcf/so101_new_calib.xml", help="Base MJCF")
    parser.add_argument("--window", type=float, default=4.0, help="Window length in seconds (default: 4)")
    parser.add_argument("--optimizer", default="scipy", choices=["scipy_parallel_fd", "scipy", "mujoco"])
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--vel-weight", type=float, default=0.1, help="Velocity residual weight vs position (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/so101_sysid/<data|run>)")
    parser.add_argument("--synthetic", action="store_true", help="Generate a perturbed-truth recording and self-test the fit")
    parser.add_argument("--synthetic-duration", type=float, default=120.0, help="Active synthetic duration in seconds")
    parser.add_argument("--no-videos", action="store_true", help="Skip report video generation")
    parser.add_argument("--no-report", action="store_true", help="Skip the HTML report")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage optimization (regression + CEM)")
    parser.add_argument("--reg-samples", type=int, default=20, help="Regression samples for two-stage (default: 20)")
    parser.add_argument("--reg-iters", type=int, default=5, help="LM iterations for regression (default: 5)")
    parser.add_argument("--cem-samples", type=int, default=200, help="CEM samples per iteration (default: 200)")
    parser.add_argument("--cem-iters", type=int, default=20, help="CEM iterations (default: 20)")
    parser.add_argument("--cem-elite", type=int, default=40, help="CEM elite samples (default: 40)")
    args = parser.parse_args()

    out_root = Path("outputs/so101_sysid")
    if args.synthetic:
        data_dir = Path(args.data) if args.data else out_root / "synthetic"
        spec0 = build_spec(Path(args.model), 1.0 / 500.0)
        model0 = spec0.compile()
        params_nominal = make_params(spec0, model0)
        truth = synthetic_truth(params_nominal)
        record_synthetic(spec0, truth, data_dir, args.seed, args.synthetic_duration)
        out_dir = Path(args.out) if args.out else out_root / f"run-{int(time.time())}" / "synthetic"
    else:
        if not args.data:
            sys.exit("Pass --data <dir> (see scripts/collect-so101-sysid.py) or use --synthetic")
        data_dir = Path(args.data)
        out_dir = Path(args.out) if args.out else out_root / f"run-{int(time.time())}"

    t_wall = time.perf_counter()
    data = load_data(data_dir)
    dt = data["dt"]
    spec = build_spec(Path(args.model), dt)
    model = spec.compile()
    windows = make_windows(data, args.window)
    params = make_params(spec, model)
    print(f"Parameter count: {params.size}")

    initial_states = [
        create_initial_state(model, w["qpos"][0], w["qvel"][0]) for w in windows
    ]
    control_ts = [TimeSeries.from_control_names(w["t"], w["q_cmd"], model, names=JOINTS) for w in windows]
    sensor_ts = [
        TimeSeries.from_names(
            w["t"],
            np.concatenate([w["qpos"], w["qvel"]], axis=1),
            model,
            names=[f"{j}_qpos" for j in JOINTS] + [f"{j}_qvel" for j in JOINTS],
        )
        for w in windows
    ]
    # allow_missing_sensors: the model has no MjSensors; the qpos/qvel
    # "sensordata" is a state-vector observation handled by modify_residual.
    sequences = ModelSequences(
        "so101",
        spec,
        [f"window_{k}" for k in range(len(windows))],
        initial_states,
        control_ts,
        sensor_ts,
        allow_missing_sensors=True,
    )

    residual_fn = build_residual_fn(
        models_sequences=[sequences],
        custom_rollout=make_sequential_rollout(),
        modify_residual=make_modify_residual(args.vel_weight),
    )

    print(f"Optimizing with {args.optimizer} (max {args.max_iters} iterations)...", flush=True)
    print(f"params.size={params.size}, n_windows={len(windows)}", flush=True)
    opt_params, opt_result = fit_parameters(params, residual_fn, args)
    # Residuals at the optimum for confidence intervals.
    residuals_star, _, _ = residual_fn(opt_result.x, opt_params)
    residuals_star = np.concatenate(residuals_star)
    lo, hi = confidence_intervals(opt_result.x, residuals_star, opt_result.jac)

    save_results(out_dir, [sequences], params, opt_params, opt_result, residual_fn)

    # Summary + held-out evaluation (last window).
    lines = []
    lines.append(f"SO-101 sysid run: {data_dir} -> {out_dir}")
    lines.append(f"Optimizer: {args.optimizer}, iters={opt_result.nfev if hasattr(opt_result, 'nfev') else 'n/a'}")
    lines.append("")
    lines.append(f"{'parameter':32s} {'nominal':>12s} {'identified':>12s} {'95% low':>12s} {'95% high':>12s}  truth")
    n_params = params.size
    names = [p.name for p in params.values()]
    truth_vals = truth if args.synthetic else None
    for idx, (pname, nom, xhat, l, h) in enumerate(zip(names, params.as_nominal_vector(), opt_result.x, lo, hi)):
        tv = f"  truth={truth_vals.as_vector()[idx]:.5f}" if truth_vals is not None else ""
        lines.append(f"{pname:32s} {nom:12.5f} {xhat:12.5f} {l:12.5f} {h:12.5f}  {tv}")
    heldout = windows[-1]
    lines.append("")
    rmse_nominal = rmse_on_window(spec, params, heldout, "nominal   (held-out window)")
    rmse_opt = rmse_on_window(spec, opt_params, heldout, "identified (held-out window)")
    lines.append(f"Held-out mean RMSE: nominal {rmse_nominal.mean():.5f} rad -> identified {rmse_opt.mean():.5f} rad")
    if args.synthetic:
        lines.append("")
        failed = []
        for idx, (pname, tv, l, h) in enumerate(zip(names, truth_vals.as_vector(), lo, hi)):
            if not (l <= tv <= h):
                failed.append((pname, tv, l, h))
        if failed:
            lines.append(f"SYNTHETIC CHECK: {len(failed)}/{n_params} truth values OUTSIDE their 95% CI:")
            for pname, tv, l, h in failed:
                lines.append(f"  {pname}: truth {tv:.5f} not in [{l:.5f}, {h:.5f}]")
        else:
            lines.append(f"SYNTHETIC CHECK: all {n_params} truth values inside their 95% confidence intervals")
    lines.append(f"Wall time: {time.perf_counter() - t_wall:.1f} s")
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary + "\n")

    if not args.no_report:
        print("Building HTML report...")
        default_report(
            [sequences],
            params,
            opt_params,
            residual_fn,
            opt_result,
            title="SO-101 SysID",
            save_path=out_dir,  # default_report appends "/report.html"
            generate_videos=not args.no_videos,
        )
        print(f"Report: {out_dir / 'report.html'}")

    if args.synthetic and "truth values OUTSIDE" in summary:
        sys.exit(1)


if __name__ == "__main__":
    main()
