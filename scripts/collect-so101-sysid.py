#!/usr/bin/env python
"""Collect system-identification data for an SO-101 follower arm.

Drives all six joints with an eased per-joint multisine trajectory (rizon-style
excitation) and records commanded position, measured position and temperature
at a high capture rate (default 500 Hz) using a single Feetech GroupSyncRead
per cycle. The recording is consumed by scripts/run-so101-sysid.py.

Data format (data.csv), one row per capture cycle:
    t, q_cmd_<joint> x6, qpos_<joint> x6, qvel_<joint> x6 (deg/s, informational),
    temp_<joint> x6 (deg C)

meta.json records the port, motor ids, calibration, trajectory definition
(amplitude, per-joint frequencies, phases, home), servo PD gains and rates.

Safety: commands are bounded to --amplitude degrees around home (capped at
60% of the calibrated range per joint), velocity is bounded by the trajectory
itself (max |dq/dt| = A * 2*pi * fmax), and the run aborts if any motor
temperature exceeds --temp-limit. Ctrl-C commands the home pose before exit.

Run with --dry-run to write the planned trajectory without touching motors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}
MOTOR_MODELS = {name: "sts3215" for name in MOTOR_NAMES}
# STS3215: 4096 counts/turn -> max_res = 4095, half-turn home = 2047.
STS3215_MAX_RES = 4095
STS3215_MID = STS3215_MAX_RES // 2
# Feetech STS3215 datasheet velocity resolution (deg/s per raw count).
STS3215_VEL_UNIT = 0.22

HOME_DEG = [0.0] * 5 + [50.0]  # gripper home is mid-stroke in 0-100 norm.


@dataclass
class Plan:
    amplitude: float
    freqs: list[float]
    phases: list[float]
    t0: float
    t1: float
    dwell: float
    command_rate: float
    rate: float

    @property
    def active_start(self) -> float:
        return self.dwell

    @property
    def active_end(self) -> float:
        return self.t1 - self.dwell

    def q_cmd(self, t: float) -> list[float]:
        """Eased multisine position command (degrees) at time t."""
        if t < self.active_start or t >= self.active_end:
            return list(HOME_DEG)
        a = (t - self.active_start) / (self.active_end - self.active_start)
        # smoothstep ramp on amplitude over the first/last 12% of the run
        ramp = 1.0
        if a < 0.12:
            s = a / 0.12
            ramp = s * s * (3 - 2 * s)
        elif a > 0.88:
            s = (1.0 - a) / 0.12
            ramp = s * s * (3 - 2 * s)
        q = []
        for j in range(len(MOTOR_NAMES)):
            q.append(HOME_DEG[j] + ramp * self.amplitude * math.sin(2 * math.pi * self.freqs[j] * t + self.phases[j]))
        return q

    def max_velocity(self) -> float:
        return self.amplitude * 2 * math.pi * max(self.freqs)


def plan_trajectory(args: argparse.Namespace, rng) -> Plan:
    freqs = [
        args.fmin + (args.fmax - args.fmin) * j / (len(MOTOR_NAMES) - 1)
        for j in range(len(MOTOR_NAMES))
    ]
    phases = [float(rng.uniform(0, 2 * math.pi)) for _ in MOTOR_NAMES]
    return Plan(
        amplitude=args.amplitude,
        freqs=freqs,
        phases=phases,
        t0=0.0,
        t1=args.duration,
        dwell=2.0,
        command_rate=args.command_rate,
        rate=args.rate,
    )


def clamp_amplitude(plan: Plan, calib: dict) -> None:
    """Cap the amplitude at 60% of the calibrated range around home, per joint."""
    from lerobot.motors.motors_bus import MotorCalibration  # noqa: PLC0415

    worst = 1.0
    for j, name in enumerate(MOTOR_NAMES):
        cal = calib[name]
        max_res = STS3215_MAX_RES
        # Calibrated range is raw counts around the half-turn home (STS3215_MID).
        below = STS3215_MID - cal.range_min
        above = cal.range_max - STS3215_MID
        half_range = min(below, above)
        if name == "gripper":
            # gripper normalized 0-100 around home 50
            half_deg = half_range * 100.0 / max_res
        else:
            half_deg = half_range * 360.0 / max_res
        cap = 0.6 * half_deg
        if cap < plan.amplitude:
            print(f"  joint {name}: amplitude capped {plan.amplitude:.1f} -> {cap:.1f} deg (calibrated range)")
        worst = min(worst, cap / plan.amplitude) if plan.amplitude > 0 else 1.0
    plan.amplitude = max(0.5, plan.amplitude * worst)


def write_outputs(out_dir: Path, plan: Plan, meta_extra: dict, rows: list[list[float]] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["t"]
    for prefix in ("q_cmd", "qpos", "qvel", "temp"):
        fields += [f"{prefix}_{name}" for name in MOTOR_NAMES]
    with open(out_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        if rows:
            w.writerows(rows)
    meta = {
        "motor_names": MOTOR_NAMES,
        "motor_ids": MOTOR_IDS,
        "motor_models": MOTOR_MODELS,
        "home_deg": HOME_DEG,
        "amplitude_deg": plan.amplitude,
        "freqs_hz": plan.freqs,
        "phases_rad": plan.phases,
        "duration_s": plan.t1,
        "dwell_s": plan.dwell,
        "capture_rate_hz": plan.rate,
        "command_rate_hz": plan.command_rate,
        "max_velocity_deg_s": plan.max_velocity(),
        "velocity_unit_deg_s": STS3215_VEL_UNIT,
        "vel_resolution_counts_per_turn": STS3215_MAX_RES,
    }
    meta.update(meta_extra)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir / 'data.csv'} and {out_dir / 'meta.json'}")


def load_calibration(path: Path) -> dict:
    """Load a lerobot calibration file: {motor: {id, drive_mode, homing_offset, range_min, range_max}}."""
    from lerobot.motors.motors_bus import MotorCalibration  # noqa: PLC0415

    raw = json.loads(path.read_text())
    calib = {}
    for name, vals in raw.items():
        calib[name] = MotorCalibration(**vals)
    return calib


def find_calibration(explicit: str | None, robot_id: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"Calibration file not found: {p}")
        return p
    candidates = [
        Path("left") if robot_id == "left" else Path(robot_id),
        Path.home() / ".cache" / "hub" / "lerobot" / "calibrations" / "robots" / "so_follower" / f"{robot_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    sys.exit(
        "No calibration file found. Pass --calibration <file> or run:\n"
        "  uv run lerobot-calibrate --robot.type=so101_follower --robot.port=COM4 --robot.id=left"
    )


def make_bus(port: str, calib: dict):
    from lerobot.motors.feetech.feetech import FeetechMotorsBus  # noqa: PLC0415
    from lerobot.motors.motors_bus import Motor, MotorNormMode  # noqa: PLC0415

    motors = {}
    for name in MOTOR_NAMES:
        norm = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES
        motors[name] = Motor(MOTOR_IDS[name], MOTOR_MODELS[name], norm)
    return FeetechMotorsBus(port=port, motors=motors, calibration=calib)


def raw_position_deg(raw: int, motor: str) -> float:
    if motor == "gripper":
        return (raw - STS3215_MID) * 100.0 / STS3215_MAX_RES
    return (raw - STS3215_MID) * 360.0 / STS3215_MAX_RES


def collect(args: argparse.Namespace, plan: Plan, calib: dict, out_dir: Path) -> None:
    import scservo_sdk as scs  # noqa: PLC0415

    bus = make_bus(args.port, calib)
    try:
        bus.connect()
        bus.enable_torque()
        # Position mode + servo PD gains (recorded in meta for the sysid actuator model).
        for name in MOTOR_NAMES:
            bus.write("Operating_Mode", name, 3)
            bus.write("P_Coefficient", name, args.p_coefficient)
            bus.write("D_Coefficient", name, args.d_coefficient)
        # Move to home and wait for it to settle.
        for name, home in zip(MOTOR_NAMES, HOME_DEG):
            bus.write("Goal_Position", name, home)
        print("At home; capturing for 2 s...")
        time.sleep(2.0)

        reader = bus.sync_reader
        ph = bus.packet_handler
        port_h = bus.port_handler
        from lerobot.motors.motors_bus import get_address  # noqa: PLC0415

        param_addrs = {}
        for name in MOTOR_NAMES:
            addrs = {}
            for dn in ("Present_Position", "Present_Velocity", "Temperature"):
                addr, length = get_address(bus.model_ctrl_table, MOTOR_MODELS[name], dn)
                addrs[dn] = (addr, length)
            param_addrs[MOTOR_IDS[name]] = addrs

        def bulk_read() -> tuple[list[float], list[float], list[float]] | None:
            for attempt in range(3):
                reader.clearTxPacketRxs()
                for mid, addrs in param_addrs.items():
                    reader.addParamMotor(mid, addrs)
                if not reader.txRx():
                    continue
                pos, vel, temp = [], [], []
                for name in MOTOR_NAMES:
                    mid = MOTOR_IDS[name]
                    pos.append(raw_position_deg(reader.getRxPacketData(mid, "Present_Position"), name))
                    vel.append(reader.getRxPacketData(mid, "Present_Velocity") * STS3215_VEL_UNIT)
                    temp.append(float(reader.getRxPacketData(mid, "Temperature")))
                return pos, vel, temp
            return None

        n_captures = int(plan.t1 * plan.rate)
        rows: list[list[float]] = []
        dropped = 0
        abort = threading.Event()

        def on_sigint(signum, frame):
            print("\nCtrl-C: commanding home, then aborting.")
            abort.set()

        signal.signal(signal.SIGINT, on_sigint)

        t_start = time.perf_counter()
        next_cmd_i = 0
        n_cmds = int(plan.t1 * plan.command_rate) + 1
        cmd_times = [i / plan.command_rate for i in range(n_cmds)]
        current_cmd = [HOME_DEG] * len(MOTOR_NAMES)

        for i in range(n_captures):
            t_target = i / plan.rate
            if abort.is_set():
                break
            # Command at the scheduled tick(s).
            while next_cmd_i < n_cmds and cmd_times[next_cmd_i] <= t_target:
                current_cmd = plan.q_cmd(cmd_times[next_cmd_i])
                for name, val in zip(MOTOR_NAMES, current_cmd):
                    bus.write("Goal_Position", name, val)
                next_cmd_i += 1
            # Capture.
            now = time.perf_counter() - t_start
            reading = bulk_read()
            if reading is None:
                dropped += 1
                continue
            pos, vel, temp = reading
            rows.append([now, *current_cmd, *pos, *vel, *temp])
            if max(temp) >= args.temp_limit:
                print(f"\nTemperature {max(temp):.1f} C >= {args.temp_limit} C: aborting.")
                abort.set()
            if (i + 1) % int(plan.rate) == 0:
                err = max(abs(p - c) for p, c in zip(pos, current_cmd))
                print(f"t={now:6.1f}s  max tracking error {err:5.2f} deg  max temp {max(temp):.1f} C  dropped={dropped}")
        # Settle at home.
        for name, home in zip(MOTOR_NAMES, HOME_DEG):
            bus.write("Goal_Position", name, home)
    finally:
        try:
            bus.disable_torque()
            bus.disconnect()
        except Exception:
            pass

    write_outputs(
        out_dir,
        plan,
        meta_extra={
            "port": args.port,
            "robot_id": args.id,
            "calibration": {k: vars(v) for k, v in calib.items()},
            "p_coefficient": args.p_coefficient,
            "d_coefficient": args.d_coefficient,
            "samples": len(rows),
            "dropped_reads": dropped,
            "captured_s": rows[-1][0] if rows else 0.0,
        },
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM4", help="Serial port of the arm (default: COM4)")
    parser.add_argument("--id", default="left", help="Robot id used to locate the calibration file")
    parser.add_argument("--calibration", default=None, help="Path to the calibration JSON (default: <id> file in repo root, then lerobot cache)")
    parser.add_argument("--duration", type=float, default=300.0, help="Total capture duration in seconds (default: 300)")
    parser.add_argument("--amplitude", type=float, default=8.0, help="Multisine amplitude in degrees around home (default: 8)")
    parser.add_argument("--fmin", type=float, default=0.3, help="Lowest excitation frequency in Hz (default: 0.3)")
    parser.add_argument("--fmax", type=float, default=0.9, help="Highest excitation frequency in Hz (default: 0.9)")
    parser.add_argument("--rate", type=float, default=500.0, help="Capture rate in Hz (default: 500)")
    parser.add_argument("--command-rate", type=float, default=100.0, help="Command rate in Hz (default: 100)")
    parser.add_argument("--temp-limit", type=float, default=75.0, help="Abort if any motor temperature exceeds this (deg C, default: 75)")
    parser.add_argument("--p-coefficient", type=int, default=16, help="Servo P coefficient written at start (default: 16)")
    parser.add_argument("--d-coefficient", type=int, default=32, help="Servo D coefficient written at start (default: 32)")
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/so101_sysid/<stamp>)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for excitation phases")
    parser.add_argument("--dry-run", action="store_true", help="Plan the trajectory and write outputs without touching motors")
    args = parser.parse_args()

    if not (0 < args.fmin < args.fmax):
        sys.exit(f"Require 0 < fmin < fmax, got {args.fmin}, {args.fmax}")
    if args.rate < 100:
        sys.exit("Capture rate must be at least 100 Hz")
    if args.command_rate > args.rate:
        sys.exit("Command rate must not exceed the capture rate")

    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(args.seed)
    plan = plan_trajectory(args, rng)
    print(
        f"Plan: {args.duration:.0f}s at {args.rate:.0f} Hz, amplitude {plan.amplitude} deg, "
        f"freqs {[round(f, 3) for f in plan.freqs]} Hz, max |dq/dt| = {plan.max_velocity():.1f} deg/s"
    )

    out_dir = Path(args.out) if args.out else Path("outputs/so101_sysid") / time.strftime("%Y%m%d-%H%M%S")

    if args.dry_run:
        print("--dry-run: not touching motors")
        rows = []
        t = 0.0
        nan18 = [float("nan")] * 18
        while t < plan.t1:
            rows.append([t, *plan.q_cmd(t), *nan18])
            t += 1.0 / args.rate
        write_outputs(out_dir, plan, meta_extra={"dry_run": True, "samples": len(rows)}, rows=rows)
        return

    calib = load_calibration(find_calibration(args.calibration, args.id))
    print(f"Calibration loaded from: {find_calibration(args.calibration, args.id)}")
    clamp_amplitude(plan, calib)
    print(f"Effective amplitude: {plan.amplitude:.2f} deg (max |dq/dt| = {plan.max_velocity():.1f} deg/s)")
    print("Support the arm before starting. The run lasts ~"
          f"{(args.duration + 4) / 60:.1f} min. Ctrl-C aborts safely.")
    input("Press ENTER to start (or Ctrl-C to cancel): ")
    collect(args, plan, calib, out_dir)


if __name__ == "__main__":
    main()
