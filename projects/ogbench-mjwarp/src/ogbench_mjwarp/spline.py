"""Object-relative event programs and smooth, bounded joint trajectories.

No simulator or GPU dependency: the task adapter supplies poses and contact
expectations; the executor supplies collision-checked joint paths between them.
Quaternions throughout this module use MuJoCo's wxyz order.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
    return result


def matrix_pose(matrix):
    return np.r_[
        matrix[:3, 3], np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)
    ]


@dataclass
class Event:
    name: str
    pose: np.ndarray
    object_index: int
    gripper: float
    contact: bool = False
    carrying: bool = False
    stop: bool = False
    dwell: float = 0.0
    duration_scale: float = 1.0
    transfer_index: int = 0
    support_contact: bool = False
    # Geometric constraint, independent of expected contact and stop semantics.
    anchor: bool = True
    max_linear_speed: float | None = None
    max_angular_speed: float | None = None

    def __post_init__(self):
        for limit in (self.max_linear_speed, self.max_angular_speed):
            if limit is not None and (not np.isfinite(limit) or limit <= 0):
                raise ValueError(
                    "Event speed bounds must be positive; use stop for zero speed"
                )


def grasp_samples(config, grasp_rng, path_rng, timing_rng):
    """Sample a reproducible attempt without resampling its difficulty after failure."""
    challenging = (
        config.variation == "mixed" and grasp_rng.random() < config.challenging_fraction
    )
    stratum = (
        "nominal"
        if config.variation == "nominal"
        else "challenging"
        if challenging
        else "moderate"
    )
    level = int(challenging)
    enabled = stratum != "nominal"
    samples = []
    for candidate in range(config.grasp_candidates):
        offset = (
            grasp_rng.uniform(
                -config.grasp_offset[level], config.grasp_offset[level], 2
            )
            if enabled
            else np.zeros(2)
        )
        yaw = (
            grasp_rng.uniform(-config.yaw_degrees[level], config.yaw_degrees[level])
            if enabled
            else 0.0
        )
        symmetry = int(grasp_rng.integers(4)) if enabled else candidate % 4
        # Uniform solid angle in a cone, with no preferred tilt direction.
        tilt = (
            np.arccos(
                grasp_rng.uniform(np.cos(np.deg2rad(config.tilt_degrees[level])), 1)
            )
            if enabled
            else 0.0
        )
        azimuth = grasp_rng.uniform(-np.pi, np.pi) if enabled else 0.0
        rotation = (
            Rotation.from_euler("z", symmetry * np.pi / 2 + np.deg2rad(yaw))
            * Rotation.from_rotvec(
                tilt * np.array([np.cos(azimuth), np.sin(azimuth), 0])
            )
            * Rotation.from_euler("x", np.pi)
        )
        # Raise the pinch point enough to keep the low tilted pad off the table.
        depth = 0.025 * np.sin(tilt)
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = [*offset, depth]
        samples.append(
            {
                "object_to_hand": matrix_pose(transform),
                "offset_xy": offset,
                "yaw_symmetry": symmetry,
                "yaw_delta_degrees": yaw,
                "tilt_radians": tilt,
                "tilt_azimuth_radians": azimuth,
                "depth_offset_m": depth,
            }
        )
    result = {
        "schema_version": 2,
        "stratum": stratum,
        "distribution": asdict(config),
        "candidates": samples,
        "placement_offset_xy": path_rng.uniform(
            -config.placement_offset[level], config.placement_offset[level], (2, 2)
        )
        if enabled
        else np.zeros((2, 2)),
        "path_offset_xyz": path_rng.uniform(
            -config.path_offset[level], config.path_offset[level], (2, 3)
        )
        if enabled
        else np.zeros((2, 3)),
        "duration_scales": timing_rng.uniform(
            config.duration_min[level], config.duration_max[level], (2, 9)
        )
        if enabled
        else np.ones((2, 9)),
        "dwell_seconds": float(timing_rng.uniform(config.dwell_min, config.dwell_max))
        if enabled
        else 0.4,
        "method": "object_relative_uniform_offsets_symmetry_yaw_uniform_solid_angle_tilt; independent path/timing streams",
    }
    # Sample new factors after the existing streams to preserve default draws.
    angle = path_rng.uniform(-np.pi, np.pi, (2, 2))
    radius = config.approach_offset[level] * np.sqrt(path_rng.uniform(size=(2, 2)))
    height = path_rng.uniform(
        config.approach_height_min[level], config.approach_height_max[level], (2, 2)
    )
    result["approach_vectors"] = (
        np.stack((radius * np.cos(angle), radius * np.sin(angle), height), axis=-1)
        if enabled
        else np.tile([0.0, 0.0, 0.14], (2, 2, 1))
    )
    result["execution_speed"] = (
        float(timing_rng.uniform(config.speed_min[level], config.speed_max[level]))
        if enabled
        else 1.0
    )
    result["candidate_priority"] = grasp_rng.uniform(size=(2, config.grasp_candidates))
    result["factor_methods"] = {
        "approach_vectors": "world-frame offsets from pick/place; uniform-area XY disk, independent uniform height, per object and endpoint",
        "execution_speed": "uniform per attempt; slows the feasible retimed motion by 1/speed, excluding gripper dwells",
        "candidate_priority": "independent uniform priorities per transfer; random selection chooses minimum priority among feasible proposals",
    }
    return result


def transfer_events(
    object_pose, goal_pose, object_index, sample, candidate, transfer_index=0
):
    """The stacking adapter: one transfer, expressed entirely in full poses."""
    offset = pose_matrix(sample["candidates"][candidate]["object_to_hand"])
    pick = pose_matrix(object_pose) @ offset
    goal = pose_matrix(goal_pose).copy()
    goal[:2, 3] += sample["placement_offset_xy"][object_index]
    place = goal @ offset
    above_pick, above_place = pick.copy(), place.copy()
    above_pick[2, 3] += 0.14
    above_place[2, 3] += 0.14
    approach_pick, approach_place = pick.copy(), place.copy()
    approach_pick[:3, 3] += sample["approach_vectors"][object_index, 0]
    approach_place[:3, 3] += sample["approach_vectors"][object_index, 1]
    middle = above_place.copy()
    middle[:3, 3] = (above_pick[:3, 3] + above_place[:3, 3]) / 2 + sample[
        "path_offset_xyz"
    ][object_index]
    middle[2, 3] = max(middle[2, 3], 0.20)
    # Stops only at closure/release. Transit endpoints are blended in joint space.
    specs = [
        ("approach", approach_pick, 0, False, False, False),
        ("grasp", pick, 0, True, False, True),
        ("close", pick, 1, True, False, True),
        ("lift", above_pick, 1, True, True, False),
        ("transport", middle, 1, False, True, False),
        ("preplace", approach_place, 1, False, True, False),
        ("place", place, 1, True, True, True),
        ("release", place, 0, True, True, True),
        ("retreat", above_place, 0, True, False, True),
    ]
    return [
        Event(
            name,
            matrix_pose(pose),
            object_index,
            grip,
            contact,
            carry,
            stop
            and not (
                name == "retreat" and sample["distribution"]["retiming"] == "local"
            ),
            sample["dwell_seconds"] if name in ("close", "release") else 0.0,
            float(sample["duration_scales"][object_index, i]),
            transfer_index,
            name in ("place", "release", "retreat"),
            stop,
            sample["distribution"]["transit_linear_speed"]
            if sample["distribution"]["retiming"] == "local"
            else None,
            sample["distribution"]["transit_angular_speed"]
            if sample["distribution"]["retiming"] == "local"
            else None,
        )
        for i, (name, pose, grip, contact, carry, stop) in enumerate(specs)
    ]


def fit_path(
    paths,
    durations,
    max_velocity=1.0,
    max_acceleration=2.0,
    max_jerk=50.0,
    timestep=0.05,
    relaxation=0.0,
    anchors=None,
    diagnostics=None,
    retiming="uniform",
    event_speed=None,
    execution_speed=1.0,
):
    """Blend paths between contact stops, with continuous velocity/acceleration.

    Returns samples INCLUDING the initial state, realized event arrival times,
    and a dense geometric probe for validation. Local retiming preserves the
    geometric curve and event ordering, but may change relative durations.
    """
    if not np.isfinite(relaxation) or relaxation < 0:
        raise ValueError("Relaxation must be finite and nonnegative")
    anchors = [True] * len(paths) if anchors is None else anchors
    knots, times, arrivals = [np.asarray(paths[0][0], dtype=float)], [0.0], []
    fixed = [0]
    for path, duration, anchor in zip(paths, durations, anchors, strict=True):
        path = np.asarray(path)
        if len(path) < 2 or duration <= 0 or not np.isfinite(path).all():
            raise ValueError("Invalid spline initialization")
        indices = np.unique(
            np.linspace(0, len(path) - 1, min(9, len(path))).astype(int)
        )
        for index in indices[1:]:
            knots.append(path[index])
            times.append(
                (arrivals[-1] if arrivals else 0.0) + duration * index / (len(path) - 1)
            )
        arrivals.append(times[-1])
        if anchor:
            fixed.append(len(knots) - 1)
    fixed.append(len(knots) - 1)
    curve = make_interp_spline(
        times,
        np.asarray(knots),
        k=5,
        bc_type=(
            [(1, np.zeros(6)), (2, np.zeros(6))],
            [(1, np.zeros(6)), (2, np.zeros(6))],
        ),
    )
    # Probe at >= 200 Hz and subdivide any step exceeding 0.01 rad below.
    grid = np.unique(
        np.r_[np.linspace(0, times[-1], max(33, int(times[-1] / 0.005) + 1)), arrivals]
    )
    original = curve
    if relaxation:
        # The interpolation map is linear in knot values. Minimize integrated
        # squared jerk while holding anchors and the boundary derivatives fixed.
        # Three Gauss points per polynomial span integrate squared quintic jerk
        # exactly (degree four), including nonuniform planner knot spacing.
        n = len(knots)
        basis = make_interp_spline(
            np.asarray(times) / times[-1],
            np.eye(n),
            k=5,
            bc_type=(
                [(1, np.zeros(n)), (2, np.zeros(n))],
                [(1, np.zeros(n)), (2, np.zeros(n))],
            ),
        )
        spans = np.unique(basis.t)
        low, high = spans[:-1], spans[1:]
        nodes, weights = np.polynomial.legendre.leggauss(3)
        probe_t = (
            (low[:, None] + high[:, None]) / 2 + (high - low)[:, None] * nodes / 2
        ).ravel()
        weights = ((high - low)[:, None] * weights / 2).ravel()
        jerk = basis(probe_t, 3) * np.sqrt(weights[:, None])
        # Scale the objective without changing its minimizer.
        jerk /= max(np.linalg.norm(jerk), 1e-12)
        free = np.setdiff1d(np.arange(n), fixed)
        delta = np.zeros_like(knots, dtype=float)
        for joint in range(delta.shape[1]):
            if not len(free):
                break
            result = lsq_linear(
                jerk[:, free],
                -(jerk @ np.asarray(knots)[:, joint]),
                bounds=(-relaxation, relaxation),
                method="bvls",
                tol=1e-10,
                max_iter=100,
            )
            if not result.success:
                raise ValueError("Bounded spline smoothing did not converge")
            delta[free, joint] = result.x
        # A B-spline lies in the convex hull of its coefficients. This bounds
        # displacement along the ENTIRE curve, not just at sampled waypoints.
        coefficient_change = basis.c @ delta
        deviation = float(np.max(np.abs(coefficient_change)))
        delta *= min(1.0, relaxation / max(deviation, 1e-12))
        curve = make_interp_spline(
            times,
            np.asarray(knots) + delta,
            k=5,
            bc_type=(
                [(1, np.zeros(6)), (2, np.zeros(6))],
                [(1, np.zeros(6)), (2, np.zeros(6))],
            ),
        )
    if diagnostics is not None:
        diagnostics.update(
            relaxation_radians=relaxation,
            max_joint_deviation_bound=float(np.max(np.abs(curve.c - original.c))),
            nominal_jerk_integral_before=float(
                np.trapezoid(np.sum(original(grid, 3) ** 2, axis=1), grid)
            ),
            nominal_jerk_integral_after=float(
                np.trapezoid(np.sum(curve(grid, 3) ** 2, axis=1), grid)
            ),
        )
    return retime_curve(
        curve,
        grid,
        np.asarray(arrivals),
        (max_velocity, max_acceleration, max_jerk),
        timestep,
        retiming,
        event_speed,
        diagnostics,
        execution_speed,
    )


def retime_curve(
    curve,
    grid,
    arrivals,
    limits,
    timestep,
    mode,
    event_speed,
    diagnostics,
    execution_speed=1.0,
):
    """Choose a checked monotone clock without changing the geometric path.

    Local derivative demands seed a smooth time density; full chain-rule and
    discrete command checks correct its timing. This is bounded heuristic
    retiming, not a globally time-optimal solver. A compressed uniform clock
    competes with the local clock, avoiding a slower local solution.
    """
    if mode not in ("uniform", "local"):
        raise ValueError("Unknown retiming mode")
    if not np.isfinite(execution_speed) or not 0 < execution_speed <= 1:
        raise ValueError("Execution speed must be in (0, 1]")
    derivatives = [curve(grid, n) for n in (1, 2, 3)]
    demands = np.array(
        [
            (np.max(abs(d), axis=1) / limit) ** (1 / n)
            for n, (d, limit) in enumerate(zip(derivatives, limits, strict=True), 1)
        ]
    )
    required = (
        np.zeros(len(arrivals))
        if event_speed is None
        else np.asarray(event_speed(curve(arrivals), curve(arrivals, 1)), dtype=float)
    )
    if (
        required.shape != arrivals.shape
        or not np.isfinite(required).all()
        or np.any(required < 0)
    ):
        raise ValueError("Invalid event speed constraints")
    # Constant groups are interaction dwells: never compress them.
    moving = np.max(np.ptp(curve(grid), axis=0)) > 1e-8
    uniform = max(
        float(np.max(demands)),
        float(np.max(required)),
        0.01 if mode == "local" and moving else 1.0,
    )
    endpoints = np.array([0.0, grid[-1]])
    clocks = [("uniform", endpoints, endpoints * uniform)]
    if mode == "local" and moving:
        # Use a uniform parameter grid for convolution, independently of the
        # event probes inserted into the geometry-validation grid.
        u = np.linspace(0, grid[-1], max(129, len(grid)))
        density = np.interp(u, grid, np.max(demands, axis=0))
        density = gaussian_filter1d(density, max(2, len(u) / 24), mode="nearest")
        density = np.maximum(density, uniform * 0.1)
        for stamp, demand in zip(arrivals, required, strict=True):
            density = np.maximum(
                density, demand * np.exp(-0.5 * ((u - stamp) / (grid[-1] / 12)) ** 2)
            )
        wall = cumulative_trapezoid(density, u, initial=0)
        clocks.append(("local", u, wall))
    candidates = []
    for name, parameter, wall in clocks:
        clock = CubicSpline(wall, parameter)
        # Check the quadratic clock derivative's minima on EVERY span.
        c = clock.c
        vertex = np.divide(
            -c[1], 3 * c[0], out=np.zeros_like(c[0]), where=abs(c[0]) > 1e-15
        )
        vertex = np.clip(vertex, 0, np.diff(wall))
        if (
            min(
                np.min(clock(wall, 1)),
                np.min(3 * c[0] * vertex**2 + 2 * c[1] * vertex + c[2]),
            )
            <= 0
        ):
            continue

        def inverse(value, clock=clock, wall=wall, parameter=parameter):
            t = np.interp(value, parameter, wall)
            for _ in range(5):
                t = np.clip(t - (clock(t) - value) / clock(t, 1), 0, wall[-1])
            return t

        event_times = inverse(arrivals)
        check_t = np.unique(
            np.r_[wall, (wall[:-1] + wall[1:]) / 2, inverse(grid), event_times]
        )
        u = np.clip(clock(check_t), 0, grid[-1])
        r, a, j = [clock(check_t, n)[:, None] for n in (1, 2, 3)]
        q1, q2, q3 = [curve(u, n) for n in (1, 2, 3)]
        continuous = (q1 * r, q2 * r * r + q1 * a, q3 * r**3 + 3 * q2 * r * a + q1 * j)
        scale = (
            max(
                1.0,
                *(
                    float(np.max(abs(d)) / limit) ** (1 / n)
                    for n, (d, limit) in enumerate(
                        zip(continuous, limits, strict=True), 1
                    )
                ),
                float(np.max(required * clock(event_times, 1))),
            )
            * 1.02
            / (execution_speed if moving else 1.0)
        )
        for _ in range(30):
            duration = np.ceil(wall[-1] * scale / timestep) * timestep
            scale = duration / wall[-1]
            ticks = np.linspace(0, wall[-1], round(duration / timestep) + 1)
            values = curve(np.clip(clock(ticks), 0, grid[-1]))
            padded = np.vstack(
                (values[:1].repeat(3, 0), values, values[-1:].repeat(3, 0))
            )
            ratio = max(
                float(np.max(abs(np.diff(padded, n=n, axis=0))) / timestep**n / limit)
                ** (1 / n)
                for n, limit in enumerate(limits, 1)
            )
            if ratio <= 1 + 1e-7:
                # Include the exact executed targets in geometric validation.
                probe_u = np.unique(np.r_[grid, np.clip(clock(ticks), 0, grid[-1])])
                probe = curve(probe_u)
                while np.max(abs(np.diff(probe, axis=0))) > 0.01:
                    probe_u = np.unique(
                        np.r_[probe_u, (probe_u[:-1] + probe_u[1:]) / 2]
                    )
                    probe = curve(probe_u)
                stats = {
                    "method": name,
                    "execution_speed": execution_speed if moving else 1.0,
                    "duration_seconds": float(duration),
                    "event_speed_ratios": (
                        required * clock(event_times, 1) / scale
                    ).tolist(),
                    "event_arrivals_seconds": (event_times * scale).tolist(),
                }
                candidates.append(
                    (
                        values,
                        event_times * scale,
                        (probe, inverse(probe_u) * scale),
                        duration / grid[-1],
                        stats,
                    )
                )
                break
            scale *= max(1.05, ratio * 1.01)
    if not candidates:
        raise ValueError("Spline could not satisfy timing and derivative limits")
    best = min(candidates, key=lambda c: len(c[0]))
    if diagnostics is not None:
        diagnostics["retiming"] = best[-1] | {
            "requested": mode,
            "candidate_durations_seconds": {
                c[-1]["method"]: c[-1]["duration_seconds"] for c in candidates
            },
        }
    return best[:4]


def checked_fit(paths, durations, validate, *, relaxation=0.0, **kwargs):
    """Try bounded smoothing, back off once, then validate the original curve.

    The callback checks task geometry; this fitting policy knows no task names.
    No unvalidated candidate or fallback is ever returned.
    """
    attempts = []
    for radius in (relaxation, relaxation / 2, 0.0) if relaxation else (0.0,):
        record = {}
        try:
            fitted = fit_path(
                paths, durations, relaxation=radius, diagnostics=record, **kwargs
            )
        except ValueError as error:
            attempts.append(
                {"relaxation_radians": radius, "valid": False, "reason": str(error)}
            )
            continue
        valid, reason = validate(fitted, radius, record)
        attempts.append(record | {"valid": bool(valid), "reason": reason})
        if valid:
            return fitted, attempts
    return None, attempts


def quality_selection(row, quality):
    """Unknown/legacy quality is never silently accepted as audited."""
    if quality == "all":
        return True
    if quality not in ("validated-success", "valid-failure"):
        raise ValueError(f"Unknown quality filter: {quality}")
    check = row.get("quality", {})
    if check.get("physical_valid") is not True or check.get("completed") is not True:
        return False
    return check.get("stable_success") is (quality == "validated-success")
