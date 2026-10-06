"""Shared physical stack acceptance, independent of the planner's success flag."""

import numpy as np

from .tasks import restore_cpu


def audit_stack(env, states, fps=20):
    """Require a released, supported, upright stack for the final second."""
    base = env.unwrapped
    model, data = base._model, base._data
    if base._num_cubes != 2:
        raise ValueError("Stack pilot audit requires exactly two cubes")
    failures = set()
    if len(states["qpos"]) < fps + 1:
        failures.add("insufficient_stability_window")
    positions = []
    for frame in range(max(0, len(states["qpos"]) - fps - 1), len(states["qpos"])):
        restore_cpu(env, {key: value[frame] for key, value in states.items()})
        base.post_step()
        goals = data.mocap_pos[base._cube_target_mocap_ids]
        order = np.argsort(goals[:, 2])
        geoms = [model.geom(f"object_{i}").id for i in order]
        bodies = [model.geom_bodyid[g] for g in geoms]
        pos = data.xpos[bodies].copy()
        positions.append(pos)
        if not base._success:
            failures.add("native_goal")
        grip = data.qpos[model.jnt_qposadr[base._gripper_opening_joint_id]] / 0.8
        if grip > 0.15:
            failures.add("gripper_not_released")
        rotation = data.xmat[bodies].reshape(2, 3, 3)
        if np.min(rotation[:, 2, 2]) < np.cos(np.deg2rad(10)):
            failures.add("tilted_cube")
        if (
            np.linalg.norm(pos[1, :2] - pos[0, :2]) > 0.012
            or abs(pos[1, 2] - pos[0, 2] - 0.04) > 0.004
        ):
            failures.add("stack_alignment")
        support = False
        for contact in data.contact:
            if (
                set(contact.geom) == set(geoms)
                and contact.dist <= 0.001
                and abs(contact.frame[2]) >= 0.8
            ):
                support = True
            if contact.dist < -0.003 and any(g in geoms for g in contact.geom):
                failures.add("cube_penetration")
            other = [g for g in contact.geom if g not in geoms]
            if (
                any(g in geoms for g in contact.geom)
                and any(
                    model.body(model.geom_bodyid[g]).name.startswith("ur5e/")
                    for g in other
                    if g >= 0
                )
                and contact.dist <= 0
            ):
                failures.add("robot_still_touching_cube")
        if not support:
            failures.add("missing_cube_support")
    if positions and np.max(np.ptp(np.asarray(positions), axis=0)) > 0.003:
        failures.add("unstable_stack")
    return {
        "valid": not failures,
        "failures": sorted(failures),
        "stability_seconds": 1,
        "alignment_tolerance_m": 0.012,
        "motion_tolerance_m": 0.003,
    }
