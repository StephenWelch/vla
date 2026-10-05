"""Contact roles and recorded-state audits; distances are in meters."""

import numpy as np

from .io import load_sim_states, rollout_path
from .tasks import make_env, restore_cpu


def contact_roles(model):
    """Environment=0, arm=1, gripper linkage=2, finger pad=3, button=4."""
    roles = np.zeros(model.ngeom, dtype=np.int32)
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom]).name
        if body.startswith("ur5e/robotiq/"):
            roles[geom] = 3 if body.endswith(("/left_pad", "/right_pad")) else 2
        elif body.startswith("ur5e/"):
            roles[geom] = 1
        elif "button" in body:
            roles[geom] = 4
    return roles


def contact_depths(model, data, roles):
    nonpad, penetration = 0.0, 0.0
    worst = None
    for contact in data.contact:
        a, b = map(int, contact.geom)
        if a < 0 or b < 0:
            continue
        r1, r2 = roles[a], roles[b]
        if 1 <= r1 <= 3 and r2 in (0, 4):
            robot, other = r1, r2
        elif 1 <= r2 <= 3 and r1 in (0, 4):
            robot, other = r2, r1
        else:
            continue
        depth = max(0.0, -float(contact.dist))
        if depth > penetration:
            penetration = depth
            worst = [model.body(model.geom_bodyid[g]).name for g in (a, b)]
        # Closed gripper links can intentionally press buttons; arm links cannot.
        if robot != 3 and not (robot == 2 and other == 4):
            nonpad = max(nonpad, depth)
    return nonpad, penetration, worst


def update_contact_quality(quality, *depths):
    """Merge CPU/GPU peaks into episode metadata; validity is sticky across steps."""
    nonpad, penetration = np.max(depths, axis=0)
    quality["peak_nonpad_penetration"] = max(
        quality["peak_nonpad_penetration"], float(nonpad)
    )
    quality["peak_penetration"] = max(quality["peak_penetration"], float(penetration))
    quality["valid"] &= bool(
        nonpad <= quality["max_nonpad_penetration"]
        and penetration <= quality["max_penetration"]
    )
    return quality["valid"]


def audit_contacts(
    root, episode=0, max_nonpad_penetration=0.001, max_penetration=0.003
):
    if min(max_nonpad_penetration, max_penetration) < 0:
        raise ValueError("Contact tolerances must be nonnegative")
    row, archive = rollout_path(root, episode)
    states = load_sim_states(archive)
    env = make_env(row["env_id"], row["seed"], row["task_id"], row["image_size"])
    peak = np.zeros(2)
    violations = []
    try:
        base = env.unwrapped
        roles = contact_roles(base._model)
        for frame in range(len(states["qpos"])):
            restore_cpu(env, {key: value[frame] for key, value in states.items()})
            nonpad, penetration, pair = contact_depths(base._model, base._data, roles)
            peak = np.maximum(peak, [nonpad, penetration])
            if nonpad > max_nonpad_penetration or penetration > max_penetration:
                violations.append(
                    {
                        "frame": frame,
                        "nonpad_penetration": nonpad,
                        "penetration": penetration,
                        "bodies": pair,
                    }
                )
    finally:
        env.close()
    return {
        "episode": episode,
        "recorded_outcome": row["outcome"],
        "contact_valid": not violations,
        "peak_nonpad_penetration": float(peak[0]),
        "peak_penetration": float(peak[1]),
        "violations": violations,
        "scope": "Recorded states only; substep peaks and visual-mesh overlap are not reconstructed.",
    }
