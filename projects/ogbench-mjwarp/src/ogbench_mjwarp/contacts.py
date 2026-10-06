"""Contact roles and recorded-state audits; distances are in meters."""

import numpy as np
from vla_tools.contacts import (  # noqa: F401
    contact_depths,
    contact_roles,
    update_contact_quality,
)

from .io import load_sim_states, rollout_path
from .tasks import make_env, restore_cpu


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
