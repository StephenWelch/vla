"""Independent episode streams and explicit randomization provenance."""

import hashlib
from dataclasses import asdict

import numpy as np


def initial_state_fingerprint(state):
    """Identify saved reset states while allowing different controller offsets."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key != "joint_target_offset":
            array = np.asarray(value)
            digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def episode_randomization(seed, episode_id, config, planner):
    scenario, variant = divmod(episode_id, config.variants_per_reset)
    root = seed if config.seed is None else config.seed
    names = ("oracle", "order", "grasp", "path", "timing", "planner", "joint_targets")
    # Fixed stream IDs avoid changing other factors when a factor is enabled.
    seeds = {
        name: int(
            np.random.SeedSequence([root, episode_id, index]).generate_state(1)[0]
        )
        for index, name in enumerate(names)
    }
    seeds["environment"] = seed + scenario
    factors = {
        "initial_conditions": {
            "enabled": True,
            "method": "ogbench_reset",
            "seed": seeds["environment"],
        },
        "order": {"enabled": config.order, "method": "uniform_feasible_choice"},
        "cube_grasps": {
            "enabled": config.cube_grasps,
            "method": "uniform_quarter_turn_symmetry",
            "unit": "radian",
        },
        "handle_grasps": {
            "enabled": config.handle_grasps,
            "method": "uniform_half_turn_symmetry",
            "unit": "radian",
        },
        "path": {
            "enabled": bool(config.position_noise or config.yaw_noise),
            "method": "uniform_free_keyframe_offsets",
            "position_bounds": [-config.position_noise, config.position_noise],
            "height_bounds": [0.0, config.position_noise],
            "yaw_bounds": [-config.yaw_noise, config.yaw_noise],
            "position_unit": "meter",
            "angle_unit": "radian",
            "frame": "world",
        },
        "timing": {
            "enabled": config.duration_scale_min != 1 or config.duration_scale_max != 1,
            "method": "uniform_segment_scale_with_safety_floors",
            "bounds": [config.duration_scale_min, config.duration_scale_max],
        },
        "joint_targets": {
            "enabled": bool(planner.joint_target_noise),
            "method": "uniform_post_ik_episode_offset",
            "bounds": [-planner.joint_target_noise, planner.joint_target_noise],
            "unit": "radian",
            "joint_order": [f"joint_{i}" for i in range(6)],
        },
        "oracle": {
            "enabled": True,
            "method": "seeded_ogbench_keyframes",
            "note": "Upstream heights, clearance positions and timing jitter are recorded per skill.",
        },
        "cem": {
            "enabled": True,
            "method": "gaussian_action_offset_search",
            "settings": planner.to_dict(),
        },
    }
    return {
        "schema_version": 1,
        "available": True,
        "scenario_id": scenario,
        "variant_id": variant,
        "config": asdict(config),
        "seeds": seeds,
        "factors": factors,
        "skills": [],
    }
