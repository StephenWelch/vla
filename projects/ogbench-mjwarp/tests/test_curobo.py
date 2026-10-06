"""Pilot contracts and geometry; optional cuRobo import stays off the CEM path."""

import json

import numpy as np
import pytest
import torch
from ogbench_mjwarp.actions import action_profile, joint_profile, validate_profile
from ogbench_mjwarp.config import CuroboConfig, PlannerConfig
from ogbench_mjwarp.curobo_model import robot_config
from ogbench_mjwarp.curobo_planner import resample_path, timed_path
from ogbench_mjwarp.randomization import initial_state_fingerprint
from ogbench_mjwarp.stack_audit import audit_stack
from ogbench_mjwarp.tasks import cpu_snapshot, make_env, restore_cpu


def test_default_and_config_roundtrip():
    config = PlannerConfig()
    assert config.backend == "cem"
    assert PlannerConfig(**config.to_dict()) == config
    with pytest.raises(ValueError):
        PlannerConfig(backend="invalid")
    with pytest.raises(ValueError):
        CuroboConfig(max_velocity=float("nan"))


def test_arc_resampling_avoids_tick_per_spline_sample():
    path = torch.linspace(0, 0.1, 81)[:, None].repeat(1, 6)
    result = resample_path(path, 1)
    assert len(result) <= 3
    torch.testing.assert_close(result[-1], path[-1])
    assert torch.diff(torch.cat((path[:1], result))).abs().max() <= 0.05 + 1e-7
    assert len(resample_path(torch.zeros(81, 6), 1)) == 1


def test_timed_path_preserves_timing_and_limits():
    # Unequal segment travel must retain unequal velocity instead of being
    # flattened by arc-length sampling. Include a sharp bend that needs slowing.
    position = torch.tensor([[0.0] * 6, [0.01] * 6, [0.4] * 6, [0.45] * 6])
    velocity = torch.zeros_like(position)
    path, scale = timed_path(position, velocity, 0.2, 1.0, 2.0)
    assert scale > 1
    torch.testing.assert_close(path[-1], position[-1])
    v = torch.diff(torch.cat((position[:1], path)), dim=0) / 0.05
    a = torch.diff(torch.cat((torch.zeros_like(v[:1]), v)), dim=0) / 0.05
    assert v.abs().max() <= 1.0 + 1e-5
    assert a.abs().max() <= 2.0 + 1e-4
    assert v[-1].abs().max() == 0
    assert path[round(0.2 * scale / 0.05) - 1, 0] < 0.03
    with pytest.raises(ValueError, match="timestep"):
        timed_path(position, velocity, 0, 1, 2)
    # Gravity/contact tracking offset must be blended over the curve, rather
    # than corrected in one tiny source interval (which stretched hold phases).
    constant = torch.zeros(81, 6)
    corrected, _ = timed_path(
        constant, constant, 0.0034, 1, 2, start=torch.full((6,), 0.03)
    )
    assert len(corrected) < 20
    torch.testing.assert_close(corrected[-1], constant[-1])
    v = torch.diff(torch.cat((torch.full((1, 6), 0.03), corrected)), dim=0) / 0.05
    a = torch.diff(torch.cat((torch.zeros_like(v[:1]), v)), dim=0) / 0.05
    assert a.abs().max() < 2.0 + 1e-4


def test_motion_metrics_detect_local_oscillation_and_short_failures():
    from ogbench_mjwarp.execution_ablation import motion_metrics

    position = np.zeros((21, 6))
    velocity = np.zeros_like(position)
    velocity[1:10, 0] = np.tile([0.2, -0.2, 0.2], 3)
    result = motion_metrics(position, position, velocity)
    assert result["command"]["reversals_above_0_05_rad_s"] == 0
    assert result["actual"]["reversals_above_0_05_rad_s"] == 6
    assert result["actual"]["acceleration_peak_rad_s2"] == pytest.approx(8)
    assert (
        motion_metrics(position[:2], position[:2], velocity[:2])["command"][
            "acceleration_peak_rad_s2"
        ]
        == 0
    )


def test_matching_reset_ignores_controller_auxiliary_state():
    state = {"qpos": np.zeros(6), "joint_target_offset": np.zeros(6)}
    assert initial_state_fingerprint(state) == initial_state_fingerprint(
        state | {"joint_target_velocity": np.ones(6)}
    )


def test_pilot_pairs_and_intervals():
    from ogbench_mjwarp.pilot import paired_outcomes, success_interval

    trials = [
        {
            "backend": backend,
            "batch_size": 1,
            "repeat": 0,
            "outcomes": [{"stable_success": value} for value in outcomes],
        }
        for backend, outcomes in (("cem", [True, False]), ("curobo", [False, True]))
    ]
    pair = paired_outcomes(trials)[0]
    assert pair["cem_only"] == pair["curobo_only"] == 1
    assert pair["both"] == pair["neither"] == 0
    assert success_interval(0, 50)[1] < 0.08


def test_curobo_rejects_other_tasks_before_cuda(tmp_path):
    from ogbench_mjwarp.recording import generate

    with pytest.raises(ValueError, match="only cube-double-v0 task 5"):
        generate(tmp_path, "scene-v0", 1, [1], 0, PlannerConfig(backend="curobo"))


def test_joint_contract_and_checkpoint_mismatch(tmp_path):
    env = make_env("cube-double-v0", task_id=5, size=16)
    try:
        profile = joint_profile(env.unwrapped, CuroboConfig())
        assert validate_profile(profile) == profile
        model, grip, digest = robot_config(env.unwrapped, tmp_path, CuroboConfig())
        assert len(model["robot_cfg"]["kinematics"]["cspace"]["joint_names"]) == 6
        assert grip and len(digest) == 64
    finally:
        env.close()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format": "ogbench-mjwarp-3", "action_profile": profile})
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="profiles differ"):
        action_profile(tmp_path, checkpoint)
    (checkpoint / "action_profile.json").write_text(json.dumps(profile))
    assert action_profile(tmp_path, checkpoint) == profile
    with pytest.raises(ValueError):
        validate_profile(profile | {"bounds": [[float("nan"), 1]] * 7})


def test_stack_audit_rejects_held_or_unsupported_goal():
    env = make_env("cube-double-v0", task_id=5, size=16)
    try:
        base = env.unwrapped
        state = cpu_snapshot(env)
        for i in range(2):
            q = int(base._model.joint(f"object_joint_{i}").qposadr[0])
            state["qpos"][q : q + 3] = base._data.mocap_pos[
                base._cube_target_mocap_ids[i]
            ]
            state["qpos"][q + 3 : q + 7] = [1, 0, 0, 0]
        grip = int(base._model.jnt_qposadr[base._gripper_opening_joint_id])
        state["qpos"][grip] = 0
        restore_cpu(env, state)
        states = {
            key: np.repeat(np.asarray(value)[None], 21, axis=0)
            for key, value in state.items()
        }
        assert audit_stack(env, states)["valid"]
        states["qpos"][:, grip] = 0.8
        assert "gripper_not_released" in audit_stack(env, states)["failures"]
        states["qpos"][:, grip] = 0
        upper = int(np.argmax(base._data.mocap_pos[base._cube_target_mocap_ids, 2]))
        q = int(base._model.joint(f"object_joint_{upper}").qposadr[0])
        states["qpos"][:, q + 2] += 0.01
        assert "missing_cube_support" in audit_stack(env, states)["failures"]
    finally:
        env.close()


@pytest.mark.gpu
def test_joint_controller_rate_limits_and_closure():
    from ogbench_mjwarp.environment import BatchEnvironment

    env = make_env("cube-double-v0", task_id=5, size=16)
    try:
        config = PlannerConfig(backend="curobo", episodes=1)
        sim = BatchEnvironment(env, 1, config)
        previous = sim.ctrl[:, sim.arm_act].clone()
        previous_velocity = torch.zeros_like(previous)
        action = torch.cat((previous + 0.5, sim.tensor([[0.7]])), 1)
        for _ in range(4):
            sim.step(action)
            actual = sim.ctrl[:, sim.arm_act].clone()
            velocity = (actual - previous) / 0.05
            assert velocity.abs().max() <= config.curobo.max_velocity + 1e-5
            assert (
                (velocity - previous_velocity) / 0.05
            ).abs().max() <= config.curobo.max_acceleration + 1e-4
            torch.testing.assert_close(
                sim.ctrl[:, sim.gripper_act] / 255, action[:, 6:7]
            )
            previous, previous_velocity = actual, velocity
    finally:
        env.close()
