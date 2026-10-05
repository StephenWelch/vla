import numpy as np
import pytest
from ogbench_mjwarp.config import PlannerConfig
from ogbench_mjwarp.tasks import (
    cpu_snapshot,
    make_env,
    reset_task,
    restore_cpu,
    solve_puzzle,
    task_description,
    task_registry,
)


def test_config_rejects_invalid_budget():
    with pytest.raises(ValueError):
        PlannerConfig(candidates=1)
    with pytest.raises(ValueError):
        PlannerConfig(horizon=0)


@pytest.mark.parametrize("noise", [-0.01, float("nan"), float("inf")])
def test_config_rejects_invalid_joint_target_noise(noise):
    with pytest.raises(ValueError, match="joint_target_noise"):
        PlannerConfig(joint_target_noise=noise)


def test_joint_target_seeds_and_legacy_snapshots():
    env = make_env("cube-single-v0", seed=17, size=16)
    try:
        nominal = cpu_snapshot(env)
        reset_task(env, 17, 1, joint_target_noise=0.01)
        randomized = cpu_snapshot(env)
        np.testing.assert_array_equal(randomized["qpos"], nominal["qpos"])
        expected = np.random.default_rng(17).uniform(-0.01, 0.01, 6).astype(np.float32)
        np.testing.assert_array_equal(randomized["joint_target_offset"], expected)
        reset_task(env, 18, 1, joint_target_noise=0.01)
        assert not np.array_equal(cpu_snapshot(env)["joint_target_offset"], expected)
        reset_task(env, 17, 1, joint_target_noise=0.01)
        np.testing.assert_array_equal(
            cpu_snapshot(env)["joint_target_offset"], expected
        )
        legacy = {
            key: value for key, value in nominal.items() if key != "joint_target_offset"
        }
        restore_cpu(env, legacy)
        assert not cpu_snapshot(env)["joint_target_offset"].any()
    finally:
        env.close()


@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 4), (4, 5), (4, 6)])
def test_puzzle_solution(rows, cols):
    rng = np.random.default_rng(12)
    current = rng.integers(0, 2, rows * cols, dtype=np.uint8)
    goal = current.copy()
    for i in rng.choice(rows * cols, size=8):
        x, y = divmod(int(i), cols)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < rows and 0 <= b < cols:
                goal[a * cols + b] ^= 1
    for i in solve_puzzle(current, goal, rows, cols):
        x, y = divmod(i, cols)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < rows and 0 <= b < cols:
                current[a * cols + b] ^= 1
    np.testing.assert_array_equal(current, goal)


def test_scene_open_instruction_preserves_numeric_goal():
    env = make_env("scene-v0", task_id=1, size=16)
    try:
        instruction, goal = task_description(env)
        assert instruction == "Open the drawer and window. Leave the cube in place."
        assert goal["task_name"] == "task1_open"
        assert goal["goal"]["drawer_pos"] == -0.16
        assert goal["goal"]["window_pos"] == 0.2
        assert "meters" not in instruction
    finally:
        env.close()


def test_task_registry_and_snapshot():
    assert {"cube-single-v0", "scene-v0", "puzzle-4x6-v0"} <= task_registry().keys()
    env = make_env("cube-single-v0")
    try:
        state = cpu_snapshot(env)
        env.step(np.zeros(5))
        restore_cpu(env, state)
        np.testing.assert_array_equal(env.unwrapped._data.qpos, state["qpos"])
        assert env.unwrapped._model.camera("ur5e/wrist").id >= 0
    finally:
        env.close()


def test_reset_is_reproducible():
    first = make_env("scene-v0", seed=17, task_id=4)
    second = make_env("scene-v0", seed=17, task_id=4)
    try:
        expected, actual = cpu_snapshot(first), cpu_snapshot(second)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    finally:
        first.close()
        second.close()


def test_scene_insertion_goal_follows_open_drawer():
    import mujoco
    from ogbench_mjwarp.skills import SkillPlan

    env = make_env("scene-v0", task_id=4)
    try:
        base = env.unwrapped
        base._cur_button_states[0] = 1
        base._apply_button_states()
        base._data.joint("drawer_slide").qpos[0] = -0.16
        mujoco.mj_forward(base._model, base._data)
        skill = SkillPlan(env, 0)
        kind, index, placement = skill.choose_skill()
        assert kind == "cube" and index == 0
        goal = base._data.mocap_pos[base._cube_target_mocap_ids[0]].copy()
        assert placement[1] > goal[1] + 0.14
        skill.build()
        near_goal = np.argmin(np.linalg.norm(skill.plan[:, :3] - placement, axis=1))
        assert abs(skill.plan[near_goal, 3] - np.pi / 2) < 0.01
        base._data.joint("object_joint_0").qpos[:3] = placement
        mujoco.mj_forward(base._model, base._data)
        assert skill.choose_skill() == ("drawer", 0, 0.0)
        skill.build()
        handle_goal = base._data.site_xpos[base._drawer_site_id].copy()
        handle_goal += base._data.xaxis[base._model.joint("drawer_slide").id] * 0.16
        assert np.min(np.linalg.norm(skill.plan[:, :3] - handle_goal, axis=1)) < 0.005
        np.testing.assert_array_equal(
            base._data.mocap_pos[base._cube_target_mocap_ids[0]], goal
        )
    finally:
        env.close()


def test_button_skill_finishes_at_safe_clearance():
    from ogbench_mjwarp.skills import SkillPlan

    env = make_env("puzzle-3x3-v0")
    try:
        skill = SkillPlan(env, 0)
        skill.build()
        base = env.unwrapped
        index = skill.objective["index"]
        clearance = base._data.site_xpos[base._button_site_ids[index]].copy()
        clearance[2] += 0.06
        assert np.linalg.norm(skill.plan[-1, :3] - clearance) < 0.01
        assert skill.plan[-1, 4] == 1.0
    finally:
        env.close()


@pytest.mark.gpu
def test_ik_and_candidate_isolation():
    import torch
    from ogbench_mjwarp.environment import BatchEnvironment

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = make_env("cube-single-v0")
    try:
        batch = BatchEnvironment(
            env, 2, PlannerConfig(candidates=2, horizon=2, iterations=1)
        )
        action = np.array([0.1, -0.1, 0.1, 0.1, -0.2], dtype=np.float32)
        env.unwrapped.set_control(action)
        batch.set_control(torch.tensor(np.tile(action, (2, 1)), device=batch.device))
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            batch.ctrl[0].cpu(), env.unwrapped._data.ctrl, atol=2e-4
        )
        saved = batch.snapshot()
        actions = torch.tensor(np.array([action, -action]), device=batch.device)
        batch.step(actions)
        qpos = batch.qpos.clone()
        assert not torch.equal(qpos[0], qpos[1])
        batch.restore(saved)
        torch.testing.assert_close(batch.qpos, saved["qpos"], rtol=0, atol=0)
        batch.step(actions)
        torch.testing.assert_close(batch.qpos, qpos, atol=1e-5, rtol=1e-5)
        assert batch.valid().all()
        batch.restore(saved)
        from ogbench_mjwarp.tasks import restore_cpu

        restore_cpu(env, batch.cpu_state(0))
        for _ in range(3):
            env.step(action)
            batch.step(torch.tensor(np.tile(action, (2, 1)), device=batch.device))
            np.testing.assert_allclose(
                batch.qpos[0, batch.arm_q].cpu(),
                env.unwrapped._data.qpos[:6],
                atol=2e-3,
            )
            assert bool(batch.success()[0]) == bool(env.unwrapped._success)
        _, metadata = batch.reset([11, 12], [1, 2])
        assert [row["task_id"] for row in metadata] == [1, 2]
        assert not torch.equal(batch.qpos[0], batch.qpos[1])
    finally:
        env.close()


@pytest.mark.gpu
def test_generation_timeout_and_resume(tmp_path):
    import json

    import torch
    from ogbench_mjwarp.recording import generate

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = PlannerConfig(
        episodes=2, candidates=2, horizon=1, iterations=1, joint_target_noise=0.01
    )
    root = tmp_path / "attempts"
    result = generate(root, "cube-single-v0", 2, [1], 40, config, size=32, max_steps=1)
    assert result["failures"] == 2 and result["frames"] == 2
    for path in sorted(root.glob("episode-*.json")):
        row = json.loads(path.read_text())
        assert row["outcome"] == "failure" and row["reason"] == "timeout"
        expected = (
            np.random.default_rng(row["randomization"]["seeds"]["joint_targets"])
            .uniform(-0.01, 0.01, 6)
            .astype(np.float32)
        )
        np.testing.assert_array_equal(row["joint_target_offset"], expected)
        with np.load(root / row["archive"]) as data:
            assert data["done"].tolist() == [True]
            assert data["truncated"].tolist() == [True]
            assert data["success"].tolist() == [False]
            assert len(data["sim/qpos"]) == len(data["action"]) + 1
            np.testing.assert_array_equal(
                data["sim/joint_target_offset"], np.tile(expected, (2, 1))
            )
    assert (
        generate(root, "cube-single-v0", 2, [1], 40, config, size=32, max_steps=1)
        == result
    )
    from ogbench_mjwarp.dataset import replay

    replayed = replay(root, episode=0, restore_frames=True)
    assert replayed["steps"] == 1 and replayed["outcome_matches"]
    assert replayed["max_qpos_error"] < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("joint_target_noise", [0.0, 0.01])
def test_generation_finishes_cube_release(tmp_path, joint_target_noise):
    import torch
    from ogbench_mjwarp.recording import generate

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = PlannerConfig(
        episodes=1,
        candidates=2,
        horizon=2,
        iterations=1,
        joint_target_noise=joint_target_noise,
    )
    root = tmp_path / "released"
    result = generate(
        root, "cube-single-v0", 1, [1], 2026, config, size=32, max_steps=200
    )
    assert result["successes"] == 1
    with np.load(root / "episode-000000.npz") as data:
        assert data["state"][-1, 16] < 0.1
        np.testing.assert_allclose(
            data["sim/qpos"][-1, -7:-4], data["sim/mocap_pos"][-1, 0], atol=0.015
        )
        assert data["success"][-1] and data["done"][-1]
        assert not data["truncated"][-1]


@pytest.mark.gpu
def test_joint_target_offsets_in_controller_rollouts_and_limits():
    import torch
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.planner import SamplingMPC

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = make_env("cube-single-v0", size=16)
    try:
        config = PlannerConfig(
            episodes=2, candidates=2, horizon=1, iterations=1, joint_target_noise=0.01
        )
        batch = BatchEnvironment(env, 2, config)
        batch.reset([31, 32])
        saved = batch.snapshot()
        offsets = saved["joint_target_offset"]
        actions = torch.zeros((2, 5), device=batch.device)
        batch.fields["joint_target_offset"].zero_()
        batch.set_control(actions)
        nominal = batch.ctrl.clone()
        batch.restore(saved)
        batch.set_control(actions)
        torch.testing.assert_close(
            batch.ctrl[:, batch.arm_act], nominal[:, batch.arm_act] + offsets
        )
        torch.testing.assert_close(
            batch.ctrl[:, batch.gripper_act], nominal[:, batch.gripper_act]
        )
        planner = SamplingMPC(batch, config)
        refs = batch.proprioception()[:, None, 12:17].cpu().numpy()
        _, stats = planner.plan(refs)
        assert stats["valid"] == [True, True]
        torch.testing.assert_close(
            planner.rollouts.fields["joint_target_offset"],
            offsets.repeat_interleave(config.candidates, 0),
        )
        batch.fields["joint_target_offset"].fill_(100)
        batch.set_control(actions)
        torch.testing.assert_close(
            batch.ctrl[:, batch.arm_act], batch.joint_target_bounds[1].expand(2, -1)
        )
        batch.restore(
            {key: value for key, value in saved.items() if key != "joint_target_offset"}
        )
        assert not batch.fields["joint_target_offset"].any()
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.parametrize("env_id", ["scene-v0", "puzzle-3x3-v0"])
def test_button_transitions_and_damping(env_id):
    import torch
    from ogbench_mjwarp.environment import BatchEnvironment

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = make_env(env_id)
    try:
        batch = BatchEnvironment(env, 2, PlannerConfig(candidates=2))
        initial = batch.fields["buttons"].clone()
        previous = batch.qpos[:, batch.button_q].clone()
        batch.qpos[0, batch.button_q[0]] = -0.021
        batch.update_buttons(previous)
        base = env.unwrapped
        base.pre_step()
        base._data.joint("buttonbox_joint_0").qpos[0] = -0.021
        base.post_step()
        np.testing.assert_array_equal(
            batch.fields["buttons"][0].cpu(), base._cur_button_states
        )
        torch.testing.assert_close(batch.fields["buttons"][1], initial[1])
        if batch.is_scene:
            expected_damping = base._model.joint("drawer_slide").damping[0]
            assert batch.fields["dof_damping"][0, batch.drawer_v] == expected_damping
        saved = batch.snapshot()
        batch.fields["buttons"].zero_()
        batch.restore(saved)
        torch.testing.assert_close(batch.fields["buttons"], saved["buttons"])
    finally:
        env.close()


@pytest.mark.gpu
def test_scene_unlock_objective_allows_required_intermediate_state():
    import torch
    from ogbench_mjwarp.environment import BatchEnvironment
    from ogbench_mjwarp.planner import SamplingMPC
    from ogbench_mjwarp.skills import SkillPlan

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Task 2 starts and ends locked, but requires unlocking to move both joints.
    env = make_env("scene-v0", task_id=2)
    try:
        config = PlannerConfig(episodes=1, candidates=2, horizon=1, iterations=1)
        sim = BatchEnvironment(env, 1, config)
        skill = SkillPlan(env, 0)
        refs = skill.references(1)
        assert skill.objective["kind"] == "button"
        planner = SamplingMPC(sim, config)
        goals = planner.stage_goals([skill.objective])
        before = planner.stage_error(goals).clone()
        final_before = planner.rollouts.task_error().clone()
        planner.rollouts.fields["buttons"][:, 0] = 1
        assert (planner.stage_error(goals) < before).all()
        assert (planner.rollouts.task_error() > final_before).all()
        action, stats = planner.plan(refs[None], [skill.objective])
        assert stats["valid"] == [True]
        assert action.isfinite().all() and action.abs().max() <= 1
    finally:
        env.close()
