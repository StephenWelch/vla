from types import SimpleNamespace

import numpy as np
import pytest
from ogbench_mjwarp.contacts import (
    contact_depths,
    contact_roles,
    update_contact_quality,
)
from ogbench_mjwarp.tasks import make_env, task_registry


def test_contact_quality_merges_sources_and_retains_violations():
    quality = {
        "valid": True,
        "peak_nonpad_penetration": 0.0,
        "peak_penetration": 0.0,
        "max_nonpad_penetration": 0.001,
        "max_penetration": 0.003,
    }
    # Exact tolerances are allowed; CPU and GPU can supply different maxima.
    assert update_contact_quality(quality, (0.001, 0.002), (0.0005, 0.003))
    assert quality["peak_nonpad_penetration"] == 0.001
    assert quality["peak_penetration"] == 0.003
    # A terminal-state violation cannot be erased by a later clear frame.
    assert not update_contact_quality(quality, (0.002, 0.002))
    assert not update_contact_quality(quality, (0.0, 0.0))
    assert quality["peak_nonpad_penetration"] == 0.002
    assert quality["peak_penetration"] == 0.003


@pytest.mark.parametrize("env_id", list(task_registry()))
def test_contacts_across_manipulation_models(env_id):
    env = make_env(env_id, size=16)
    try:
        model = env.unwrapped._model
        roles = contact_roles(model)
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        pads = np.flatnonzero((roles == 3) & collidable)
        links = np.flatnonzero((roles == 2) & collidable)
        environment = np.flatnonzero((roles == 0) & collidable)
        assert len(pads) >= 2 and len(links) and len(environment)
        # Link/object contact must not inherit the pad or button exception.
        for pair in ([links[0], environment[-1]], [environment[-1], links[0]]):
            data = SimpleNamespace(contact=[SimpleNamespace(geom=pair, dist=-0.002)])
            nonpad, depth, bodies = contact_depths(model, data, roles)
            assert nonpad == depth == 0.002
            assert len(bodies) == 2
        data = SimpleNamespace(
            contact=[SimpleNamespace(geom=[pads[0], environment[-1]], dist=-0.004)]
        )
        nonpad, depth, _ = contact_depths(model, data, roles)
        assert nonpad == 0 and depth > 0.003
        if getattr(env.unwrapped, "_num_buttons", 0):
            assert np.any((roles == 4) & collidable)
    finally:
        env.close()


@pytest.mark.parametrize(
    "recorded_valid,cpu_valid", [(True, False), (False, True), (True, True)]
)
def test_validation_requires_substep_and_cpu_contact_checks(
    tmp_path, monkeypatch, recorded_valid, cpu_valid
):
    from ogbench_mjwarp import contacts, recording, tasks
    from ogbench_mjwarp.cli import validate_tasks
    from ogbench_mjwarp.config import PlannerConfig
    from ogbench_mjwarp.io import write_json

    def generate(root, *args, **kwargs):
        root.mkdir()
        write_json(
            root / "episode-000000.json",
            {
                "episode_id": 0,
                "task_id": 1,
                "seed": 2026,
                "outcome": "success",
                "contact_quality": {"valid": recorded_valid},
            },
        )
        return {"outcomes": [{"task_id": 1, "outcome": "success"}]}

    monkeypatch.setattr(recording, "generate", generate)
    monkeypatch.setattr(
        tasks,
        "make_env",
        lambda _: SimpleNamespace(
            unwrapped=SimpleNamespace(num_tasks=1), close=lambda: None
        ),
    )
    monkeypatch.setattr(
        contacts, "audit_contacts", lambda *args: {"contact_valid": cpu_valid}
    )
    report = validate_tasks(tmp_path, ["scene-v0"], 1, 2026, 16, PlannerConfig())
    assert report["all_tasks_succeeded"] == (recorded_valid and cpu_valid)
    assert report["all_successes_contact_valid"] == (recorded_valid and cpu_valid)


@pytest.mark.gpu
def test_generation_rejects_cpu_contact_disagreement(tmp_path, monkeypatch):
    import json

    import torch
    from ogbench_mjwarp import recording
    from ogbench_mjwarp.config import PlannerConfig

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # A CPU-only violation must defeat an otherwise healthy GPU execution.
    monkeypatch.setattr(recording, "contact_depths", lambda *args: (0.002, 0.002, None))
    result = recording.generate(
        tmp_path,
        "cube-single-v0",
        1,
        [1],
        2026,
        PlannerConfig(episodes=1, candidates=2, horizon=2, iterations=1),
        size=16,
        max_steps=1,
        progress=lambda _: None,
        record_images=False,
    )
    assert result["outcomes"][0]["reason"] == "contact_violation"
    row = json.loads((tmp_path / "episode-000000.json").read_text())
    assert row["outcome"] == "failure" and not row["contact_quality"]["valid"]
    with np.load(tmp_path / row["archive"]) as archive:
        assert "front" not in archive and "wrist" not in archive
        assert len(archive["sim/qpos"]) == 2


@pytest.mark.parametrize("kind", ["drawer", "window"])
def test_contact_roles_and_handle_plan(kind):
    from ogbench_mjwarp.skills import SkillPlan

    env = make_env("scene-v0", seed=2026, task_id=4 if kind == "drawer" else 1)
    try:
        base = env.unwrapped
        roles = contact_roles(base._model)
        assert roles[base._model.geom("ur5e/robotiq/left_pad1").id] == 3
        assert roles[base._model.geom("drawer_handle").id] == 0
        base._cur_button_states[0] = 1
        base._apply_button_states()
        if kind == "window":
            import mujoco

            base._data.joint("drawer_slide").qpos[0] = -0.16
            mujoco.mj_forward(base._model, base._data)
        skill = SkillPlan(env, 0)
        skill.build()
        assert skill.skill[0] == kind
        # Handle translation while grasped stays below 1cm per control tick.
        moving = skill.plan[skill.plan[:, 4] > 0.99]
        assert len(moving) > 20
        assert np.linalg.norm(np.diff(moving[:, :3], axis=0), axis=1).max() < 0.01
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.parametrize("reverse", [False, True])
def test_gpu_contact_peaks_ignore_padding_and_allow_pads_and_buttons(reverse):
    import torch
    import warp as wp
    from ogbench_mjwarp.environment import collect_contact_depth

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        count = wp.array([4], dtype=wp.int32)
        pairs = np.array([[0, 1], [2, 1], [3, 4], [0, 4], [0, 1]])
        geom = wp.array(pairs[:, ::-1].copy() if reverse else pairs, dtype=wp.vec2i)
        distance = wp.array([-0.002, -0.0003, -0.001, -0.0015, -0.1], dtype=wp.float32)
        world = wp.array([0, 1, 1, 1, 0], dtype=wp.int32)
        roles = wp.array([1, 0, 3, 2, 4], dtype=wp.int32)
        peak = wp.zeros((2, 2), dtype=wp.float32)
        wp.launch(
            collect_contact_depth, 5, inputs=[count, geom, distance, world, roles, peak]
        )
        wp.synchronize()
        np.testing.assert_allclose(peak.numpy(), [[0.002, 0.002], [0.0015, 0.0015]])
        # A later physics substep must preserve the earlier maximum.
        count.assign(np.array([1], dtype=np.int32))
        distance.assign(np.array([-0.0001, 0, 0, 0, 0], dtype=np.float32))
        wp.launch(
            collect_contact_depth, 5, inputs=[count, geom, distance, world, roles, peak]
        )
        wp.synchronize()
        np.testing.assert_allclose(peak.numpy(), [[0.002, 0.002], [0.0015, 0.0015]])
