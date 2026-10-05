from dataclasses import replace

import numpy as np
import pytest
from ogbench_mjwarp.config import PlannerConfig, RandomizationConfig
from ogbench_mjwarp.diversity import diverse_selection, measure_diversity
from ogbench_mjwarp.io import write_json
from ogbench_mjwarp.randomization import episode_randomization
from ogbench_mjwarp.skills import SkillPlan
from ogbench_mjwarp.tasks import make_env, task_registry


def test_independent_streams_and_variants():
    config = RandomizationConfig(seed=123, variants_per_reset=2)
    first = episode_randomization(40, 0, config, PlannerConfig())
    second = episode_randomization(40, 1, config, PlannerConfig())
    third = episode_randomization(40, 2, config, PlannerConfig())
    assert first["seeds"]["environment"] == second["seeds"]["environment"] == 40
    assert third["seeds"]["environment"] == 41
    assert len(set(first["seeds"].values())) == len(first["seeds"])
    assert first["seeds"]["path"] != second["seeds"]["path"]
    enabled = episode_randomization(
        40, 0, replace(config, order=True, cube_grasps=True), PlannerConfig()
    )
    assert enabled["seeds"] == first["seeds"]


@pytest.mark.parametrize(
    "values",
    [
        {"variants_per_reset": 0},
        {"position_noise": -1},
        {"yaw_noise": float("nan")},
        {"duration_scale_min": 0},
        {"duration_scale_min": 2, "duration_scale_max": 1},
    ],
)
def test_randomization_validation(values):
    with pytest.raises(ValueError):
        RandomizationConfig(**values)


@pytest.mark.parametrize("env_id", list(task_registry()))
def test_randomized_keyframes_and_provenance(env_id):
    env = make_env(env_id, seed=2026, size=16)
    try:
        config = RandomizationConfig(
            order=True,
            cube_grasps=True,
            position_noise=0.01,
            yaw_noise=0.1,
            duration_scale_min=0.8,
            duration_scale_max=1.2,
        )
        provenance = episode_randomization(2026, 0, config, PlannerConfig())
        skill = SkillPlan(env, 2026, config, provenance)
        refs = skill.references(8)
        assert refs.shape == (8, 5) and np.isfinite(refs).all()
        record = provenance["skills"][0]
        assert record["choices"] and record["upstream_keyframes"]
        times = np.array(record["phase_times"])
        assert np.all(np.diff(times) > 0)
        for draw in record["samples"].get("timing", []):
            assert draw["applied_seconds"] >= draw["minimum_seconds"]
        baseline = {frame["name"]: frame for frame in record["baseline_keyframes"]}
        applied = {
            frame["name"]: frame for frame in record["keyframes_before_workspace_clip"]
        }
        free = {
            name
            for draw in record["samples"].get("path", [])
            for name in draw["keyframes"]
        }
        for name in baseline:
            if name not in free:
                np.testing.assert_array_equal(
                    applied[name]["position"], baseline[name]["position"]
                )
            assert applied[name]["gripper"] == baseline[name]["gripper"]
        for draw in record["samples"].get("path", []):
            assert np.max(np.abs(draw["position_offset"])) <= 0.01
            assert draw["position_offset"][2] >= 0
        annotation = skill.annotation()
        np.testing.assert_array_equal(annotation["annotation/reference"], refs[0])
        repeated = SkillPlan(
            env, 2026, config, episode_randomization(2026, 0, config, PlannerConfig())
        )
        np.testing.assert_array_equal(repeated.references(8), refs)
        skill.advance()
        assert record["end_frame_exclusive"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["drawer", "window"])
def test_handle_symmetry_and_timing_floors(kind):
    env = make_env("scene-v0", size=16)
    try:
        config = RandomizationConfig(
            handle_grasps=True, duration_scale_min=0.5, duration_scale_max=0.5
        )
        provenance = episode_randomization(2026, 0, config, PlannerConfig())
        skill = SkillPlan(env, 2026, config, provenance)
        skill.choose_skill = lambda: (
            kind,
            0,
            getattr(env.unwrapped, f"_target_{kind}_pos"),
        )
        skill.references(8)
        event = provenance["skills"][0]
        assert event["samples"]["grasp"]["symmetry"] in (0, 1)
        assert all(
            draw["applied_seconds"] >= draw["minimum_seconds"]
            for draw in event["samples"]["timing"]
        )
    finally:
        env.close()


def test_diversity_selects_distinct_paths_and_excludes_failures(tmp_path):
    rows = []
    for episode, x in enumerate((0.0, 0.0, 0.2, 0.5)):
        state = np.zeros((5, 18), dtype=np.float32)
        state[:, 12] = x
        row = {
            "episode_id": episode,
            "env_id": "cube-single-v0",
            "task_id": 1,
            "seed": 20,
            "length": 5,
            "fps": 20,
            "outcome": "failure" if episode == 3 else "success",
            "contact_quality": {"valid": True},
            "archive": f"episode-{episode:06d}.npz",
        }
        np.savez(tmp_path / row["archive"], state=state)
        write_json(tmp_path / f"episode-{episode:06d}.json", row)
        rows.append((tmp_path, row))
    selected = diverse_selection(rows, 2)
    assert [row["episode_id"] for _, row in selected] == [0, 2]
    report = measure_diversity(tmp_path)
    assert report["groups"][0]["pairs"] == 3
    assert report["groups"][0]["min_distance"] == 0
    assert report["groups"][0]["max_distance"] > 0


@pytest.mark.gpu
def test_generation_variants_share_reset_and_save_annotations(tmp_path):
    import json

    import torch
    from ogbench_mjwarp.recording import generate

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = RandomizationConfig(variants_per_reset=2, position_noise=0.01)
    generate(
        tmp_path,
        "cube-single-v0",
        2,
        [1],
        2026,
        PlannerConfig(episodes=2, candidates=2, horizon=1, iterations=1),
        size=16,
        max_steps=1,
        record_images=False,
        progress=lambda _: None,
        randomization=config,
    )
    rows = [
        json.loads(path.read_text()) for path in sorted(tmp_path.glob("episode-*.json"))
    ]
    assert rows[0]["seed"] == rows[1]["seed"] == 2026
    assert (
        rows[0]["randomization"]["initial_state_fingerprint"]
        == rows[1]["randomization"]["initial_state_fingerprint"]
    )
    assert (
        rows[0]["randomization"]["seeds"]["path"]
        != rows[1]["randomization"]["seeds"]["path"]
    )
    with (
        np.load(tmp_path / rows[0]["archive"]) as first,
        np.load(tmp_path / rows[1]["archive"]) as second,
    ):
        np.testing.assert_array_equal(first["sim/qpos"][0], second["sim/qpos"][0])
        assert first["annotation/skill_id"].tolist() == [0]
        assert second["annotation/route_id"].tolist() == [1]
        assert first["annotation/reference"].shape == (1, 5)
