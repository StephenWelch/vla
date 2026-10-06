"""Spline geometry, reproducibility, execution failures, and dataset contracts."""

import numpy as np
import pytest
from ogbench_mjwarp.config import PlannerConfig, SplineConfig
from ogbench_mjwarp.io import episode_metadata, write_json
from ogbench_mjwarp.spline import fit_path, grasp_samples, pose_matrix, transfer_events


def sample(config=None, seed=123):
    streams = [np.random.default_rng(seed + i) for i in range(3)]
    return grasp_samples(config or SplineConfig(), *streams)


def test_config_and_independent_streams():
    config = PlannerConfig(backend="spline")
    assert config.joint_actions
    assert config == PlannerConfig(**config.to_dict())
    a, b = sample(), sample()
    for x, y in zip(a["candidates"], b["candidates"], strict=True):
        np.testing.assert_array_equal(x["object_to_hand"], y["object_to_hand"])
    c = sample(SplineConfig(grasp_candidates=2))
    np.testing.assert_array_equal(a["duration_scales"], c["duration_scales"])
    np.testing.assert_array_equal(a["path_offset_xyz"], c["path_offset_xyz"])
    assert any(x["tilt_radians"] > 0 for x in a["candidates"])
    with pytest.raises(ValueError):
        SplineConfig(duration_min=(0, 1))
    with pytest.raises(ValueError):
        PlannerConfig(backend="spline", joint_target_noise=0.01)


def test_aggressive_factors_and_persistent_speed():
    config = SplineConfig(
        approach_offset=(0.04, 0.08),
        approach_height_min=(0.10, 0.08),
        approach_height_max=(0.20, 0.23),
        speed_min=(0.5, 0.5),
        speed_max=(0.9, 0.9),
        grasp_selection="random",
    )
    a, b = sample(config), sample(config)
    np.testing.assert_array_equal(a["approach_vectors"], b["approach_vectors"])
    level = int(a["stratum"] == "challenging")
    vectors = a["approach_vectors"]
    assert (
        np.max(np.linalg.norm(vectors[:, :, :2], axis=-1))
        <= config.approach_offset[level]
    )
    assert np.all(vectors[:, :, 2] >= config.approach_height_min[level])
    assert np.all(vectors[:, :, 2] <= config.approach_height_max[level])
    assert 0.5 <= a["execution_speed"] <= 0.9
    events = transfer_events(
        np.array([0.4, 0, 0.02, 1, 0, 0, 0]),
        np.array([0.4, 0.2, 0.06, 1, 0, 0, 0]),
        0,
        a,
        0,
    )
    np.testing.assert_allclose(events[0].pose[:3] - events[1].pose[:3], vectors[0, 0])
    np.testing.assert_allclose(events[5].pose[:3] - events[6].pose[:3], vectors[0, 1])
    path = [np.linspace(np.zeros(6), np.full(6, 0.5), 9)]
    fast = fit_path(path, [1], retiming="local")
    slow = fit_path(path, [1], retiming="local", execution_speed=0.5)
    assert slow[1][-1] >= fast[1][-1] * 1.9
    np.testing.assert_allclose(slow[0][-1], fast[0][-1])
    hold = [np.full((2, 6), 0.2)]
    assert (
        fit_path(hold, [0.4], execution_speed=0.5)[1][-1]
        == fit_path(hold, [0.4])[1][-1]
    )
    with pytest.raises(ValueError):
        SplineConfig(speed_max=(1.1, 1))


def test_collection_coverage_respects_outcome_filters():
    from ogbench_mjwarp.collect import factor_summary

    sampled = sample(SplineConfig(speed_min=(0.6, 0.6)))
    sampled["selected_candidates"] = [0, 1]
    rows = [
        {
            "randomization": {"factors": {"spline": sampled}},
            "quality": {
                "physical_valid": True,
                "completed": True,
                "stable_success": success,
            },
        }
        for success in (True, False)
    ]
    rows.append(
        {
            "randomization": rows[0]["randomization"],
            "quality": {"physical_valid": False},
        }
    )
    report = factor_summary(rows)
    assert report["all"]["episodes"] == 3
    assert report["successes"]["episodes"] == report["failures"]["episodes"] == 1
    assert report["all"]["execution_speed"]["min"] == sampled["execution_speed"]


def test_object_relative_tilt_survives_placement():
    sampled = sample()
    initial = np.array([0.4, -0.1, 0.02, np.cos(0.3), 0, 0, np.sin(0.3)])
    goal = np.array([0.425, 0, 0.06, 1, 0, 0, 0])
    events = transfer_events(initial, goal, 0, sampled, 0)
    offset = pose_matrix(sampled["candidates"][0]["object_to_hand"])
    np.testing.assert_allclose(
        pose_matrix(events[1].pose), pose_matrix(initial) @ offset, atol=1e-10
    )
    placed = pose_matrix(events[6].pose) @ np.linalg.inv(offset)
    np.testing.assert_allclose(placed[:3, :3], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(
        placed[:2, 3], goal[:2] + sampled["placement_offset_xy"][0]
    )
    assert [e.name for e in events if e.dwell] == ["close", "release"]


def test_blending_and_discrete_limits():
    paths = [
        np.linspace(np.zeros(6), np.full(6, 0.2), 8),
        np.linspace(np.full(6, 0.2), np.full(6, 0.4), 8),
    ]
    values, ends, (probe, times), scale = fit_path(paths, [0.2, 0.3])
    padded = np.vstack((values[:1].repeat(3, 0), values, values[-1:].repeat(3, 0)))
    for order, limit in ((1, 1), (2, 2), (3, 50)):
        assert (
            np.max(np.abs(np.diff(padded, n=order, axis=0))) / 0.05**order
            <= limit + 1e-6
        )
    np.testing.assert_allclose(values[-1], 0.4, atol=1e-10)
    boundary = round(ends[0] / 0.05)
    assert np.linalg.norm(values[boundary + 1] - values[boundary - 1]) > 0.001
    assert np.max(np.abs(np.diff(probe, axis=0))) <= 0.01
    assert ends[0] / ends[-1] == pytest.approx(0.4)
    assert times[-1] == pytest.approx(ends[-1])
    assert scale > 1


def test_relaxation_preserves_anchors_and_bounds_the_entire_curve():
    x = np.linspace(0, 1, 17)
    q = np.zeros((17, 6))
    q[:, 0] = x
    q[:, 1] = 0.1 * np.sin(4 * np.pi * x)
    paths = [q[:9], q[8:]]
    before = fit_path(paths, [1, 1])
    stats = {}
    after = fit_path(
        paths, [1, 1], relaxation=0.1, anchors=[True, True], diagnostics=stats
    )
    samples, ends, (probe, times), scale = after
    np.testing.assert_allclose(probe[np.argmin(abs(times - ends[0]))], q[8], atol=1e-9)
    np.testing.assert_allclose(samples[-1], q[-1], atol=1e-9)
    assert (
        stats["nominal_jerk_integral_after"]
        < stats["nominal_jerk_integral_before"] * 0.95
    )
    assert stats["max_joint_deviation_bound"] <= 0.1 + 1e-10
    baseline, stamps = before[2]
    interpolated = np.column_stack(
        [np.interp(times / scale, stamps / before[3], baseline[:, j]) for j in range(6)]
    )
    assert np.max(abs(probe - interpolated)) <= 0.1 + 1e-5
    padded = np.vstack((samples[:1].repeat(3, 0), samples, samples[-1:].repeat(3, 0)))
    for order, limit in ((1, 1), (2, 2), (3, 50)):
        assert (
            np.max(abs(np.diff(padded, n=order, axis=0))) / 0.05**order <= limit + 1e-5
        )


@pytest.mark.parametrize("retiming", ["uniform", "local"])
def test_smoothing_rechecks_obstacles_and_never_returns_invalid_fallback(retiming):
    from ogbench_mjwarp.spline import checked_fit

    x = np.linspace(0, 1, 17)
    q = np.zeros((17, 6))
    q[:, 0] = x
    q[:, 1] = 0.6 * np.sin(np.pi * x)
    paths = [q[:9], q[8:]]

    def collision_check(fitted, radius, record):
        probe = fitted[2][0]
        collision = (
            (probe[:, 0] > 0.45) & (probe[:, 0] < 0.55) & (probe[:, 1] < 0.54)
        ).any()
        return not collision, "obstacle" if collision else None

    fitted, attempts = checked_fit(
        paths,
        [1, 1],
        collision_check,
        relaxation=0.2,
        anchors=[False, True],
        retiming=retiming,
    )
    assert fitted is not None and attempts[-1]["valid"]
    assert any(not a["valid"] and a["reason"] == "obstacle" for a in attempts)
    assert collision_check(fitted, 0, {})[0]
    rejected, attempts = checked_fit(
        paths,
        [1, 1],
        lambda *_: (False, "obstacle"),
        relaxation=0.2,
        anchors=[False, True],
        retiming=retiming,
    )
    assert rejected is None and len(attempts) == 3
    assert attempts[-1]["relaxation_radians"] == 0


def test_transit_guides_are_independent_of_contact_state():
    events = transfer_events(
        np.array([0.4, 0, 0.02, 1, 0, 0, 0]),
        np.array([0.4, 0.2, 0.06, 1, 0, 0, 0]),
        0,
        sample(),
        0,
    )
    assert all(e.anchor for e in events if e.dwell)
    assert not next(e for e in events if e.name == "lift").anchor
    assert next(e for e in events if e.name == "lift").carrying


def test_local_retiming_speeds_up_uneven_paths_with_bounded_derivatives():
    paths = [
        np.linspace(np.zeros(6), np.full(6, 0.5), 9),
        np.linspace(np.full(6, 0.5), np.full(6, 0.55), 9),
    ]

    def speed_constraint(q, tangent):
        return abs(tangent[:, 0]) / 0.12

    baseline = fit_path(paths, [1, 5], event_speed=speed_constraint)
    stats = {}
    values, ends, (probe, times), _ = fit_path(
        paths,
        [1, 5],
        retiming="local",
        diagnostics=stats,
        # Synthetic task-space map: x = q0, e.g. constrained linear motion.
        event_speed=speed_constraint,
    )
    assert ends[-1] < baseline[1][-1] * 0.7
    assert np.all(np.diff(times) > 0)
    assert np.all(np.diff(ends) > 0)
    np.testing.assert_allclose(
        probe[np.argmin(abs(times - ends[0]))], paths[0][-1], atol=1e-9
    )
    np.testing.assert_allclose(values[-1], paths[-1][-1], atol=1e-9)
    assert max(stats["retiming"]["event_speed_ratios"]) <= 1 + 1e-8
    padded = np.vstack((values[:1].repeat(3, 0), values, values[-1:].repeat(3, 0)))
    for n, limit in enumerate((1, 2, 50), 1):
        assert np.max(abs(np.diff(padded, n=n, axis=0))) / 0.05**n <= limit + 1e-6
    assert np.max(abs(np.diff(probe, axis=0))) <= 0.01


def test_local_retiming_preserves_dwells_and_separates_anchor_from_stop():
    q = np.full((2, 6), 0.2)
    values, ends, _, _ = fit_path([q], [0.4], retiming="local")
    assert ends[0] >= 0.4
    np.testing.assert_allclose(values, 0.2)
    events = transfer_events(
        np.array([0.4, 0, 0.02, 1, 0, 0, 0]),
        np.array([0.4, 0.2, 0.06, 1, 0, 0, 0]),
        0,
        sample(SplineConfig(retiming="local")),
        0,
    )
    retreat = events[-1]
    assert retreat.anchor and not retreat.stop
    assert all(e.stop for e in events if e.dwell)
    assert all(e.max_linear_speed == 0.3 for e in events)
    with pytest.raises(ValueError):
        SplineConfig(transit_linear_speed=0)


def test_quality_filters_exclude_unknown_invalid_and_incomplete(tmp_path):
    checks = [
        None,
        (True, True, True),
        (True, True, False),
        (False, True, False),
        (True, False, False),
    ]
    for i, check in enumerate(checks):
        row = {"episode_id": i, "outcome": "success" if i in (0, 1) else "failure"}
        if check:
            row["quality"] = dict(
                zip(
                    ("physical_valid", "completed", "stable_success"),
                    check,
                    strict=True,
                )
            )
        write_json(tmp_path / f"episode-{i:06d}.json", row)
    assert len(episode_metadata(tmp_path)) == 5
    assert [
        r["episode_id"] for r in episode_metadata(tmp_path, quality="validated-success")
    ] == [1]
    assert [
        r["episode_id"] for r in episode_metadata(tmp_path, quality="valid-failure")
    ] == [2]
    with pytest.raises(ValueError):
        episode_metadata(tmp_path, quality="unknown")


def test_cli_overrides_yaml(tmp_path):
    from ogbench_mjwarp.cli import parse_args

    config = tmp_path / "spline.yaml"
    config.write_text(
        "planner:\n  backend: spline\n  spline:\n    variation: nominal\n"
    )
    args = parse_args(
        [
            "generate",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "out"),
            "--planner.spline.variation",
            "mixed",
            "--planner.spline.retiming",
            "local",
            "--planner.spline.transit-linear-speed",
            "0.2",
        ]
    )
    assert args.planner.backend == "spline"
    assert args.planner.spline.variation == "mixed"
    assert args.planner.spline.retiming == "local"
    assert args.planner.spline.transit_linear_speed == 0.2


def test_missed_grasp_and_drop_are_nonterminal():
    from types import SimpleNamespace

    import torch
    from ogbench_mjwarp.spline_planner import SplineSkill

    skill = SplineSkill.__new__(SplineSkill)
    skill.phases = transfer_events(
        np.array([0.4, 0, 0.02, 1, 0, 0, 0]),
        np.array([0.4, 0.2, 0.06, 1, 0, 0, 0]),
        0,
        sample(),
        0,
    )
    skill.objects = [np.array([0.4, 0, 0.02, 1, 0, 0, 0])]
    skill.goals = [np.array([0.4, 0.2, 0.06, 1, 0, 0, 0])]
    skill.observed, skill.had_grasp, skill.measured_grasps = [], set(), {}
    skill.phase_index, skill.last_observed_phase, skill.elapsed = 4, -1, 50
    skill.failure = None
    sim = SimpleNamespace(
        qpos=torch.tensor([[0.4, 0, 0.02, 1, 0, 0, 0]]),
        cube_q=[0],
        effector=torch.tensor([[0.4, 0, 0.2]]),
        site_rot=torch.eye(3)[None, None],
        pinch=0,
    )
    skill.observe(sim, 0)
    assert skill.observed[-1]["event"] == "grasp" and not skill.observed[-1]["observed"]
    assert skill.failure is None
    skill.had_grasp.add(0)
    skill.observe(sim, 0)
    assert skill.observed[-1]["failure"] == "object_dropped"
    assert skill.failure is None


def test_prescribed_splits_survive_separate_outcome_exports(tmp_path):
    from ogbench_mjwarp.hooks import scene_split

    rows = [
        {
            "episode_index": i,
            "env_id": "cube-double-v0",
            "task_id": 5,
            "seed": i // 2,
            "dataset_split": "val" if i // 2 == 0 else "train",
        }
        for i in range(4)
    ]
    write_json(tmp_path / "manifest.json", {"episodes": rows})
    result = scene_split(tmp_path)
    assert [r["episode_index"] for r in result["val"]] == [0, 1]
    assert [r["episode_index"] for r in result["train"]] == [2, 3]
    rows[1]["dataset_split"] = "train"
    write_json(tmp_path / "manifest.json", {"episodes": rows})
    with pytest.raises(ValueError, match="consistently"):
        scene_split(tmp_path)


@pytest.mark.dataset
def test_spline_annotations_and_quality_roundtrip(tmp_path):
    pytest.importorskip("lerobot")
    from ogbench_mjwarp.dataset import export_dataset, load_dataset
    from ogbench_mjwarp.recording import EpisodeBuffer

    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(2):
        buffer = EpisodeBuffer(
            raw,
            i,
            {
                "instruction": "Stack the red cube on the blue cube.",
                "image_size": 32,
                "fps": 20,
                "rendering": {
                    "backend": "mujoco-warp",
                    "revision": 2,
                    "resolution": [32, 32],
                },
                "env_id": "cube-double-v0",
                "task_id": 5,
                "seed": i,
                "annotation_schema_version": 2,
                "randomization": {"available": True, "schema_version": 2},
                "quality": {
                    "physical_valid": True,
                    "completed": True,
                    "stable_success": i == 0,
                },
            },
        )
        for tick in range(3):
            buffer.states.append({"qpos": np.array([tick])})
            buffer.add(
                {v: np.zeros((32, 32, 3), np.uint8) for v in ("front", "wrist")},
                np.zeros(18, np.float32),
                np.zeros(5, np.float32),
                i == 0 and tick == 2,
                tick == 2,
                False,
                0.0,
                {
                    "annotation/skill_id": 0,
                    "annotation/phase_id": tick,
                    "annotation/route_id": 0,
                    "annotation/reference": np.zeros(5, np.float32),
                    "annotation/target_pose": np.array(
                        [0.4, 0, 0.2, 0, 1, 0, 0], np.float32
                    ),
                    "annotation/task_error": i == 1,
                },
            )
        buffer.save("success" if i == 0 else "failure", "test", {"qpos": np.array([3])})
    for i, quality in enumerate(("validated-success", "valid-failure")):
        output = tmp_path / quality
        assert export_dataset(raw, output, quality=quality)["episodes"] == 1
        dataset = load_dataset(output, quality=quality)
        assert dataset[0]["annotation.target_pose"].shape == (7,)
        assert dataset[0]["annotation.task_error"].item() == bool(i)
        with np.load(output / "replay/episode-000000.npz") as data:
            assert len(data["sim/qpos"]) == 4
