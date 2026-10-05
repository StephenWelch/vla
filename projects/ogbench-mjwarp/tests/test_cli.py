from pathlib import Path

import pytest
from ogbench_mjwarp.cli import parse_args


def test_rectangular_camera_defaults_and_overrides(tmp_path):
    assert parse_args(["generate", "--output", "demo"]).size == (480, 640)
    assert parse_args(["generate", "--output", "demo", "--size", "32"]).size == 32
    config = tmp_path / "camera.yaml"
    config.write_text("size: [480, 640]\n")
    args = parse_args(["generate", "--output", "demo", "--config", str(config)])
    assert args.size == (480, 640)


def test_yaml_and_cli_precedence(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(
        "output: runs/demo\nepisodes: 3\ntask_ids: [2, 4]\n"
        "planner:\n  episodes: 2\n  candidates: 8\n  horizon: 4\n  joint_target_noise: 0.02\n"
    )
    args = parse_args(
        [
            "generate",
            "--config",
            str(config),
            "--planner.candidates",
            "16",
            "--task-ids",
            "1",
            "--episodes",
            "5",
            "--planner.joint-target-noise",
            "0.01",
        ]
    )
    assert args.output == Path("runs/demo")
    assert args.episodes == 5
    assert args.task_ids == [1]
    assert args.planner.episodes == 2
    assert args.planner.candidates == 16
    assert args.planner.horizon == 4
    assert args.planner.iterations == 2
    assert args.planner.joint_target_noise == 0.01


def test_yaml_paths_booleans_and_interpolation(tmp_path):
    config = tmp_path / "replay.yaml"
    config.write_text(
        "root: data/demo\nvideo: ${root}/replay.mp4\nrestore_frames: true\n"
    )
    args = parse_args(["replay", f"--config={config}", "--no-restore-frames"])
    assert args.root == Path("data/demo")
    assert args.video == Path("data/demo/replay.mp4")
    assert args.restore_frames is False


def test_nested_randomization_booleans_and_cli_override(tmp_path):
    config = tmp_path / "diverse.yaml"
    config.write_text(
        "randomization:\n  order: false\n  cube_grasps: true\n  variants_per_reset: 3\n"
    )
    args = parse_args(
        [
            "generate",
            "--output",
            "demo",
            "--config",
            str(config),
            "--randomization.no-cube-grasps",
            "--randomization.order",
        ]
    )
    assert args.randomization.order and not args.randomization.cube_grasps
    assert args.randomization.variants_per_reset == 3


@pytest.mark.parametrize(
    "text", ["unknown: 1", "planner:\n  candidates: wrong", "outcome: invalid"]
)
def test_yaml_rejects_unknown_or_invalid_fields(tmp_path, text):
    config = tmp_path / "bad.yaml"
    config.write_text(text)
    with pytest.raises(SystemExit):
        parse_args(["inspect", "--root", "data", "--config", str(config)])


def test_required_fields_and_validation_defaults():
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["generate"])
    args = parse_args(["validate", "--output", "validation"])
    assert args.planner.episodes == 5
    assert args.planner.candidates == 8
    with pytest.raises(SystemExit):
        parse_args(["benchmark", "--planner.candidates", "1"])


def test_yaml_validates_planner_types_and_mapping(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("planner:\n  candidates: wrong")
    with pytest.raises(SystemExit):
        parse_args(["benchmark", "--config", str(config)])
    config.write_text("- not-a-mapping")
    with pytest.raises(TypeError, match="mapping"):
        parse_args(["benchmark", "--config", str(config)])
