"""Evaluation validation, partial-batch reporting, and interruption cleanup."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("lerobot")
from ogbench_mjwarp import evaluate as evaluation


@pytest.fixture(params=[(16, 16), (480, 640)])
def config(tmp_path, request):
    dataset, checkpoint = tmp_path / "dataset", tmp_path / "policy"
    (dataset / "meta").mkdir(parents=True)
    checkpoint.mkdir()
    cameras = {
        f"observation.images.{view}": {"shape": [3, *request.param], "type": "VISUAL"}
        for view in ("front", "wrist")
    }
    (dataset / "meta/info.json").write_text(json.dumps({"features": cameras}))
    profile = {
        "backend": "mujoco-warp",
        "revision": 2,
        "resolution": list(request.param),
    }
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ogbench-mjwarp-2",
                "rendering": profile,
                "episodes": [{"seed": 7}],
            }
        )
    )
    (checkpoint / "rendering.json").write_text(json.dumps(profile))
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {"observation.state": {"shape": [18]}, **cameras},
                "output_features": {"action": {"shape": [5]}},
            }
        )
    )
    (checkpoint / "model.safetensors").write_bytes(b"dummy")
    (checkpoint / "policy_preprocessor.json").write_text(json.dumps({"steps": []}))
    return evaluation.EvalConfig(
        checkpoint, tmp_path / "run", dataset, episodes=3, batch_size=2
    )


@pytest.mark.parametrize(
    "field,value",
    [("episodes", 0), ("task_ids", (1, 1)), ("device", "cpu"), ("videos", -1)],
)
def test_invalid_config_fails_before_loading(config, monkeypatch, field, value):
    setattr(config, field, value)
    loader = Mock()
    monkeypatch.setattr(evaluation, "load_policy", loader)
    with pytest.raises(ValueError):
        evaluation.evaluate(config)
    loader.assert_not_called()
    assert not config.output.exists()


def test_faulty_renderer_rejected_before_loading(config, monkeypatch):
    path = config.dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rendering"].pop("revision")
    path.write_text(json.dumps(manifest))
    loader = Mock()
    monkeypatch.setattr(evaluation, "load_policy", loader)
    with pytest.raises(ValueError, match="faulty renderer"):
        evaluation.evaluate(config)
    loader.assert_not_called()


def test_camera_mismatch_rejected(config):
    path = config.dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["observation.images.wrist"]["shape"] = [3, 32, 32]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="matching RGB"):
        evaluation.prepare_evaluation(config)


def test_saved_camera_mapping_validated(config):
    path = config.checkpoint / "config.json"
    policy = json.loads(path.read_text())
    policy["input_features"]["observation.images.camera1"] = policy[
        "input_features"
    ].pop("observation.images.front")
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="camera mapping"):
        evaluation.prepare_evaluation(config)
    (config.checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {
                            "rename_map": {
                                "observation.images.front": "observation.images.camera1"
                            }
                        },
                    }
                ]
            }
        )
    )
    env, report = evaluation.prepare_evaluation(config)
    expected = tuple(
        json.loads((config.checkpoint / "rendering.json").read_text())["resolution"]
    )
    assert env.image_size == expected and report["training_reset_seeds"] == [7]


def test_model_load_interruption_records_failure(config, monkeypatch):
    monkeypatch.setattr(evaluation, "load_policy", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate(config)
    report = json.loads((config.output / "eval_info.json").read_text())
    assert report["status"] == "failed_or_interrupted" and report["tasks"] == {}
    assert report["config"]["checkpoint"] == str(config.checkpoint)


def test_task_records_reordered_and_partial_batch_trimmed(config, monkeypatch):
    import lerobot.envs
    import lerobot.scripts.lerobot_eval
    from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig

    records = [
        {"seed": seed, "task_success": True, "contact_valid": seed != 8}
        for seed in (8, 7, 10, 9)
    ]
    env = SimpleNamespace(records=records, close=Mock())
    monkeypatch.setattr(
        lerobot.envs, "make_env", Mock(return_value={config.env: {1: env}})
    )
    monkeypatch.setattr(
        lerobot.envs, "make_env_pre_post_processors", Mock(return_value=(None, None))
    )
    monkeypatch.setattr(
        lerobot.scripts.lerobot_eval,
        "eval_policy",
        Mock(
            return_value={
                "per_episode": [{"seed": seed} for seed in (7, 8, 9)],
                "aggregated": {},
            }
        ),
    )
    metrics = evaluation.evaluate_task(
        config, OGBenchEnvConfig(), SimpleNamespace(config=None), None, None, 1
    )
    assert [row["seed"] for row in metrics["per_episode"]] == [7, 8, 9]
    assert metrics["aggregated"]["pc_contact_valid"] == pytest.approx(200 / 3)
    env.close.assert_called_once()


def test_task_failure_closes_environment(config, monkeypatch):
    import lerobot.envs
    from ogbench_mjwarp.lerobot_env import OGBenchEnvConfig

    env = SimpleNamespace(close=Mock())
    monkeypatch.setattr(
        lerobot.envs, "make_env", Mock(return_value={config.env: {1: env}})
    )
    monkeypatch.setattr(
        lerobot.envs,
        "make_env_pre_post_processors",
        Mock(side_effect=RuntimeError("processor failed")),
    )
    with pytest.raises(RuntimeError, match="processor failed"):
        evaluation.evaluate_task(
            config, OGBenchEnvConfig(), SimpleNamespace(config=None), None, None, 1
        )
    env.close.assert_called_once()
