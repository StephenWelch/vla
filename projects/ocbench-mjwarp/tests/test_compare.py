import json

import pytest
from ocbench_mjwarp.compare import selections
from vla_tools.hooks import scene_split


def test_shared_holdout_keeps_failures_out_of_success_training(tmp_path):
    rows = [
        {
            "episode_id": i,
            "episode_index": i,
            "source_root": "raw",
            "physical_valid": True,
            "randomization": {"plans": [{"num_pick_retries": 0}]},
            "native_success": success,
            "dataset_split": split,
            "seed": 100 + i,
            "env_id": "stack",
            "task_id": 2,
        }
        for i, (success, split) in enumerate(
            [(True, "train"), (False, "train"), (True, "val"), (False, "val")]
        )
    ]
    manifest = {"episodes": rows}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    train, val = selections(manifest)
    assert train == {"successes": [0], "all": [0, 1]}
    assert val == [2, 3]
    splits = {
        name: scene_split(tmp_path, episodes=ids + val) for name, ids in train.items()
    }
    assert splits["successes"]["val"] == splits["all"]["val"]
    assert all(r["native_success"] for r in splits["successes"]["train"])
    assert not set(val) & set(train["all"])
    rows.append(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        selections(manifest)


def test_explicit_eval_seeds_yaml_roundtrip(tmp_path):
    from ocbench_mjwarp.train import TrainConfig
    from vla_tools.config import parse_args

    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(tmp_path / "dataset"),
                "policy": None,
                "output": str(tmp_path / "run"),
                "validation_fraction": 0.2,
                "image_normalization": "imagenet",
                "train_eval_seeds": [1, 2],
                "val_eval_seeds": [3, 4],
            }
        )
    )
    parsed = parse_args(TrainConfig, ["--config", str(config)])
    assert parsed.train_eval_seeds == [1, 2]
    assert parsed.val_eval_seeds == [3, 4]
    assert parsed.image_normalization == "imagenet"


def test_retry_filter_preserves_all_data_and_holdout():
    rows = [
        {
            "episode_id": i,
            "episode_index": i,
            "source_root": "raw",
            "physical_valid": True,
            "native_success": success,
            "dataset_split": split,
            "randomization": {"plans": [{"num_pick_retries": n} for n in retries]},
        }
        for i, (success, split, retries) in enumerate(
            [
                (True, "train", [0, 0]),
                (True, "train", [0, 1, 0]),
                (False, "train", [0]),
                (True, "val", [1]),
                (False, "val", [0]),
            ]
        )
    ]
    manifest = {"episodes": rows}
    assert selections(manifest) == ({"successes": [0], "all": [0, 1, 2]}, [3, 4])
    assert selections(manifest, False)[0]["successes"] == [0, 1]
    rows[0]["randomization"]["plans"] = []
    with pytest.raises(ValueError, match="annotations"):
        selections(manifest)

    rows[0]["randomization"]["plans"] = [{}]
    with pytest.raises(ValueError, match="annotations"):
        selections(manifest)


def test_clean_successes_exclude_high_level_mistakes_and_keep_holdout():
    rows = [
        {
            "episode_id": i,
            "episode_index": i,
            "source_root": "raw",
            "physical_valid": True,
            "native_success": True,
            "dataset_split": "train" if i < 3 else "val",
            "randomization": {
                "plans": [{"num_pick_retries": retry, "is_mistake": mistake}]
            },
        }
        for i, (retry, mistake) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])
    ]
    train, val = selections({"episodes": rows}, exclude_mistakes=True)
    assert train == {"successes": [0], "all": [0, 1, 2]}
    assert val == [3]
    del rows[0]["randomization"]["plans"][0]["is_mistake"]
    with pytest.raises(ValueError, match="mistake annotations"):
        selections({"episodes": rows}, exclude_mistakes=True)


def test_overfit_requires_clean_training_episode(tmp_path, monkeypatch):
    from ocbench_mjwarp import profile
    from ocbench_mjwarp.config import ACTION
    from ocbench_mjwarp.train import TrainConfig
    from ocbench_mjwarp.training import validate_profiles

    monkeypatch.setattr(profile, "profiles", lambda *args: ({}, ACTION))
    row = {
        "episode_index": 1,
        "native_success": True,
        "physical_valid": True,
        "dataset_split": "train",
        "randomization": {"plans": [{"num_pick_retries": 0, "is_mistake": 0}]},
    }

    def save():
        (tmp_path / "manifest.json").write_text(
            json.dumps({"action_profile": ACTION, "episodes": [row]})
        )

    cfg = TrainConfig(
        dataset=tmp_path, output=tmp_path / "run", episodes=[1], overfit=True
    )
    save()
    validate_profiles(cfg)
    row["randomization"]["plans"][0]["num_pick_retries"] = 1
    save()
    with pytest.raises(ValueError, match="clean audited"):
        validate_profiles(cfg)
    row["randomization"]["plans"][0]["num_pick_retries"] = 0
    row["dataset_split"] = "val"
    save()
    with pytest.raises(ValueError, match="clean audited"):
        validate_profiles(cfg)
