import json
from types import SimpleNamespace

import numpy as np
import pytest
from ocbench_mjwarp.actions import (
    absolute_targets,
    prepare_training_views,
    set_absolute_gripper,
)


def test_absolute_targets_match_native_command():
    state = np.zeros((4, 18), np.float32)
    state[:, 16] = [0.4, 0.95, 0.05, -0.01]
    action = np.ones((4, 7), np.float32)
    action[2:, 6] = -1
    converted = absolute_targets(action, state)
    np.testing.assert_allclose(converted[:, 6], [0.52, 1, 0, 0])
    np.testing.assert_array_equal(converted[:, :6], action[:, :6])
    assert action[0, 6] == 1


def test_absolute_arm_targets_use_current_position_and_actuator_limits():
    from ocbench_mjwarp.config import ARM_LIMITS

    state = np.zeros((2, 18), np.float32)
    state[0, :6] = np.asarray(ARM_LIMITS) - 0.01
    state[1, :6] = -np.asarray(ARM_LIMITS) + 0.01
    action = np.ones((2, 7), np.float32)
    action[1, :6] = -1
    targets = absolute_targets(action, state, absolute_arm=True)
    np.testing.assert_allclose(targets[0, :6], ARM_LIMITS)
    np.testing.assert_allclose(targets[1, :6], -np.asarray(ARM_LIMITS))
    state[:, :6] = 0.5
    targets = absolute_targets(action, state, absolute_arm=True)
    np.testing.assert_allclose(targets[0, :6], [0.68, 0.68, 0.68, 0.86, 0.86, 0.86])
    np.testing.assert_allclose(
        targets[1, :6], [0.32, 0.32, 0.32, 0.14, 0.14, 0.14], atol=1e-7
    )


def test_training_views_fit_train_only_and_preserve_sources():
    from datasets import Dataset
    from lerobot.datasets.io_utils import hf_transform_to_torch

    class View:
        @property
        def hf_dataset(self):
            return self.reader.hf_dataset

    views = {}
    sources = {}
    for name, offset in (("train", 0), ("val", 100)):
        state = np.zeros((100, 18), np.float32)
        state[:, 0] = np.arange(100) + offset
        state[:, 16] = np.linspace(0, 1, 100)
        action = np.zeros((100, 7), np.float32)
        action[:, 0] = np.linspace(-1, 1, 100)
        action[:, 6] = 1
        sources[name] = Dataset.from_dict(
            {"action": action.tolist(), "observation.state": state.tolist()}
        )
        sources[name].set_transform(hf_transform_to_torch)
        views[name] = View()
        views[name].reader = SimpleNamespace(hf_dataset=sources[name])
    stats = {"action": {}, "observation.state": {}}
    prepare_training_views(views, stats, True, True)
    np.testing.assert_allclose(stats["observation.state"]["q01"][0], 0.99)
    np.testing.assert_allclose(stats["observation.state"]["q99"][0], 98.01)
    assert stats["action"]["q01"][6] == 0
    assert stats["action"]["q99"][6] == 1
    # Every future frame has its own target, before action chunk sampling.
    np.testing.assert_allclose(
        np.stack(views["train"].hf_dataset[:2]["action"])[:, 6], [0.12, 0.12 + 1 / 99]
    )
    assert sources["train"][0]["action"][6] == 1
    assert views["val"].hf_dataset[0]["observation.state"][0] == 100


def test_absolute_control_clamps_and_preserves_finished_worlds():
    import warp as wp

    action = wp.array(
        np.array([[0] * 6 + [1.2], [0] * 6 + [-1], [0] * 6 + [1]], np.float32),
        dtype=float,
        device="cpu",
    )
    ctrl = wp.array(np.full((3, 2), 42, np.float32), dtype=float, device="cpu")
    wp.launch(
        set_absolute_gripper,
        dim=3,
        inputs=[
            action,
            wp.array([0, 0, 1], dtype=int, device="cpu"),
            wp.array([1], dtype=int, device="cpu"),
            ctrl,
            1,
        ],
        device="cpu",
    )
    np.testing.assert_allclose(ctrl.numpy(), [[42, 255], [42, 0], [42, 42]])


def test_checkpoint_profile_selects_matching_execution(tmp_path):
    from ocbench_mjwarp.config import ABSOLUTE_ACTION, ABSOLUTE_GRIPPER_ACTION, ACTION
    from ocbench_mjwarp.profile import profiles

    dataset, checkpoint = tmp_path / "data", tmp_path / "checkpoint"
    dataset.mkdir()
    checkpoint.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ocbench-mjwarp-1",
                "rendering": {"revision": 2},
                "action_profile": ACTION,
            }
        )
    )
    (checkpoint / "rendering.json").write_text(json.dumps({"revision": 2}))
    action_path = checkpoint / "action_profile.json"
    for profile in (ACTION, ABSOLUTE_GRIPPER_ACTION, ABSOLUTE_ACTION):
        action_path.write_text(json.dumps(profile))
        assert profiles(dataset, checkpoint)[1] == profile
    action_path.write_text(json.dumps(ABSOLUTE_GRIPPER_ACTION | {"fps": 25}))
    with pytest.raises(ValueError, match="action_profile"):
        profiles(dataset, checkpoint)


def test_offset_quantiles_exclude_padding_and_validation():
    from datasets import Dataset

    class View:
        def __init__(self, values, frames):
            self.hf_dataset = Dataset.from_dict(
                {
                    "action": np.repeat(
                        np.asarray(values)[:, None], 7, axis=1
                    ).tolist(),
                    "observation.state": np.zeros((len(values), 18)).tolist(),
                    "frame_index": frames,
                }
            )
            self.reader = SimpleNamespace(delta_indices={"action": [0, 1, 2]})

    views = {
        "train": View([0, 1, 2, 3, 100, 101, 102], [0, 1, 2, 3, 0, 1, 2]),
        "val": View([999, 999], [0, 1]),
    }
    stats = {"action": {}, "observation.state": {}}
    prepare_training_views(views, stats, False, True, per_timestep=True)
    for h, values in enumerate(
        ([0, 1, 2, 3, 100, 101, 102], [1, 2, 3, 101, 102], [2, 3, 102])
    ):
        np.testing.assert_allclose(stats["action"]["q01"][h], np.quantile(values, 0.01))
        np.testing.assert_allclose(stats["action"]["q99"][h], np.quantile(values, 0.99))
    np.testing.assert_array_equal(stats["action"]["offset_count"], [7, 5, 3])
