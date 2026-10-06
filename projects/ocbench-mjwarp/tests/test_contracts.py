import json

import pytest
from ocbench_mjwarp.config import ACTION, CollectionConfig, validate_action
from ocbench_mjwarp.profile import profiles


def test_absolute_targets_rejected_despite_matching_shape():
    with pytest.raises(ValueError, match="action contract"):
        validate_action({"name": "ogbench-joint-target", "shape": [7]})


def test_checkpoint_camera_and_action_identity(tmp_path):
    dataset, checkpoint = tmp_path / "dataset", tmp_path / "checkpoint"
    dataset.mkdir()
    checkpoint.mkdir()
    manifest = {
        "format": "ocbench-mjwarp-1",
        "action_profile": ACTION,
        "rendering": {"model_sha256": "a"},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (checkpoint / "action_profile.json").write_text(json.dumps(ACTION))
    (checkpoint / "rendering.json").write_text(json.dumps({"model_sha256": "b"}))
    with pytest.raises(ValueError, match="rendering"):
        profiles(dataset, checkpoint)
    (checkpoint / "rendering.json").write_text(json.dumps(manifest["rendering"]))
    assert profiles(dataset, checkpoint) == (manifest["rendering"], ACTION)
    (dataset / "INCOMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete"):
        profiles(dataset)


def test_unsupported_task_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="supports"):
        CollectionConfig(tmp_path, task="block-single-task1-v0")


@pytest.mark.gpu
def test_contact_instrumentation_preserves_native_physics():
    import numpy as np
    import warp as wp
    from ocbench_mjwarp.collect import quality
    from ocbench_mjwarp.environment import Simulation

    native, audited = Simulation([82001], audit=False), Simulation([82001])
    try:
        action = wp.array(
            np.zeros((1, 7), np.float32), dtype=float, device=native.warp_device
        )
        for _ in range(5):
            native.env.step_joint_actions_gpu(action)
            audited.env.step_joint_actions_gpu(action)
        np.testing.assert_allclose(
            native.data.qpos.numpy(), audited.data.qpos.numpy(), atol=1e-6, rtol=1e-6
        )
        assert audited.physics_status()[0].all()
        assert quality(True, True, [0.0005, 0.002])
        assert not quality(True, True, [0.002, 0.002])
        assert not quality(True, False, [0, 0])
    finally:
        native.close()
        audited.close()
