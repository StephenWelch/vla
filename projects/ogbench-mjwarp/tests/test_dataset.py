import json

import numpy as np
import pytest
from ogbench_mjwarp.recording import EpisodeBuffer


@pytest.mark.dataset
@pytest.mark.parametrize("size", [32, (480, 640)])
def test_lerobot_roundtrip_and_outcome_filter(tmp_path, size):
    pytest.importorskip("lerobot")
    from ogbench_mjwarp.dataset import export_dataset, load_dataset
    from ogbench_mjwarp.tasks import image_shape

    raw = tmp_path / "raw"
    raw.mkdir()
    for episode, outcome in enumerate(("success", "failure")):
        buffer = EpisodeBuffer(
            raw,
            episode,
            {
                "instruction": "Move the red cube.",
                "image_size": size,
                "fps": 20,
                "rendering": {
                    "backend": "mujoco-warp",
                    "revision": 2,
                    "resolution": list(image_shape(size)),
                },
                "env_id": "cube-single-v0",
                "task_id": 1,
                "seed": episode,
                "contact_quality": {"valid": outcome == "success"},
            },
        )
        image = np.zeros((*image_shape(size), 3), dtype=np.uint8)
        if episode == 0:
            buffer.metadata["randomization"] = {
                "schema_version": 1,
                "available": True,
                "skills": [
                    {
                        "skill_id": 0,
                        "kind": "cube",
                        "index": 0,
                        "phase_names": ["pick"],
                        "samples": {"path": [{"position_offset": [0.01, 0, 0]}]},
                    }
                ],
            }
        image[..., episode] = 180
        for tick in range(3):
            buffer.states.append({"qpos": np.array([tick], dtype=np.float32)})
            buffer.add(
                {"front": image, "wrist": image},
                np.full(18, tick, dtype=np.float32),
                np.full(5, tick + episode * 10, dtype=np.float32),
                outcome == "success" and tick == 2,
                tick == 2,
                outcome == "failure" and tick == 2,
                0.01,
                {
                    "annotation/skill_id": 0,
                    "annotation/phase_id": 0,
                    "annotation/route_id": 2,
                    "annotation/reference": np.array(
                        [0.4, 0.1, 0.2, 0, 1], dtype=np.float32
                    ),
                }
                if episode == 0
                else None,
            )
        buffer.save(
            outcome,
            "success" if outcome == "success" else "timeout",
            {"qpos": np.array([3], dtype=np.float32)},
        )
    output = tmp_path / "dataset"
    result = export_dataset(raw, output)
    assert result["episodes"] == 2 and result["frames"] == 6
    dataset = load_dataset(output, chunk_length=4)
    assert len(dataset) == 6
    item = dataset[2]
    assert item["action"].shape == (4, 5)
    assert item["action"][0, 0] == 2
    assert (item["action"][:, 0] == 2).all()  # End padding cannot leak next episode.
    assert item["action_is_pad"].tolist() == [False, True, True, True]
    assert dataset[0]["observation.images.front"].shape == (3, *image_shape(size))
    assert dataset[0]["annotation.available"].item()
    assert dataset[0]["annotation.route_id"].item() == 2
    assert not dataset[3]["annotation.available"].item()
    assert dataset[3]["annotation.skill_id"].item() == -1
    assert load_dataset(output, outcome="success").num_episodes == 1
    assert load_dataset(output, outcome="failure").num_episodes == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["episodes"][0]["randomization"]["skills"][0]["samples"]["path"][0][
        "position_offset"
    ] == [0.01, 0, 0]
    assert manifest["episodes"][1]["randomization"]["available"] is False
    with np.load(
        output / manifest["episodes"][0]["replay"], allow_pickle=False
    ) as replay:
        np.testing.assert_array_equal(replay["sim/qpos"].ravel(), [0, 1, 2, 3])
        assert replay["annotation/route_id"].tolist() == [2, 2, 2]
    with pytest.raises(ValueError):
        export_dataset(raw, output)
    # Different task runs may share source episode IDs; export must remap them.
    combined = tmp_path / "combined"
    export_dataset([raw, raw], combined)
    combined_manifest = json.loads((combined / "manifest.json").read_text())
    assert [r["episode_index"] for r in combined_manifest["episodes"]] == [0, 1, 2, 3]
    assert len(load_dataset(combined, chunk_length=1)) == 12
    assert load_dataset(output, require_contact_valid=True).num_episodes == 1
    filtered = export_dataset(
        raw, tmp_path / "contact-valid", require_contact_valid=True
    )
    assert filtered["episodes"] == 1 and filtered["frames"] == 3
    selected = export_dataset(raw, tmp_path / "diverse", diverse_per_task=1)
    assert selected["episodes"] == 1
    # A legacy success without substep evidence must not enter strict selections.
    manifest["episodes"][0].pop("contact_quality")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="No episodes match"):
        load_dataset(output, outcome="success", require_contact_valid=True)
    for path in raw.glob("episode-*.json"):
        row = json.loads(path.read_text())
        row.pop("contact_quality")
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="No completed nonempty episodes"):
        export_dataset(raw, tmp_path / "legacy", require_contact_valid=True)
    row["record_images"] = False
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="requires recorded images"):
        export_dataset(raw, tmp_path / "image-free")
