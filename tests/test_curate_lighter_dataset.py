"""Selection tests for the lighter demonstration curation script."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "curate-lighter-dataset.py"
SPEC = importlib.util.spec_from_file_location("curate_lighter_dataset", SCRIPT)
CURATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURATE)


class CurationTests(unittest.TestCase):
    def test_successful_recoveries_are_kept_and_failures_excluded(self):
        annotations = []
        for index in range(70):
            annotations.append({
                "episode_index": index,
                "outcome": "success" if index < 60 else "failure",
                "mistake": 50 <= index < 65,
                "position_id": "ABCDE"[index % 5],
            })
        splits = CURATE.choose_splits(annotations)
        train, val = splits["train"], splits["val"]
        self.assertEqual((len(train), len(val)), (48, 12))
        self.assertEqual(sum(row["mistake"] for row in train), 8)
        self.assertEqual(sum(row["mistake"] for row in val), 2)
        self.assertEqual({row["position_id"] for row in val if not row["mistake"]}, set("ABCDE"))
        self.assertTrue(all(row["outcome"] == "success" for row in train + val))
        self.assertEqual(len({row["episode_index"] for row in train + val}), 60)

    def test_unpositioned_old_success_does_not_displace_collected_episode(self):
        rows = [{"episode_index": 0, "outcome": "success", "mistake": True}]
        rows += [{"episode_index": index + 1, "outcome": "success",
                  "mistake": index >= 50, "position_id": "ABCDE"[index % 5]}
                 for index in range(60)]
        splits = CURATE.choose_splits(rows)
        self.assertNotIn(0, {row["episode_index"] for split in splits.values() for row in split})

    def test_curated_task_matches_frames_and_preserves_raw_tasks(self):
        from lerobot.datasets.dataset_tools import split_dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            key: {"dtype": "float32", "shape": (1,), "names": ["joint"]}
            for key in ("observation.state", "action")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = LeRobotDataset.create(
                repo_id="test/lighter", fps=30, features=features,
                root=root / "raw", robot_type="test", use_videos=False,
            )
            for task in ("", CURATE.TASK, CURATE.TASK):
                source.add_frame({"observation.state": np.array([1], dtype=np.float32),
                                  "action": np.array([2], dtype=np.float32), "task": task})
                source.save_episode()
            source.finalize()
            source = LeRobotDataset("test/lighter", root=root / "raw")
            selections = {"train": [{"episode_index": 0}, {"episode_index": 1}],
                          "val": [{"episode_index": 2}]}
            CURATE.check_selected_tasks(source, selections)
            split_dataset(source, {"train": [0, 1], "val": [2]}, output_dir=root / "curated")
            for name in ("train", "val"):
                split_root = root / "curated" / name
                CURATE.normalize_split_task(split_root)
                dataset = LeRobotDataset(f"test/lighter_{name}", root=split_root)
                CURATE.verify_split_task(dataset)
                for episode in dataset.meta.episodes:
                    self.assertEqual(episode["tasks"], [CURATE.TASK])
                for frame_index in range(len(dataset)):
                    self.assertEqual(dataset[frame_index]["task_index"].item(), 0)
            self.assertEqual(list(source.meta.tasks.index), ["", CURATE.TASK])
            self.assertEqual(source.meta.episodes[0]["tasks"], [""])

    def test_conflicting_recorded_task_is_rejected(self):
        class Source:
            class Meta:
                episodes = [{"tasks": ["Pick up a different object"]}]
            meta = Meta()

        with self.assertRaisesRegex(ValueError, "different recorded task"):
            CURATE.check_selected_tasks(Source(), {"train": [{"episode_index": 0}]})


if __name__ == "__main__":
    unittest.main()
