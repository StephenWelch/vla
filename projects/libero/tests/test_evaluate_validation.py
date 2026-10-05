import tempfile
import unittest
from pathlib import Path

import numpy as np
from vla_libero import evaluate_validation as module


class ValidationTests(unittest.TestCase):
    def test_full_sequence_match_recovers_removed_prefix(self):
        actions = np.arange(21, dtype=np.float32).reshape(3, 7)
        source = {"demo_5": {"actions": np.vstack([np.zeros((2, 7)), actions])}}
        name, offset, error = module.match_episode(actions, source)
        self.assertEqual((name, offset, error), ("demo_5", 2, 0.0))
        with self.assertRaises(ValueError):
            module.match_episode(actions, {"short": {"actions": actions[:1]}})

    def test_ambiguous_match_is_rejected(self):
        actions = np.ones((3, 7))
        with self.assertRaises(ValueError):
            module.match_episode(
                actions, {"a": {"actions": actions}, "b": {"actions": actions}}
            )

    def test_state_mapping_sorts_frames_and_reports_initial_state_aliases(self):
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            rows = []
            path = root / "source.hdf5"
            with h5py.File(path, "w") as source:
                for index in (0, 1):
                    demo = source.create_group(f"data/demo_{index}")
                    actions = np.full((2, 7), index + 1.0)
                    actions[1] += 0.1
                    demo["actions"] = actions
                    states = np.zeros((2, 79))
                    states[:, 0] = (
                        index  # Time differs, physical reset state is identical.
                    )
                    demo["states"] = states
                    demo["obs/ee_states"] = np.zeros((2, 6))
                    demo["obs/gripper_states"] = np.zeros((2, 2))
                    for frame in (1, 0):
                        rows.append(
                            {
                                "episode_index": index,
                                "frame_index": frame,
                                "action": actions[frame].tolist(),
                                "observation.state": [0.0] * 8,
                            }
                        )
            pq.write_table(pa.Table.from_pylist(rows), root / "data/frames.parquet")
            record = {
                "dataset": {"root": str(root)},
                "split": {"train": [0], "val": [1]},
            }
            mapping, states, _ = module.prepare_states(record, path)
            self.assertEqual(mapping[1]["source_demo"], "demo_1")
            self.assertEqual(mapping[1]["matching_train_initial_states"], [0])
            self.assertEqual(states[1][0], 1)


if __name__ == "__main__":
    unittest.main()
