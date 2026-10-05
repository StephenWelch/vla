"""Synthetic checks for the timestamp alignment math."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from camera_sync import active_command, interpolate_state  # noqa: E402

spec = importlib.util.spec_from_file_location("estimate_camera_lag", SCRIPTS / "estimate-camera-lag.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
align_spec = importlib.util.spec_from_file_location("align_lighter_dataset", SCRIPTS / "align-lighter-dataset.py")
align_module = importlib.util.module_from_spec(align_spec)
align_spec.loader.exec_module(align_module)


class TestCameraSync(unittest.TestCase):
    def test_interpolates_encoder_and_uses_active_command(self):
        state = interpolate_state(np.array([0.0, 0.1]),
                                  np.array([[0., 2.], [10., 4.]]), 0.05)
        np.testing.assert_allclose(state, [5., 3.])
        commands = [{"end_time": 0.01, "action": [1., 2.]},
                    {"end_time": 0.08, "action": [3., 4.]}]
        np.testing.assert_array_equal(active_command(commands, 0.05), [1., 2.])
        with self.assertRaisesRegex(ValueError, "No command"):
            active_command(commands, 0.0)

    def test_recovers_known_motion_lag(self):
        encoder_time = np.arange(0, 20, 1 / 30)
        angle = 30 * np.sin(2 * np.pi * 0.37 * encoder_time) + 12 * np.sin(2 * np.pi * 0.71 * encoder_time)
        image_time = np.arange(0, 20, 1 / 30) + 0.08
        observed = np.interp(image_time - 0.08, encoder_time, angle)
        rotations = np.diff(np.deg2rad(observed))
        result = module.estimate_lag(image_time, encoder_time, angle, rotations)
        self.assertLess(abs(result["lag_ms"] - 80), 15)

    def test_aligned_episode_loads_as_lerobot_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            key: {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]}
            for key in ("observation.state", "action")
        }
        features["observation.images.left_wrist_cam"] = {
            "dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = LeRobotDataset.create("test/sync", fps=30, features=features,
                                           root=root / "raw", robot_type="test")
            timing, commands = [], []
            for index in range(12):
                tick = index / 30
                source.add_frame({"observation.state": np.full(6, index, dtype=np.float32),
                                  "action": np.full(6, index + 1, dtype=np.float32),
                                  "observation.images.left_wrist_cam":
                                      np.full((32, 32, 3), index * 10, dtype=np.uint8),
                                  "task": "grasp"})
                timing.append({"frame": index, "camera_capture_times": {"left_wrist_cam": tick + 0.05},
                               "state_read_times": {"left": [tick + 0.008, tick + 0.012]}})
                commands.append({"end_time": tick + 0.005,
                                 "action": [index + 1] * 6, "write_performed": True})
            source.save_episode()
            source.finalize()
            source = LeRobotDataset("test/sync", root=root / "raw", video_backend="pyav")
            target = LeRobotDataset.create("test/sync_aligned", fps=30, features=features,
                                           root=root / "aligned", robot_type="test")
            provenance = align_module.align_episode(source, 0, timing, commands, 0.04, target)
            target.finalize()
            checked = LeRobotDataset("test/sync_aligned", root=root / "aligned", video_backend="pyav")
            self.assertEqual(len(checked), len(provenance))
            self.assertGreater(len(checked), 5)
            self.assertEqual(checked[0]["observation.state"].numel(), 6)
            sidecar = root / "raw" / "meta" / "telegrip_annotations"
            sidecar.mkdir(parents=True)
            (sidecar / "episode_000000_timing.json").write_text(json.dumps(timing))
            (sidecar / "episode_000000_commands.json").write_text(json.dumps(commands))
            (sidecar / "episode_000000.json").write_text(json.dumps({
                "episode_index": 0, "outcome": "success", "mistake": True,
                "mistake_frames": [3], "position_id": "A", "duration_seconds": 0.4}))
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"camera": "left_wrist_cam", "width": 32,
                                               "height": 32, "fps": 30, "lag_ms": 40}))
            old_argv = sys.argv
            try:
                sys.argv = ["align-lighter-dataset.py", "--source", str(root / "raw"),
                            "--output", str(root / "full"), "--calibration", str(calibration),
                            "--repo-id", "test/sync"]
                align_module.main()
            finally:
                sys.argv = old_argv
            full = LeRobotDataset("test/sync", root=root / "full", video_backend="pyav")
            self.assertEqual(len(full), len(provenance))
            annotation = json.loads((root / "full" / "meta" / "telegrip_annotations" /
                                     "episode_000000.json").read_text())
            self.assertTrue(annotation["mistake_frames"])
            self.assertEqual(annotation["source_duration_seconds"], 0.4)


if __name__ == "__main__":
    unittest.main()
