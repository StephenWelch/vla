"""Dataset/checkpoint contract and reproducible training-launch tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ogbench_mjwarp import train as TRAIN


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.dataset, self.policy = root / "dataset", root / "policy"
        (self.dataset / "meta").mkdir(parents=True)
        self.policy.mkdir()
        self.info = {
            "total_episodes": 4,
            "total_frames": 512,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [18]},
                "action": {"dtype": "float32", "shape": [5]},
                "observation.images.front": {"dtype": "video", "shape": [3, 32, 32]},
                "observation.images.wrist": {"dtype": "video", "shape": [3, 32, 32]},
                "annotation.reference": {"dtype": "float32", "shape": [5]},
            },
        }
        self.checkpoint = {
            "type": "smolvla",
            "max_state_dim": 32,
            "max_action_dim": 32,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                **{
                    f"observation.images.camera{i}": {
                        "type": "VISUAL",
                        "shape": [3, 256, 256],
                    }
                    for i in range(1, 4)
                },
            },
        }
        self.save_metadata()
        (self.policy / "model.safetensors").touch()
        (self.dataset / "manifest.json").write_text(
            json.dumps({"repo_id": "local/test"})
        )
        self.config = TRAIN.TrainConfig(self.dataset, self.policy, root / "run")

    def save_metadata(self):
        (self.dataset / "meta/info.json").write_text(json.dumps(self.info))
        (self.policy / "config.json").write_text(json.dumps(self.checkpoint))

    def test_simulator_features_replace_checkpoint_hardware_dimensions(self):
        plan = TRAIN.training_plan(self.config)
        args = dict(s[2:].split("=", 1) for s in plan["command"][3:])
        inputs = json.loads(args["policy.input_features"])
        self.assertEqual(inputs["observation.state"]["shape"], [18])
        self.assertEqual(inputs["observation.images.camera1"]["shape"], [3, 32, 32])
        self.assertNotIn("annotation.reference", inputs)
        self.assertNotIn("observation.images.camera3", inputs)
        self.assertEqual(plan["dataset"]["action"]["shape"], [5])
        self.assertEqual(
            plan["camera_map"]["observation.images.wrist"], "observation.images.camera2"
        )
        self.assertIn(str(self.dataset / "manifest.json"), plan["metadata_sha256"])

    def test_single_hardware_camera_can_use_camera2(self):
        self.config.camera_keys = ["observation.images.wrist"]
        self.config.camera_names = ["observation.images.camera2"]
        plan = TRAIN.training_plan(self.config)
        self.assertEqual(
            plan["camera_map"],
            {"observation.images.wrist": "observation.images.camera2"},
        )

    def test_act_from_scratch_uses_native_dataset_features(self):
        self.config.policy = None
        self.config.policy_type = "act"
        plan = TRAIN.training_plan(self.config)
        self.assertIn("--policy.type=act", plan["command"])
        self.assertFalse(
            any(arg.startswith("--policy.path=") for arg in plan["command"])
        )
        self.assertEqual(plan["camera_map"], {})
        inputs_arg = next(
            arg for arg in plan["command"] if arg.startswith("--policy.input_features=")
        )
        inputs = json.loads(inputs_arg.split("=", 1)[1])
        self.assertIn("observation.images.front", inputs)
        self.assertIn("observation.images.wrist", inputs)
        self.assertNotIn("annotation.reference", inputs)

    def test_pi05_binds_dataset_dimensions_and_camera_map(self):
        self.checkpoint["type"] = "pi05"
        self.save_metadata()
        self.config.policy_type = "pi05"
        plan = TRAIN.training_plan(self.config)
        self.assertEqual(plan["dataset"]["state"]["shape"], [18])
        self.assertEqual(plan["dataset"]["action"]["shape"], [5])
        self.assertEqual(len(plan["camera_map"]), 2)
        self.config.policy = None
        with self.assertRaisesRegex(ValueError, "pretrained checkpoint"):
            TRAIN.training_plan(self.config)

    def test_invalid_dimensions_and_camera_contracts_are_rejected(self):
        self.info["features"]["action"]["shape"] = [33]
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "max_action_dim"):
            TRAIN.training_plan(self.config)
        self.info["features"]["action"]["shape"] = [5]
        self.save_metadata()
        self.config.camera_names = ["observation.images.camera1"]
        with self.assertRaisesRegex(ValueError, "one unique"):
            TRAIN.training_plan(self.config)
        self.config.camera_names = [
            "observation.images.camera1",
            "observation.images.camera1",
        ]
        with self.assertRaisesRegex(ValueError, "one unique"):
            TRAIN.training_plan(self.config)

    def test_incomplete_export_and_existing_output_rejected(self):
        (self.dataset / "INCOMPLETE").touch()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            TRAIN.training_plan(self.config)
        (self.dataset / "INCOMPLETE").unlink()
        self.config.output.mkdir()
        with self.assertRaises(FileExistsError):
            TRAIN.training_plan(self.config)

    def test_episode_subset_uses_exported_indices(self):
        self.config.episodes = [0, 3]
        plan = TRAIN.training_plan(self.config)
        self.assertIn("--dataset.episodes=[0, 3]", plan["command"])
        self.config.episodes = [0, 4]
        with self.assertRaisesRegex(ValueError, "exported"):
            TRAIN.training_plan(self.config)

    def test_native_overrides_preserve_dataset_contract(self):
        self.config.overrides = ["policy.chunk_size=8", "policy.n_action_steps=8"]
        self.assertIn(
            "--policy.chunk_size=8", TRAIN.training_plan(self.config)["command"]
        )
        self.config.overrides = ["dataset.root=other"]
        with self.assertRaisesRegex(ValueError, "typed CLI"):
            TRAIN.training_plan(self.config)

    def test_yaml_cli_precedence_and_unknown_fields(self):
        path = Path(self.temp.name) / "train.yaml"
        path.write_text(
            f"dataset: {self.dataset.as_posix()}\npolicy: {self.policy.as_posix()}\n"
            f"output: {self.config.output.as_posix()}\nsteps: 10\ndry_run: false\n"
        )
        config = TRAIN.parse_args(
            TRAIN.TrainConfig, ["--config", str(path), "--steps", "1", "--dry-run"]
        )
        self.assertEqual(config.steps, 1)
        self.assertTrue(config.dry_run)
        path.write_text(path.read_text() + "unknown_option: 1\n")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            TRAIN.parse_args(TRAIN.TrainConfig, ["--config", str(path)])

    def test_dry_run_does_not_write_or_start_training(self):
        self.config.dry_run = True
        with (
            patch.object(TRAIN, "parse_args", return_value=self.config),
            patch.object(TRAIN, "run_training") as run,
            patch("builtins.print"),
        ):
            TRAIN.main([])
        run.assert_not_called()
        self.assertFalse(
            self.config.output.parent.joinpath("run.experiment.json").exists()
        )

    def test_training_failure_preserves_experiment(self):
        def fail(*args, **kwargs):
            self.config.output.mkdir()
            raise RuntimeError("training failed")

        with (
            patch.object(TRAIN, "parse_args", return_value=self.config),
            patch.object(TRAIN, "run_training", side_effect=fail),
            patch("builtins.print"),
            self.assertRaises(RuntimeError),
        ):
            TRAIN.main([])
        record = json.loads((self.config.output / "experiment.json").read_text())
        self.assertEqual(record["status"], "failed_or_interrupted")
        self.assertIn("metadata_sha256", record)

    def test_model_cache_override_preserves_existing_authentication(self):
        original = Path(self.temp.name) / "original-hf"
        original.mkdir()
        (original / "token").write_text("test-credential")
        self.config.hf_home = Path(self.temp.name) / "models"
        result = TRAIN.runtime_environment(
            self.config.hf_home, {"HF_HOME": str(original)}
        )
        self.assertEqual(result["HF_TOKEN_PATH"], str((original / "token").resolve()))
        self.assertEqual(result["HF_HOME"], str(self.config.hf_home.resolve()))
        self.assertNotIn("test-credential", result.values())

    def test_explicit_authentication_path_is_preserved(self):
        self.config.hf_home = Path(self.temp.name) / "models"
        result = TRAIN.runtime_environment(
            self.config.hf_home, {"HF_TOKEN_PATH": "explicit-token-file"}
        )
        self.assertEqual(result["HF_TOKEN_PATH"], "explicit-token-file")


if __name__ == "__main__":
    unittest.main()
