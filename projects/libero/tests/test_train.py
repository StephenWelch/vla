import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vla_libero import train as module


class LiberoTrainingTests(unittest.TestCase):
    def test_exact_language_selection_and_episode_holdout(self):
        rows = [
            {"episode_index": i, "tasks": ["open the middle drawer of the cabinet"]}
            for i in range(50, 100)
        ]
        rows += [{"episode_index": 0, "tasks": ["open the top drawer of the cabinet"]}]
        split = module.select_episodes(reversed(rows), module.Config().instruction, 0.2)
        self.assertEqual(split["train"], list(range(50, 90)))
        self.assertEqual(split["val"], list(range(90, 100)))

    def test_missing_task_and_duplicate_episodes_fail(self):
        row = {"episode_index": 4, "tasks": [module.Config().instruction]}
        for rows in ([], [row], [row, row]):
            with self.assertRaises(ValueError):
                module.select_episodes(rows, module.Config().instruction, 0.2)

    def test_yaml_cli_and_native_train_eval_contract(self):
        config = module.parse_args(
            module.Config,
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[3]
                    / "configs/libero/libero-drawer-act.yaml"
                ),
                "--steps",
                "4",
                "--wandb.no-enable",
            ],
        )
        self.assertEqual(config.steps, 4)
        self.assertFalse(config.wandb.enable)
        cfg = module.native_config(config, list(range(50, 100)), "abc123", Path("data"))
        self.assertEqual(cfg.dataset.episodes, list(range(50, 100)))
        self.assertEqual(cfg.dataset.revision, "abc123")
        self.assertEqual(cfg.env.task_ids, [0])
        self.assertEqual(cfg.env.control_mode, "relative")
        self.assertEqual(cfg.env.observation_width, 256)
        self.assertEqual(cfg.env.observation_height, 256)
        self.assertEqual(cfg.policy.chunk_size, 16)
        self.assertEqual(cfg.eval_steps, 1000)
        self.assertEqual(cfg.env_eval_freq, 2000)
        command = module.test_command(config, Path("model"))
        self.assertIn("--seed=2027", command)
        self.assertIn("--env.task_ids=[0]", command)
        self.assertIn("--env.observation_width=256", command)

    def test_normalization_excludes_validation_and_prunes_only_owned_checkpoints(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_dir = root / "meta/episodes/chunk-000"
            meta_dir.mkdir(parents=True)
            rows = []
            for index, mean in enumerate((2.0, 4.0, 1000.0)):
                row = {"episode_index": index}
                for name, value in {
                    "min": [mean],
                    "max": [mean],
                    "mean": [mean],
                    "std": [0.0],
                    "count": [1],
                }.items():
                    row[f"stats/action/{name}"] = value
                rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), meta_dir / "file-000.parquet")
            train = SimpleNamespace(
                episodes=[0, 1],
                meta=SimpleNamespace(camera_keys=[], stats={}),
                hf_dataset={
                    "action": [[2.0], [4.0]],
                    "observation.state": [[2.0], [4.0]],
                },
            )
            val = SimpleNamespace(episodes=[2], meta=SimpleNamespace(stats={}))

            def save(**kwargs):
                kwargs["checkpoint_dir"].mkdir(parents=True)
                (kwargs["checkpoint_dir"] / "saved.json").write_text(
                    json.dumps({"saved": True})
                )

            trainer = SimpleNamespace(
                make_train_eval_datasets=lambda cfg: (train, val), save_checkpoint=save
            )
            with patch(
                "lerobot.datasets.factory.make_dataset", side_effect=[train, val]
            ) as loader:
                module.install_data_and_checkpoint_hooks(
                    trainer, {"train": [0, 1], "val": [2]}
                )
                trainer.make_train_eval_datasets(
                    SimpleNamespace(
                        dataset=SimpleNamespace(
                            root=root, image_transforms=SimpleNamespace(enable=False)
                        )
                    )
                )
                self.assertEqual(
                    loader.call_args_list[0].args[0].dataset.episodes, [0, 1]
                )
                self.assertEqual(loader.call_args_list[1].args[0].dataset.episodes, [2])
            self.assertEqual(float(train.meta.stats["action"]["mean"][0]), 3.0)
            self.assertIs(train.meta.stats, val.meta.stats)
            old = root / "checkpoints/000001"
            old.mkdir(parents=True)
            unrelated = root / "checkpoints/notes"
            unrelated.mkdir()
            latest = root / "checkpoints/000002"
            trainer.save_checkpoint(checkpoint_dir=latest)
            self.assertFalse(old.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue((latest / "saved.json").is_file())


if __name__ == "__main__":
    unittest.main()
