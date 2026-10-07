"""Configuration layering, nested overrides and interpolation."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from vla_tools.config import parse_args


@dataclass
class Options:
    steps: int = 25
    enabled: bool = True
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    output: Path
    training: Options = field(default_factory=Options)


def test_nested_config_layering(tmp_path):
    first, second = tmp_path / "one.yaml", tmp_path / "two.yaml"
    first.write_text(
        "output: runs/demo\ntraining:\n  steps: 100\n  enabled: false\n  overrides:\n    policy.chunk_size: 25\n"
    )
    second.write_text("output: ${oc.env:HOME}/run\ntraining:\n  steps: 200\n")
    result = parse_args(
        Config,
        [
            "--config",
            str(first),
            "--config",
            str(second),
            "--training.steps",
            "300",
            "--training.enabled",
        ],
    )
    assert result.output == Path.home() / "run"
    assert result.training.steps == 300 and result.training.enabled
    assert result.training.overrides == {"policy.chunk_size": "25"}


@pytest.mark.parametrize("value", ["unknown: 1", "training:\n  steps: wrong"])
def test_unknown_or_invalid_config(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text("output: runs/demo\n" + value)
    with pytest.raises(SystemExit):
        parse_args(Config, ["--config", str(path)])


def test_mapping_required(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping")
    with pytest.raises(ValueError, match="mapping"):
        parse_args(Config, ["--config", str(path)])
