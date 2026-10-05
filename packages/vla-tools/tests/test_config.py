from dataclasses import dataclass, field
from pathlib import Path

import pytest
from vla_tools.config import parse_args


@dataclass
class Logging:
    enable: bool = True
    project: str = "default"


@dataclass
class Config:
    output: Path = Path("outputs")
    steps: int = 10
    logging: Logging = field(default_factory=Logging)


def test_layered_yaml_environment_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_TEST_OUTPUT", str(tmp_path / "run"))
    recipe, local = tmp_path / "recipe.yaml", tmp_path / "local.yaml"
    recipe.write_text(
        "output: ${oc.env:VLA_TEST_OUTPUT}\nsteps: 20\n"
        "logging:\n  enable: true\n  project: experiment\n"
    )
    local.write_text("steps: 30\nlogging:\n  enable: false\n")
    config = parse_args(
        Config,
        ["--config", str(recipe), f"--config={local}", "--steps", "40"],
    )
    assert config.output == tmp_path / "run"
    assert config.steps == 40
    assert not config.logging.enable
    assert config.logging.project == "experiment"


def test_yaml_requires_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- steps\n- 20\n")
    with pytest.raises(ValueError, match="mapping"):
        parse_args(Config, ["--config", str(path)])
