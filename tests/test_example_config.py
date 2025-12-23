from __future__ import annotations

from pathlib import Path

from echolabel.config import load_yaml
from echolabel.models_stage import experiment_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example_acoustic_model.yaml"


def test_example_configuration_specifies_a_complete_run() -> None:
    cfg = load_yaml(EXAMPLE)
    assert cfg["folds"] and cfg["seeds"] and cfg["windows_run"]["experiment"]
    assert cfg["evaluation"]["reference"] in cfg["experiments"]
    for name in cfg["experiments"]:
        exp = experiment_config(cfg, name)
        assert exp["network"] in {"mean", "gbm", "cnn", "vit", "hybrid"}
        assert exp["geometry"] and exp["protocol"] and exp["family"]
        assert exp["inputs"]["frequencies_hz"] and exp["targets"]["classes"]
        assert exp["train"]["epochs"] > 0
