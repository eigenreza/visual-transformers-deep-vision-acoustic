from __future__ import annotations

from pathlib import Path

import pytest

from echolabel.config import expand_paths, load_config, resolve_path


def test_paths_are_read_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "source_data: ${EXAMPLE_SOURCE_DATA}\n"
        "project_dir: ${EXAMPLE_PROJECT_DIR}/runs\n"
        "scratch_dir: ${EXAMPLE_UNSET_DIR}\n"
        "annotated_data: /plain/path\n"
        "annotated_data_source: null\n"
    )
    stage = tmp_path / "stage.yaml"
    stage.write_text("stage: example\n")
    monkeypatch.setenv("EXAMPLE_SOURCE_DATA", "/srv/data")
    monkeypatch.setenv("EXAMPLE_PROJECT_DIR", "/srv/project")
    monkeypatch.delenv("EXAMPLE_UNSET_DIR", raising=False)
    cfg = load_config(stage, paths)
    assert resolve_path(cfg, "source_data", "a") == Path("/srv/data", "a")
    assert resolve_path(cfg, "project_dir") == Path("/srv/project/runs")
    assert resolve_path(cfg, "annotated_data") == Path("/plain/path")
    # A variable that is not set leaves the entry empty, and the stage that needs it stops.
    with pytest.raises(KeyError, match="scratch_dir"):
        resolve_path(cfg, "scratch_dir")
    assert expand_paths({"a": 3, "b": None}) == {"a": 3, "b": None}


def test_example_paths_file_names_every_root() -> None:
    from echolabel.config import load_yaml

    example = Path(__file__).resolve().parents[1] / "configs" / "paths" / "server.example.yaml"
    entries = load_yaml(example)
    assert set(entries) == {
        "source_data",
        "annotated_data",
        "annotated_data_source",
        "project_dir",
        "scratch_dir",
    }
    assert all(str(v).startswith("${ECHOLABEL_") for v in entries.values())
