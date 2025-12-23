"""Configuration files and their hashes.

Every stage reads one YAML configuration file and one YAML paths file. The paths
file names the machine-specific locations (data roots, project directory,
scratch directory) and is kept separate so that the same stage configuration
runs unchanged on the compute server and on a laptop with sampled data. A
value of the paths file may refer to environment variables as ``${NAME}``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary. An empty file gives an empty dictionary."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(config_path: str | Path, paths_path: str | Path | None = None) -> dict[str, Any]:
    """Load a stage configuration and attach the paths file under the key ``paths``."""
    cfg = load_yaml(config_path)
    if paths_path is not None:
        cfg["paths"] = expand_paths(load_yaml(paths_path))
    cfg.setdefault("paths", {})
    return cfg


def expand_paths(paths: dict[str, Any]) -> dict[str, Any]:
    """Replace ``${NAME}`` in the values of a paths file by the environment variable of that name.

    An entry whose variable is not set becomes empty (None), so that the stage
    that needs it stops with the name of the entry.
    """
    out: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, str):
            value = os.path.expandvars(value)
            if "${" in value:
                value = None
        out[key] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Return a short, stable hash of a configuration dictionary."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def resolve_path(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    """Join a named root from ``cfg['paths']`` with optional sub-path parts."""
    try:
        root = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"paths file has no entry {key!r}") from exc
    if root is None:
        raise KeyError(f"paths entry {key!r} is empty: set it in the paths file or export its variable")
    return Path(root, *parts)
