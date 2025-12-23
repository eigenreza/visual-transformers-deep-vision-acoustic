"""Extraction of a small development sample from the full data sources.

The sample mirrors the directory layout of the sources so that every stage
runs unchanged against it with a different paths file: a few hundred annotated
image and XML pairs chosen across years and labels, a short span of the
backscatter store at all frequencies with the original float32 values, and one
camera transect reduced to its metadata file and a handful of frames.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .acoustics.store import open_sv_store
from .config import resolve_path
from .data.annotations import pair_images_and_xml
from .data.voc import parse_voc_xml
from .manifest import write_manifest

log = logging.getLogger(__name__)


def select_annotation_keys(
    pairs: pd.DataFrame,
    key_labels: dict[str, set[str]],
    n_total: int,
    per_year_min: int,
    per_label_min: int,
    seed: int,
) -> list[str]:
    """Choose paired keys so that every year and every label is represented.

    ``key_labels`` maps a key to the set of raw labels in its annotation. Keys
    are first drawn per year, then per label until each label has at least
    ``per_label_min`` keys where available, then at random up to ``n_total``.
    """
    rng = np.random.default_rng(seed)
    paired = pairs[pairs["status"] == "paired"]
    chosen: list[str] = []
    seen: set[str] = set()

    def take(candidates: list[str], n: int) -> None:
        pool = [k for k in candidates if k not in seen]
        if not pool or n <= 0:
            return
        picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
        for k in picks:
            seen.add(str(k))
            chosen.append(str(k))

    for _, group in paired.groupby("year"):
        take(group["key"].tolist(), per_year_min)
    labels: dict[str, list[str]] = {}
    for key, labs in key_labels.items():
        for lab in labs:
            labels.setdefault(lab, []).append(key)
    for lab in sorted(labels):
        have = sum(1 for k in chosen if lab in key_labels.get(k, set()))
        take(labels[lab], per_label_min - have)
    take(paired["key"].tolist(), n_total - len(chosen))
    return sorted(chosen)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _sample_annotations(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    scfg = cfg.get("annotations", {})
    images_root = resolve_path(cfg, "annotated_data", "images")
    xml_root = resolve_path(cfg, "annotated_data", "xml")
    pairs, _ = pair_images_and_xml(images_root, xml_root)
    # Drop keys that live inside nested duplicate trees (for example images/images/...).
    pairs = pairs[~pairs["key"].str.startswith(("images/", "xml/"))]
    paired = pairs[pairs["status"] == "paired"]
    key_labels: dict[str, set[str]] = {}
    for key, rel in zip(paired["key"], paired["xml_rel"], strict=True):
        ann = parse_voc_xml(Path(xml_root, rel))
        key_labels[key] = {b.label for b in ann.boxes}
    keys = select_annotation_keys(
        pairs,
        key_labels,
        n_total=int(scfg.get("n_pairs", 300)),
        per_year_min=int(scfg.get("per_year_min", 20)),
        per_label_min=int(scfg.get("per_label_min", 3)),
        seed=int(cfg.get("seed", 0)),
    )
    lookup = paired.set_index("key")
    for key in keys:
        _copy(
            Path(images_root, lookup.loc[key, "image_rel"]),
            out_root / "annotated_data" / "images" / lookup.loc[key, "image_rel"],
        )
        _copy(
            Path(xml_root, lookup.loc[key, "xml_rel"]),
            out_root / "annotated_data" / "xml" / lookup.loc[key, "xml_rel"],
        )
    labels_covered = sorted({lab for k in keys for lab in key_labels.get(k, set())})
    log.info("copied %d annotation pairs covering %d labels", len(keys), len(labels_covered))
    return {"n_pairs": len(keys), "labels": labels_covered, "years": sorted({k.split("/")[0] for k in keys})}


def _sample_acoustics(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    acfg = cfg.get("acoustics", {})
    store_name = cfg.get("zarr_store", "S2018105_sv.zarr")
    ds = open_sv_store(resolve_path(cfg, "source_data", store_name))
    start, end = str(acfg.get("start")), str(acfg.get("end"))
    max_range = float(acfg.get("max_range_m", 320))
    sub = ds.sel(ping_time=slice(start, end))
    sub = sub.sel(range=sub["range"] <= max_range)
    drop = [v for v in ("angle_alongship", "angle_athwartship") if v in sub]
    sub = sub.drop_vars(drop)
    sub = sub.load()
    for name in list(sub.variables):
        sub[name].encoding = {}
    target = out_root / "source_data" / store_name
    if target.exists():
        shutil.rmtree(target)
    encoding = {"sv": {"chunks": (1, 1000, sub.sizes["range"])}}
    sub.to_zarr(target, mode="w", consolidated=True, zarr_format=2, encoding=encoding)
    info = {
        "path": str(target),
        "n_pings": int(sub.sizes["ping_time"]),
        "n_range": int(sub.sizes["range"]),
        "first_ping": str(sub["ping_time"].values[0]),
        "last_ping": str(sub["ping_time"].values[-1]),
        "dropped_variables": drop,
    }
    log.info("wrote acoustic sample %s", info)
    return info


def _sample_transect(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    tcfg = cfg.get("transect", {})
    name = str(tcfg.get("id"))
    src = resolve_path(cfg, "source_data", name)
    dst = out_root / "source_data" / name
    _copy(src / "deepvision.xml", dst / "deepvision.xml")
    copied: dict[str, int] = {}
    for cam, n in (
        ("Left", int(tcfg.get("n_frames_left", 60))),
        ("Right", int(tcfg.get("n_frames_right", 10))),
    ):
        folder = str(tcfg.get("folder"))
        frames = sorted(p for p in (src / cam / folder).iterdir() if p.is_file())[:n]
        for p in frames:
            _copy(p, dst / cam / folder / p.name)
        empty = folder.replace("_active", "_empty")
        (dst / cam / empty).mkdir(parents=True, exist_ok=True)
        copied[cam] = len(frames)
    log.info("copied transect sample for %s: %s", name, copied)
    return {"transect": name, "folder": tcfg.get("folder"), "frames": copied}


def run_pull_sample(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Write the development sample under ``out_dir`` and record a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "annotations": _sample_annotations(cfg, out_dir),
        "acoustics": _sample_acoustics(cfg, out_dir),
        "transect": _sample_transect(cfg, out_dir),
    }
    inputs = {
        "annotated_data": resolve_path(cfg, "annotated_data"),
        "source_data": resolve_path(cfg, "source_data"),
    }
    write_manifest(out_dir, "pull-sample", cfg, inputs, [], extra=summary)
    return out_dir


def open_sample_store(path: str | Path) -> xr.Dataset:
    """Open a sample store written by :func:`run_pull_sample`."""
    return open_sv_store(path)
