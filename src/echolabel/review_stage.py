"""Stage 3 quality control: fixed samples of detected frames for manual review."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .detection.review import contact_sheets, review_summary, sample_frames_by_band
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def run_review_detections(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    transect: str = "",
    camera: str = "Left",
) -> Path:
    """Draw a fixed random sample of frames per confidence band for one class and save contact sheets."""
    rcfg = cfg.get("inference", {}).get("review", {})
    sets = rcfg.get("sets") or [
        {
            "class_name": rcfg.get("class_name", "krill"),
            "bands": rcfg.get("bands", [[0.05, 0.3], [0.3, 0.6], [0.6, 1.01]]),
            "n_per_band": rcfg.get("n_per_band", 50),
        }
    ]
    project = resolve_path(cfg, "project_dir")
    infer_dir = Path(project, "runs", "inference", f"{experiment}_seed{seed}", transect, camera)
    detections_path = infer_dir / "detections.parquet"
    target = out_dir / f"{experiment}_seed{seed}" / transect / camera
    run_cfg = {**cfg, "experiment": experiment, "seed": seed, "transect": transect, "camera": camera}
    inputs: dict[str, str | Path] = {"detections": detections_path}
    if not force and manifest_is_current(target, run_cfg, inputs):
        log.info("review sample of %s is up to date in %s", transect, target)
        return target
    if not detections_path.is_file():
        raise FileNotFoundError(f"detections not found: {detections_path}")
    target.mkdir(parents=True, exist_ok=True)
    dets = pd.read_parquet(detections_path)
    if "static" in dets.columns:
        dets = dets[~dets["static"]].reset_index(drop=True)
    dets["frame"] = dets["folder"].astype(str) + "/" + dets["filename"].astype(str)
    transect_dir = resolve_path(cfg, "source_data", transect)
    summaries = []
    outputs: list[str | Path] = []
    for spec in sets:
        class_name = str(spec["class_name"])
        bands = [(float(lo), float(hi)) for lo, hi in spec["bands"]]
        n_per_band = int(spec.get("n_per_band", 50))
        set_dir = target / class_name
        set_dir.mkdir(parents=True, exist_ok=True)
        sample = sample_frames_by_band(dets, class_name, bands, n_per_band, seed=int(rcfg.get("seed", 0)))
        sample.to_csv(set_dir / "sample.csv", index=False)
        paths = {fr: Path(transect_dir, camera, fr) for fr in sample["frame"]}
        sheets = contact_sheets(
            sample,
            dets[dets["frame"].isin(sample["frame"])],
            paths,
            class_name,
            set_dir,
            per_sheet=int(rcfg.get("per_sheet", 10)),
        )
        summaries.append(
            {
                "class_name": class_name,
                "n_frames_with_class": int(dets.loc[dets["class_name"] == class_name, "frame"].nunique()),
                "sheets": [p.name for p in sheets],
                **review_summary(sample, bands),
            }
        )
        outputs += [set_dir / "sample.csv", *sheets]
    summary = {
        "experiment": experiment,
        "seed": seed,
        "transect": transect,
        "camera": camera,
        "sets": summaries,
    }
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    outputs.append(target / "summary.json")
    write_manifest(target, "review-detections", run_cfg, inputs, outputs)
    return target
