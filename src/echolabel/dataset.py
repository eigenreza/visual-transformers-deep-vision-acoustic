"""Stage 1: annotation dataset for detector training.

Pairs images with their annotation files, maps raw labels to the taxonomy,
excludes tows and folders that carry no annotation at all, assigns tows to
splits, verifies the leakage rules and exports COCO files per split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .data.annotations import build_annotation_tables
from .data.coco import to_coco, write_coco
from .data.splits import (
    assign_splits,
    check_split_properties,
    class_counts_by_split,
    tow_key,
    tow_table,
)
from .data.taxonomy import Taxonomy, apply_taxonomy, mapping_table
from .data.transects import parse_frame_timestamp
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def frames_inside_windows(images: pd.DataFrame, windows: dict[str, tuple[str, str]]) -> pd.Series:
    """Name of the window each image's frame timestamp falls into, or empty string."""
    starts = {k: np.datetime64(v[0]) for k, v in windows.items()}
    ends = {k: np.datetime64(v[1]) for k, v in windows.items()}
    out = []
    for name in images["basename"]:
        ts = parse_frame_timestamp(str(name))
        hit = ""
        if ts is not None:
            for k in windows:
                if starts[k] <= ts <= ends[k]:
                    hit = k
                    break
        out.append(hit)
    return pd.Series(out, index=images.index)


def run_build_annotation_dataset(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    images_root = resolve_path(cfg, "annotated_data", "images")
    xml_root = resolve_path(cfg, "annotated_data", "xml")
    taxonomy_path = Path(cfg["taxonomy_file"])
    inputs: dict[str, str | Path] = {"images": images_root, "xml": xml_root, "taxonomy": taxonomy_path}
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("annotation dataset is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str | Path] = []
    acfg = cfg.get("annotations", {})
    scfg = cfg.get("splits", {})
    ecfg = cfg.get("export", {})
    taxonomy = Taxonomy.from_yaml(taxonomy_path)
    classes = taxonomy.classes

    tables = build_annotation_tables(
        images_root,
        xml_root,
        tuple(acfg.get("image_extensions", [".jpg", ".jpeg", ".png"])),
        n_workers=int(cfg.get("n_workers", 16)),
        exclude_top=tuple(acfg.get("exclude_top_level_dirs", [])),
    )
    images, boxes = tables["images"], tables["boxes"]
    statuses = list(acfg.get("paired_statuses", ["paired", "paired_by_stem"]))
    img = images[images["status"].isin(statuses)].copy()
    n_paired = len(img)
    no_size = img["width_used"].isna() | img["height_used"].isna()
    img = img[~no_size].copy()
    bx = apply_taxonomy(boxes[boxes["key"].isin(img["key"])].copy(), taxonomy)
    img["basename"] = img["key"].str.rsplit("/", n=1).str[-1]
    img["tow"] = [tow_key(str(y), str(s)) for y, s in zip(img["year"], img["station"], strict=True)]
    img["rel_dir"] = img["image_rel"].str.rsplit("/", n=1).str[0]

    # Folders and tows without a single annotated frame are unannotated, not empty.
    by_dir = img.groupby("rel_dir").agg(
        n_images=("key", "size"), n_with_boxes=("n_boxes", lambda s: int((s > 0).sum()))
    )
    unannotated_dirs = sorted(by_dir.index[by_dir["n_with_boxes"] == 0])
    by_tow = img.groupby("tow").agg(
        n_images=("key", "size"), n_with_boxes=("n_boxes", lambda s: int((s > 0).sum()))
    )
    unannotated_tows = sorted(by_tow.index[by_tow["n_with_boxes"] == 0])
    zero_box = img[img["n_boxes"] == 0]
    zero_by_tow = (
        zero_box.groupby(["year", "station", "rel_dir"]).size().rename("n_zero_box").reset_index()
        .merge(by_dir.reset_index()[["rel_dir", "n_images", "n_with_boxes"]], on="rel_dir", how="left")
    )  # fmt: skip
    zero_by_tow["excluded_as_unannotated"] = zero_by_tow["rel_dir"].isin(unannotated_dirs)
    zero_by_tow.to_csv(out_dir / "zero_box_files_by_tow.csv", index=False)
    outputs.append(out_dir / "zero_box_files_by_tow.csv")
    excluded = img["rel_dir"].isin(unannotated_dirs) | img["tow"].isin(unannotated_tows)
    n_excluded_unannotated = int(excluded.sum())
    img = img[~excluded].copy()
    bx = bx[bx["key"].isin(img["key"])].copy()

    # Splits at tow level.
    holdout = {str(k): str(v) for k, v in scfg.get("holdout_years", {}).items()}
    forced = {str(k): str(v) for k, v in scfg.get("forced_tows", {}).items()}
    tows = tow_table(img, bx, classes)
    assignment = assign_splits(
        tows,
        classes,
        fractions=dict(scfg.get("fractions", {"train": 0.7, "val": 0.15, "test": 0.15})),
        forced=forced,
        holdout=holdout,
        seed=int(cfg.get("seed", 0)),
        n_restarts=int(scfg.get("n_restarts", 40)),
        n_moves=int(scfg.get("n_moves", 5000)),
        min_tows_for_presence_penalty=int(scfg.get("min_tows_for_presence_penalty", 3)),
        min_tows_per_split={str(k): dict(v) for k, v in scfg.get("min_tows_per_split", {}).items()},
    )
    tows["split"] = tows["tow"].map(assignment)
    img["split"] = img["tow"].map(assignment)

    # Frames recorded during the label-generation transects, found by timestamp.
    windows = {k: (str(v[0]), str(v[1])) for k, v in scfg.get("inference_transect_windows", {}).items()}
    img["inference_transect"] = frames_inside_windows(img, windows) if windows else ""
    protected = sorted(set(img.loc[img["inference_transect"] != "", "tow"]) | set(forced))
    problems = check_split_properties(img, holdout, forced, protected)
    overlap = img[img["inference_transect"] != ""]
    if (overlap["split"] != "train").any():
        problems.append("frames inside inference transect windows outside the training split")
    if problems:
        raise RuntimeError("split rules violated: " + "; ".join(problems))

    # Tables for the report and the repository.
    mapping_table(bx, taxonomy).to_csv(out_dir / "taxonomy_mapping.csv", index=False)
    by_year = bx.groupby(["class_name", "year"]).size().unstack(fill_value=0)
    by_year["total"] = by_year.sum(axis=1)
    by_year = by_year.reindex(classes).fillna(0).astype(int).reset_index()
    by_year.to_csv(out_dir / "class_counts_by_year.csv", index=False)
    b2018 = bx[bx["year"].astype(str) == "2018"]
    by_tow_2018 = b2018.groupby(["station", "class_name"]).size().unstack(fill_value=0)
    by_tow_2018 = by_tow_2018.reindex(columns=classes, fill_value=0)
    by_tow_2018["total"] = by_tow_2018.sum(axis=1)
    by_tow_2018.reset_index().to_csv(out_dir / "class_counts_by_tow_2018.csv", index=False)
    tows.to_csv(out_dir / "tows.csv", index=False)
    resolution = (
        img.groupby(["year", "tow", "split", "width_used", "height_used"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    resolution.to_csv(out_dir / "resolution_by_tow.csv", index=False)
    outputs.append(out_dir / "resolution_by_tow.csv")
    split_cols = [
        "key", "year", "station", "tow", "split", "image_rel", "xml_rel", "status",
        "width_used", "height_used", "n_boxes", "inference_transect",
    ]  # fmt: skip
    splits = img[split_cols].sort_values(["split", "key"]).reset_index(drop=True)
    splits.to_csv(out_dir / "splits.csv", index=False)
    counts_split = class_counts_by_split(img, bx, classes)
    counts_split.to_csv(out_dir / "class_counts_by_split.csv", index=False)
    outputs += [
        out_dir / "taxonomy_mapping.csv", out_dir / "class_counts_by_year.csv",
        out_dir / "class_counts_by_tow_2018.csv", out_dir / "tows.csv", out_dir / "splits.csv",
        out_dir / "class_counts_by_split.csv",
    ]  # fmt: skip

    # COCO export per split.
    coco_stats: dict[str, Any] = {}
    if ecfg.get("coco", True):
        for split_name, sub in img.groupby("split"):
            coco, stats = to_coco(
                sub, bx[bx["key"].isin(sub["key"])], taxonomy,
                clip_boxes=bool(ecfg.get("clip_boxes", True)),
                min_box_side_px=float(ecfg.get("min_box_side_px", 2.0)),
            )  # fmt: skip
            path = write_coco(coco, out_dir / "coco" / f"{split_name}.json")
            outputs.append(path)
            coco_stats[str(split_name)] = {
                "images": len(coco["images"]), "annotations": len(coco["annotations"]), **stats,
            }  # fmt: skip

    hold = img[img["year"].astype(str).isin(holdout.keys())]
    summary: dict[str, Any] = {
        "taxonomy": {"classes": classes, "n_raw_labels_mapped": len(taxonomy.raw_to_class)},
        "n_paired_files": n_paired,
        "n_dropped_without_size": int(no_size.sum()),
        "n_excluded_unannotated_images": n_excluded_unannotated,
        "unannotated_dirs": unannotated_dirs,
        "unannotated_tows": unannotated_tows,
        "n_images": int(len(img)),
        "n_negatives": int((img["n_boxes"] == 0).sum()),
        "n_boxes": int(len(bx)),
        "n_tows": int(len(tows)),
        "tows_per_split": tows["split"].value_counts().to_dict(),
        "images_per_split": img["split"].value_counts().to_dict(),
        "boxes_per_split": bx.merge(img[["key", "split"]], on="key")["split"].value_counts().to_dict(),
        "frames_in_inference_transects": overlap.groupby(["inference_transect", "tow", "split"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
        .to_dict(orient="records"),
        "holdout": {
            "years": holdout,
            "tows": sorted(hold["tow"].unique()),
            "n_images": int(len(hold)),
            "n_boxes": int(bx["key"].isin(hold["key"]).sum()),
            "n_images_readable": int(hold["image_readable"].fillna(False).astype(bool).sum()),
        },
        "coco": coco_stats,
        "split_rule_violations": problems,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "build-annotation-dataset", cfg, inputs, outputs)
    return out_dir
