"""Inventory, pairing and reconciliation of the annotation collection.

Images and XML files live in two parallel trees. They are paired by their path
relative to the respective root with the extension removed, never by basename
alone, because basenames repeat across surveys. A second pass pairs files whose
relative paths differ only in the presence of a camera directory level, which
occurs for one tow; such pairs are marked separately.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..manifest import file_sha256
from .voc import parse_voc_xml

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TOW_RE = re.compile(r"^(\d{8}T\d{4}Z|ST\d+)$")
CAMERA_NAMES = {"Left", "Right", "left", "right"}


def walk_files(
    root: str | Path,
    extensions: tuple[str, ...] | None = None,
    exclude_top: tuple[str, ...] = (),
) -> list[str]:
    """Return sorted POSIX-style paths of all files under ``root``, relative to it.

    Top-level directories named in ``exclude_top`` are not descended into.
    """
    root = Path(root)
    exts = tuple(e.lower() for e in extensions) if extensions else None
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == root:
            dirnames[:] = [d for d in dirnames if d not in exclude_top]
        dirnames.sort()
        for name in filenames:
            if exts is None or name.lower().endswith(exts):
                out.append(Path(dirpath, name).relative_to(root).as_posix())
    return sorted(out)


def strip_extension(rel: str) -> str:
    return PurePosixPath(rel).with_suffix("").as_posix()


def split_key(key: str) -> dict[str, str]:
    """Decompose a relative key into year, tow (station or transect), camera and name.

    The year is the first path component. The tow is the first component that
    looks like a station (``ST12``) or a transect identifier (``20220221T2006Z``);
    when none matches, the second component is used. The camera is the first
    component named Left or Right, or empty.
    """
    parts = PurePosixPath(key).parts
    n = len(parts)
    dirs = parts[:-1]
    tow = next((p for p in dirs[1:] if TOW_RE.match(p)), dirs[1] if n > 2 else "")
    camera = next((p for p in dirs if p in CAMERA_NAMES), "")
    return {
        "year": parts[0] if n > 1 else "",
        "station": tow,
        "camera": camera,
        "rel_dir": "/".join(dirs),
        "basename": parts[-1],
    }


def pair_images_and_xml(
    images_root: str | Path,
    xml_root: str | Path,
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    exclude_top: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Pair image and XML files by relative path.

    Top-level directories named in ``exclude_top`` are skipped in both trees.

    Returns a table with one row per distinct key and the columns ``key``,
    ``image_rel``, ``xml_rel`` and ``status``, plus a list of image paths whose
    key collides with an earlier image (for example a PNG and a JPEG with the
    same stem). Status values:

    - ``paired``: identical relative paths;
    - ``paired_by_stem``: same year, tow and file stem, differing only by a
      camera directory level;
    - ``image_only`` and ``xml_only``: no partner. For ``xml_only`` rows the
      column ``stem_match_elsewhere`` names an image of the same year and stem
      under another tow when exactly one exists; such files are not paired.
    """
    images = walk_files(images_root, image_extensions, exclude_top)
    xmls = walk_files(xml_root, (".xml",), exclude_top)
    image_by_key: dict[str, str] = {}
    duplicates: list[str] = []
    for rel in images:
        key = strip_extension(rel)
        if key in image_by_key:
            duplicates.append(rel)
        else:
            image_by_key[key] = rel
    xml_by_key = {strip_extension(rel): rel for rel in xmls}

    rows: dict[str, dict[str, Any]] = {}
    for key in sorted(set(image_by_key) | set(xml_by_key)):
        has_img = key in image_by_key
        has_xml = key in xml_by_key
        status = "paired" if has_img and has_xml else ("image_only" if has_img else "xml_only")
        rows[key] = {
            "key": key,
            "image_rel": image_by_key.get(key),
            "xml_rel": xml_by_key.get(key),
            "status": status,
            "stem_match_elsewhere": None,
            **split_key(key),
        }

    # Second pass: same year, tow and stem, different directory depth.
    by_year_tow_stem: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    by_year_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in rows.items():
        if row["status"] == "image_only":
            by_year_tow_stem[(row["year"], row["station"], row["basename"])].append(key)
            by_year_stem[(row["year"], row["basename"])].append(key)
    removed: set[str] = set()
    for row in list(rows.values()):
        if row["status"] != "xml_only":
            continue
        candidates = by_year_tow_stem.get((row["year"], row["station"], row["basename"]), [])
        candidates = [c for c in candidates if c not in removed]
        if len(candidates) == 1:
            img_key = candidates[0]
            row["image_rel"] = rows[img_key]["image_rel"]
            row["status"] = "paired_by_stem"
            removed.add(img_key)
            continue
        others = [c for c in by_year_stem.get((row["year"], row["basename"]), []) if c not in removed]
        if len(others) == 1:
            row["stem_match_elsewhere"] = others[0]
    for key in removed:
        del rows[key]

    columns = [
        "key", "image_rel", "xml_rel", "status", "year", "station", "camera", "rel_dir", "basename",
        "stem_match_elsewhere",
    ]  # fmt: skip
    pairs = pd.DataFrame(list(rows.values()), columns=columns)
    counts = pairs["status"].value_counts().to_dict()
    log.info(
        "pairing: %d images, %d xml, statuses %s, %d duplicate image stems",
        len(images), len(xmls), counts, len(duplicates),
    )  # fmt: skip
    return pairs, duplicates


def read_image_size(path: str | Path) -> tuple[int, int] | None:
    """Width and height in pixels from the image header, or None if unreadable."""
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return None


def build_annotation_tables(
    images_root: str | Path,
    xml_root: str | Path,
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    n_workers: int = 16,
    exclude_top: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Parse every XML file, read every paired image header and build inventory tables.

    Returns a dictionary with the pairing table, one row per key (``images``),
    one row per box (``boxes``) and the list of duplicate stems. Box sizes are
    expressed both in pixels and as a fraction of the image area, using the XML
    size when present and the image header otherwise.
    """
    pairs, duplicates = pair_images_and_xml(images_root, xml_root, image_extensions, exclude_top)
    xml_items = [
        (k, Path(xml_root, r)) for k, r in zip(pairs.key, pairs.xml_rel, strict=True) if isinstance(r, str)
    ]
    img_items = [
        (k, Path(images_root, r))
        for k, r in zip(pairs.key, pairs.image_rel, strict=True)
        if isinstance(r, str)
    ]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parsed = dict(
            zip([k for k, _ in xml_items], ex.map(parse_voc_xml, [p for _, p in xml_items]), strict=True)
        )
        sizes = dict(
            zip([k for k, _ in img_items], ex.map(read_image_size, [p for _, p in img_items]), strict=True)
        )
    log.info("parsed %d xml files, read %d image headers", len(parsed), len(sizes))

    ann_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for _, pr in pairs.iterrows():
        key = pr["key"]
        ann = parsed.get(key)
        actual = sizes.get(key)
        has_image = isinstance(pr["image_rel"], str)
        row: dict[str, Any] = {
            "key": key,
            "year": pr["year"],
            "station": pr["station"],
            "camera": pr["camera"],
            "status": pr["status"],
            "image_rel": pr["image_rel"],
            "xml_rel": pr["xml_rel"],
            "image_width": actual[0] if actual else np.nan,
            "image_height": actual[1] if actual else np.nan,
            "image_readable": bool(actual) if has_image else None,
        }
        if ann is None:
            ann_rows.append(row)
            continue
        if ann.has_size:
            width, height = ann.width, ann.height
            size_source = "xml"
        elif actual:
            width, height = actual
            size_source = "image"
        else:
            width, height = None, None
            size_source = "none"
        mismatch = bool(actual and ann.has_size and (ann.width, ann.height) != tuple(actual))
        anomalies = set(ann.anomalies)
        if mismatch:
            anomalies.add("xml_size_differs_from_image")
        if ann.filename and PurePosixPath(ann.filename).stem != pr["basename"]:
            anomalies.add("filename_field_differs")
        row.update(
            {
                "xml_width": ann.width,
                "xml_height": ann.height,
                "xml_depth": ann.depth,
                "width_used": width,
                "height_used": height,
                "size_source": size_source,
                "n_boxes": len(ann.boxes),
                "labels": ";".join(sorted({b.label for b in ann.boxes})),
                "xml_folder_field": ann.folder,
                "xml_filename_field": ann.filename,
                "anomalies": ";".join(sorted(anomalies)),
            }
        )
        ann_rows.append(row)
        area = float(width * height) if width and height else np.nan
        for b in ann.boxes:
            box_rows.append(
                {
                    "key": key,
                    "year": pr["year"],
                    "station": pr["station"],
                    "label_raw": b.label,
                    "label_norm": b.label.strip().lower(),
                    "xmin": b.xmin,
                    "ymin": b.ymin,
                    "xmax": b.xmax,
                    "ymax": b.ymax,
                    "box_w": b.width,
                    "box_h": b.height,
                    "area_frac": (b.width * b.height) / area if area == area and area > 0 else np.nan,
                    "rel_w": b.width / width if width else np.nan,
                    "rel_h": b.height / height if height else np.nan,
                    "difficult": b.difficult,
                    "truncated": b.truncated,
                }
            )
    images = pd.DataFrame(ann_rows)
    boxes = pd.DataFrame(box_rows)
    return {"pairs": pairs, "images": images, "boxes": boxes, "duplicate_image_stems": duplicates}


def reconcile_with_source(
    dest_root: str | Path,
    source_root: str | Path,
    n_workers: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the copied collection with the original it was assembled from.

    Files present in the copy but absent from the source are listed with their
    location, type and size. Each such file is checked against files elsewhere
    in the copy with the same basename and size; when one exists and its hash
    matches, the extra file is a duplicate of it.
    """
    dest_files = walk_files(dest_root)
    source_set = set(walk_files(source_root))
    extras = [f for f in dest_files if f not in source_set]
    missing = sorted(source_set - set(dest_files))
    extra_set = set(extras)

    by_name_size: dict[tuple[str, int], list[str]] = defaultdict(list)
    for rel in dest_files:
        if rel in extra_set:
            continue
        name = PurePosixPath(rel).name
        by_name_size[(name, Path(dest_root, rel).stat().st_size)].append(rel)

    def describe(rel: str) -> dict[str, Any]:
        p = PurePosixPath(rel)
        size = Path(dest_root, rel).stat().st_size
        parts = p.parts
        candidates = by_name_size.get((p.name, size), [])
        same = False
        matched: str | None = None
        if candidates:
            own_hash = file_sha256(Path(dest_root, rel))
            for cand in candidates:
                if file_sha256(Path(dest_root, cand)) == own_hash:
                    same, matched = True, cand
                    break
        return {
            "rel": rel,
            "top": parts[0] if parts else "",
            "second": parts[1] if len(parts) > 1 else "",
            "third": parts[2] if len(parts) > 2 else "",
            "ext": p.suffix.lower(),
            "size": size,
            "n_same_name_size": len(candidates),
            "duplicate_of": matched,
            "identical_copy_exists": same,
        }

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rows = list(ex.map(describe, extras))
    columns = [
        "rel", "top", "second", "third", "ext", "size", "n_same_name_size",
        "duplicate_of", "identical_copy_exists",
    ]  # fmt: skip
    table = pd.DataFrame(rows, columns=columns)
    summary = {
        "n_dest_files": len(dest_files),
        "n_source_files": len(source_set),
        "n_extra_in_dest": len(extras),
        "n_missing_in_dest": len(missing),
        "missing_examples": missing[:20],
        "n_extra_with_identical_copy": int(table["identical_copy_exists"].sum()) if len(table) else 0,
    }
    log.info("reconciliation: %s", {k: v for k, v in summary.items() if k != "missing_examples"})
    return table, summary
