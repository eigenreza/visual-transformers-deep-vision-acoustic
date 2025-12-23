"""Stage 0: audit of the raw data sources.

Three commands inventory the annotation collection, the camera transects and
the backscattering store. Each writes small summary tables and a JSON summary
to its output directory, larger per-item tables as Parquet, and a manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .acoustics.inventory import (
    ping_time_summary,
    sv_statistics,
    transducer_draft_summary,
    transect_coverage,
)
from .acoustics.store import channel_frequency_table, open_sv_store
from .config import resolve_path
from .data.annotations import build_annotation_tables, reconcile_with_source
from .data.transects import (
    join_frames_to_metadata,
    list_camera_frames,
    parse_deepvision_xml,
    summarize_timestamps,
)
from .manifest import describe_input, manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp | np.datetime64):
        return str(o)
    return str(o)


def _dump_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _quantiles(
    s: pd.Series, qs: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    out = {f"q{int(q * 100):02d}": float(s.quantile(q)) for q in qs}
    out["min"], out["max"], out["mean"], out["n"] = (
        float(s.min()),
        float(s.max()),
        float(s.mean()),
        int(s.size),
    )
    return out


def _explode_anomalies(ann: pd.DataFrame) -> pd.DataFrame:
    exploded = ann[["year", "anomalies"]].copy()
    exploded["anomaly"] = exploded["anomalies"].fillna("").str.split(";")
    exploded = exploded.explode("anomaly")
    exploded = exploded[exploded["anomaly"].astype(str).str.len() > 0]
    counts = exploded.groupby(["anomaly", "year"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    return counts.reset_index()


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def run_audit_annotations(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    images_root = resolve_path(cfg, "annotated_data", "images")
    xml_root = resolve_path(cfg, "annotated_data", "xml")
    inputs: dict[str, str | Path] = {"images": images_root, "xml": xml_root}
    source_root = cfg.get("paths", {}).get("annotated_data_source")
    acfg = cfg.get("annotations", {})
    do_reconcile = bool(
        acfg.get("reconcile_with_source", False) and source_root and Path(source_root).is_dir()
    )
    if do_reconcile:
        inputs["source"] = Path(str(source_root))
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("annotation audit is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = int(cfg.get("n_workers", 16))
    exts = tuple(acfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))

    exclude_top = tuple(acfg.get("exclude_top_level_dirs", []))
    tables = build_annotation_tables(
        images_root, xml_root, exts, n_workers=n_workers, exclude_top=exclude_top
    )
    pairs, images, boxes = tables["pairs"], tables["images"], tables["boxes"]
    outputs: list[str | Path] = []

    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    images.to_parquet(out_dir / "images.parquet", index=False)
    boxes.to_parquet(out_dir / "boxes.parquet", index=False)
    outputs += [out_dir / "pairs.parquet", out_dir / "images.parquet", out_dir / "boxes.parquet"]

    pairing_by_year = pairs.groupby(["year", "status"]).size().unstack(fill_value=0).reset_index()
    pairing_by_year.to_csv(out_dir / "pairing_by_year.csv", index=False)
    pairing_by_station = (
        pairs.groupby(["year", "station", "status"]).size().unstack(fill_value=0).reset_index()
    )
    pairing_by_station.to_csv(out_dir / "pairing_by_station.csv", index=False)
    outputs += [out_dir / "pairing_by_year.csv", out_dir / "pairing_by_station.csv"]

    if not boxes.empty:
        labels_by_year = boxes.groupby(["label_raw", "year"]).size().unstack(fill_value=0)
        labels_by_year["total"] = labels_by_year.sum(axis=1)
        labels_by_year = labels_by_year.sort_values("total", ascending=False).reset_index()
        labels_by_year.to_csv(out_dir / "labels_by_year.csv", index=False)
        labels_by_station = (
            boxes.groupby(["label_raw", "year", "station"]).size().rename("n_boxes").reset_index()
        )
        labels_by_station.to_csv(out_dir / "labels_by_station.csv", index=False)
        box_size = (
            boxes.groupby("label_raw")
            .agg(
                n=("box_w", "size"),
                w_px_median=("box_w", "median"),
                h_px_median=("box_h", "median"),
                w_px_q05=("box_w", lambda s: s.quantile(0.05)),
                w_px_q95=("box_w", lambda s: s.quantile(0.95)),
                h_px_q05=("box_h", lambda s: s.quantile(0.05)),
                h_px_q95=("box_h", lambda s: s.quantile(0.95)),
                area_frac_median=("area_frac", "median"),
                area_frac_q05=("area_frac", lambda s: s.quantile(0.05)),
                area_frac_q95=("area_frac", lambda s: s.quantile(0.95)),
                rel_w_median=("rel_w", "median"),
                rel_h_median=("rel_h", "median"),
            )
            .sort_values("n", ascending=False)
            .reset_index()
        )
        box_size.to_csv(out_dir / "box_size_by_label.csv", index=False)
        outputs += [
            out_dir / "labels_by_year.csv",
            out_dir / "labels_by_station.csv",
            out_dir / "box_size_by_label.csv",
        ]

    ann = images[images["xml_rel"].notna()] if "xml_rel" in images else images.iloc[0:0]
    if not ann.empty:
        sizes = (
            ann.groupby(["year", "width_used", "height_used", "size_source"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        sizes.to_csv(out_dir / "image_sizes.csv", index=False)
        bpi = ann.groupby(["year", "n_boxes"]).size().unstack(fill_value=0).reset_index()
        bpi.to_csv(out_dir / "boxes_per_image_by_year.csv", index=False)
        _explode_anomalies(ann).to_csv(out_dir / "anomalies_by_year.csv", index=False)
        outputs += [
            out_dir / "image_sizes.csv",
            out_dir / "boxes_per_image_by_year.csv",
            out_dir / "anomalies_by_year.csv",
        ]

    summary: dict[str, Any] = {
        "n_keys": int(len(pairs)),
        "status_counts": pairs["status"].value_counts().to_dict(),
        "n_duplicate_image_stems": len(tables["duplicate_image_stems"]),
        "duplicate_image_stem_examples": tables["duplicate_image_stems"][:10],
        "n_boxes": int(len(boxes)),
        "n_raw_labels": int(boxes["label_raw"].nunique()) if not boxes.empty else 0,
        "raw_labels": boxes["label_raw"].value_counts().to_dict() if not boxes.empty else {},
        "n_annotations": int(len(ann)),
        "n_annotations_missing_size": int((ann["size_source"] != "xml").sum()) if not ann.empty else 0,
        "n_annotations_size_filled_from_image": int((ann["size_source"] == "image").sum())
        if not ann.empty
        else 0,
        "n_annotations_size_unavailable": int((ann["size_source"] == "none").sum()) if not ann.empty else 0,
        "n_unreadable_images": int((images["image_readable"] == False).sum()),  # noqa: E712
        "boxes_per_image": _quantiles(ann["n_boxes"]) if not ann.empty else {},
        "box_area_fraction": _quantiles(boxes["area_frac"]) if not boxes.empty else {},
        "box_width_px": _quantiles(boxes["box_w"]) if not boxes.empty else {},
        "box_height_px": _quantiles(boxes["box_h"]) if not boxes.empty else {},
    }

    if do_reconcile:
        dest_root = resolve_path(cfg, "annotated_data")
        cache = out_dir / "reconciliation_cache.json"
        extras_path = out_dir / "extra_files.csv"
        cache_key = {"dest": describe_input(dest_root), "source": describe_input(str(source_root))}
        cached = None
        if cache.is_file() and extras_path.is_file():
            with open(cache, encoding="utf-8") as fh:
                cached = json.load(fh)
        if cached and cached.get("key") == cache_key:
            log.info("reusing reconciliation from %s", cache)
            extras = pd.read_csv(extras_path, keep_default_na=False, na_values=[""])
            extras["identical_copy_exists"] = (
                extras["identical_copy_exists"].astype(str).str.lower() == "true"
            )
            rsum = cached["summary"]
        else:
            extras, rsum = reconcile_with_source(dest_root, str(source_root), n_workers=n_workers)
            extras.to_csv(extras_path, index=False)
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump({"key": cache_key, "summary": rsum}, fh, indent=2, default=_json_default)
        extras_summary = (
            extras.groupby(["top", "second", "ext", "identical_copy_exists"])
            .agg(n=("rel", "size"), total_mb=("size", lambda s: float(s.sum()) / 1e6))
            .reset_index()
        )
        extras_summary.to_csv(out_dir / "extra_files_summary.csv", index=False)
        outputs += [out_dir / "extra_files.csv", out_dir / "extra_files_summary.csv"]
        extra_keys = set()
        for rel in extras["rel"]:
            parts = rel.split("/", 1)
            if len(parts) == 2:
                extra_keys.add(parts[1].rsplit(".", 1)[0])
        in_pairs = pairs[pairs["key"].isin(extra_keys)]
        rsum["extra_key_status_counts"] = in_pairs["status"].value_counts().to_dict()
        summary["reconciliation"] = rsum
    else:
        summary["reconciliation"] = "skipped"

    _dump_json(summary, out_dir / "summary.json")
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "audit-annotations", cfg, inputs, outputs)
    return out_dir


# ---------------------------------------------------------------------------
# Transects
# ---------------------------------------------------------------------------


def _folder_table(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    g = frames.groupby(["folder", "kind", "is_zip"], sort=True)
    table = g.agg(
        n_files=("filename", "size"),
        n_images=("is_image", "sum"),
        total_mb=("size_bytes", lambda s: float(s.sum()) / 1e6),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        n_without_timestamp=("timestamp", lambda s: int(s.isna().sum())),
    ).reset_index()
    med = []
    for _, sub in g:
        t = sub["timestamp"].dropna().sort_values()
        d = t.diff().dt.total_seconds().dropna()
        med.append(float(d.median()) if len(d) else np.nan)
    table["median_interval_s"] = med
    return table


def _metadata_summary(
    meta: pd.DataFrame, info: dict[str, Any], meta_path: Path, tcfg: dict[str, Any]
) -> dict[str, Any]:
    gap_thr = tuple(tcfg.get("gap_thresholds_s", [1, 5, 60]))
    dmin, dmax = tcfg.get("depth_valid_range_m", [0.5, 1500])
    out: dict[str, Any] = {
        "path": str(meta_path),
        "n_frames": info["n_frames"],
        "attributes": info["attributes"],
        "fileformat": info.get("fileformat"),
        "decimal_comma_values": info.get("decimal_comma_values", {}),
        "info_groups": info["groups"],
        "timestamps": summarize_timestamps(meta["timestamp"], gap_thr),
    }
    if "folder" in meta.columns:
        out["folders"] = meta["folder"].value_counts().sort_index().to_dict()
    if "active_flag" in meta.columns:
        out["active_flag_counts"] = {
            str(k): int(v) for k, v in meta["active_flag"].value_counts(dropna=False).items()
        }
    if "depth_num" in meta.columns:
        d = meta["depth_num"]
        valid = d.between(dmin, dmax)
        out["depth"] = {
            "n_missing_or_non_numeric": int(d.isna().sum()),
            "n_out_of_range": int((~valid & d.notna()).sum()),
            "n_valid": int(valid.sum()),
            "quantiles_valid_m": _quantiles(d[valid]),
            "n_exact_zero": int((d == 0).sum()),
            "n_steps_over_10m_between_consecutive_frames": int((d.diff().abs() > 10).sum()),
            "n_steps_over_50m_between_consecutive_frames": int((d.diff().abs() > 50).sum()),
        }
    for col in ("lat_num", "lon_num", "rawdepth_num", "temp_num", "voltage_num"):
        if col in meta.columns:
            out[col] = _quantiles(meta[col])
    return out


def _camera_summary(
    frames: pd.DataFrame, meta: pd.DataFrame | None, tcfg: dict[str, Any], name: str, cam: str, out_dir: Path
) -> tuple[dict[str, Any], list[Path]]:
    tol = int(tcfg.get("join_tolerance_ms", 50))
    gap_thr = tuple(tcfg.get("gap_thresholds_s", [1, 5, 60]))
    dmin, dmax = tcfg.get("depth_valid_range_m", [0.5, 1500])
    outputs: list[Path] = []
    folders = _folder_table(frames)
    folders.to_csv(out_dir / f"{name}_{cam}_folders.csv", index=False)
    outputs.append(out_dir / f"{name}_{cam}_folders.csv")
    by_kind = (
        frames.groupby(["kind", "is_zip"])
        .agg(
            n_folders=("folder", "nunique"),
            n_files=("filename", "size"),
            n_images=("is_image", "sum"),
            total_mb=("size_bytes", lambda s: float(s.sum()) / 1e6),
        )
        .reset_index()
    )
    summary: dict[str, Any] = {
        "n_files": int(len(frames)),
        "n_folders": int(frames["folder"].nunique()),
        "by_kind": by_kind.to_dict(orient="records"),
        "n_empty_kind_folders_with_images": int(
            frames[(frames["kind"] == "empty") & frames["is_image"]]["folder"].nunique()
        ),
        "non_image_files": frames.loc[~frames["is_image"], "filename"].head(20).tolist(),
        "n_archives_duplicating_extracted_folders": int(
            frames.loc[frames["archive_duplicate"], "folder"].nunique()
        ),
    }
    img = frames[frames["is_image"] & ~frames["archive_duplicate"]]
    summary["timestamps_all_images"] = summarize_timestamps(img["timestamp"], gap_thr)
    summary["timestamps_active_images"] = summarize_timestamps(
        img[img["kind"] == "active"]["timestamp"], gap_thr
    )
    summary["file_size_bytes"] = _quantiles(img["size_bytes"])
    if meta is not None and not img.empty:
        joined = join_frames_to_metadata(img, meta, tolerance_ms=tol)
        matched = joined["matched"]
        summary["join"] = {
            "tolerance_ms": tol,
            "n_frames": int(len(joined)),
            "n_matched": int(matched.sum()),
            "n_unmatched": int((~matched).sum()),
            "n_folder_disagreements": int((matched & ~joined["folder_agrees"]).sum()),
            "n_metadata_rows_used_more_than_once": int(joined.loc[matched, "meta_row"].duplicated().sum()),
            "n_metadata_rows_without_frame": int(len(meta) - joined.loc[matched, "meta_row"].nunique()),
            "unmatched_examples": joined.loc[~matched, ["folder", "filename"]]
            .head(10)
            .to_dict(orient="records"),
        }
        if "meta_depth_num" in joined.columns:
            d = joined.loc[matched, "meta_depth_num"]
            summary["join"]["depth_of_matched_frames"] = {
                "n_missing": int(d.isna().sum()),
                "n_out_of_range": int((~d.between(dmin, dmax) & d.notna()).sum()),
                "quantiles_m": _quantiles(d[d.between(dmin, dmax)]),
            }
        keep_meta = {
            "meta_folder",
            "meta_depth_num",
            "meta_row",
            "meta_active_flag",
            "meta_lat_num",
            "meta_lon_num",
        }
        keep = [c for c in joined.columns if not c.startswith("meta_") or c in keep_meta]
        joined[keep].to_parquet(out_dir / f"{name}_{cam}_frames.parquet", index=False)
    else:
        img.to_parquet(out_dir / f"{name}_{cam}_frames.parquet", index=False)
    outputs.append(out_dir / f"{name}_{cam}_frames.parquet")
    return summary, outputs


def _audit_one_transect(
    transect_dir: Path, tcfg: dict[str, Any], out_dir: Path
) -> tuple[dict[str, Any], list[Path]]:
    name = transect_dir.name
    outputs: list[Path] = []
    result: dict[str, Any] = {"transect": name, "exists": transect_dir.is_dir()}
    if not transect_dir.is_dir():
        return result, outputs
    result["top_level_entries"] = sorted(p.name + ("/" if p.is_dir() else "") for p in transect_dir.iterdir())

    meta_path = transect_dir / tcfg.get("metadata_file", "deepvision.xml")
    meta: pd.DataFrame | None = None
    if meta_path.is_file():
        meta, info = parse_deepvision_xml(meta_path)
        result["metadata"] = _metadata_summary(meta, info, meta_path, tcfg)
        meta.to_parquet(out_dir / f"{name}_metadata.parquet", index=False)
        outputs.append(out_dir / f"{name}_metadata.parquet")
    else:
        result["metadata"] = {"path": str(meta_path), "missing": True}

    cameras: dict[str, Any] = {}
    for cam in tcfg.get("cameras", ["Left", "Right"]):
        frames = list_camera_frames(transect_dir, cam)
        if frames.empty:
            cameras[cam] = {"n_files": 0, "present": (transect_dir / cam).is_dir()}
            continue
        cam_summary, outs = _camera_summary(frames, meta, tcfg, name, cam, out_dir)
        cameras[cam] = cam_summary
        outputs += outs
    result["cameras"] = cameras
    return result, outputs


def run_audit_transects(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    tcfg = cfg.get("transects", {})
    ids = list(tcfg.get("ids_2018", [])) + list(tcfg.get("ids_2024", []))
    inputs: dict[str, str | Path] = {t: resolve_path(cfg, "source_data", t) for t in ids}
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("transect audit is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str | Path] = []
    results = []
    overview_rows = []
    for t in ids:
        log.info("auditing transect %s", t)
        res, outs = _audit_one_transect(Path(inputs[t]), tcfg, out_dir)
        results.append(res)
        outputs += outs
        row: dict[str, Any] = {"transect": t, "exists": res.get("exists")}
        md = res.get("metadata", {})
        row["meta_frames"] = md.get("n_frames")
        row["meta_depth_valid"] = md.get("depth", {}).get("n_valid")
        row["meta_depth_missing"] = md.get("depth", {}).get("n_missing_or_non_numeric")
        for cam, cs in res.get("cameras", {}).items():
            row[f"{cam}_files"] = cs.get("n_files")
            row[f"{cam}_folders"] = cs.get("n_folders")
            ts = cs.get("timestamps_active_images", {})
            row[f"{cam}_active_images"] = ts.get("n")
            row[f"{cam}_first"] = ts.get("first")
            row[f"{cam}_last"] = ts.get("last")
            row[f"{cam}_median_interval_s"] = ts.get("interval_s", {}).get("p50")
            j = cs.get("join", {})
            row[f"{cam}_matched"] = j.get("n_matched")
            row[f"{cam}_unmatched"] = j.get("n_unmatched")
        overview_rows.append(row)
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(out_dir / "transects_overview.csv", index=False)
    outputs.append(out_dir / "transects_overview.csv")
    _dump_json({"transects": results}, out_dir / "summary.json")
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "audit-transects", cfg, inputs, outputs)
    return out_dir


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------


def run_audit_zarr(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    zcfg = cfg.get("zarr", {})
    store = resolve_path(cfg, "source_data", zcfg.get("store", "S2018105_sv.zarr"))
    inputs: dict[str, str | Path] = {"store": store}
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("zarr audit is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str | Path] = []
    ds = open_sv_store(store)
    summary: dict[str, Any] = {
        "store": str(store),
        "attrs": dict(ds.attrs),
        "dims": dict(ds.sizes),
        "variables": {
            k: {
                "dims": list(ds[k].dims),
                "dtype": str(ds[k].dtype),
                "shape": list(ds[k].shape),
                "attrs": dict(ds[k].attrs),
            }
            for k in list(ds.coords) + list(ds.data_vars)
        },
        "sv_chunks": [c[0] for c in ds["sv"].chunks] if ds["sv"].chunks else None,
    }

    chan = channel_frequency_table(ds)
    chan.to_csv(out_dir / "channel_frequency.csv", index=False)
    outputs.append(out_dir / "channel_frequency.csv")
    summary["channel_frequency_agreement"] = bool(chan["agrees"].all())
    summary["frequencies_hz"] = chan["frequency_hz"].tolist()

    rng = ds["range"].values
    steps = np.diff(rng)
    summary["range"] = {
        "min_m": float(rng.min()),
        "max_m": float(rng.max()),
        "n": int(rng.size),
        "step_m_median": float(np.median(steps)),
        "step_m_min": float(steps.min()),
        "step_m_max": float(steps.max()),
    }

    pt = ds["ping_time"].values
    ping_summary, gaps = ping_time_summary(pt, tuple(zcfg.get("gap_thresholds_s", [5, 10, 60, 600])))
    summary["ping_time"] = ping_summary
    gaps.head(200).to_csv(out_dir / "ping_gaps.csv", index=False)
    outputs.append(out_dir / "ping_gaps.csv")

    windows = {k: (str(v[0]), str(v[1])) for k, v in zcfg.get("transect_windows", {}).items()}
    cov = transect_coverage(ds, windows)
    cov.to_csv(out_dir / "transect_coverage.csv", index=False)
    outputs.append(out_dir / "transect_coverage.csv")

    draft_summary, draft_table = transducer_draft_summary(ds)
    summary["transducer_draft"] = draft_summary
    draft_table.to_csv(out_dir / "transducer_draft.csv", index=False)
    outputs.append(out_dir / "transducer_draft.csv")

    pl = ds["pulse_length"].transpose("frequency", "ping_time").values
    pulse: dict[str, dict[str, int]] = {}
    for i, f in enumerate(ds["frequency"].values):
        vals, counts = np.unique(pl[i][np.isfinite(pl[i])], return_counts=True)
        pulse[str(float(f))] = {f"{u:g}": int(c) for u, c in zip(vals, counts, strict=True)}
    summary["pulse_length_s"] = pulse

    lat, lon = ds["latitude"].values, ds["longitude"].values
    summary["position"] = {
        "lat_nan": int(np.isnan(lat).sum()),
        "lon_nan": int(np.isnan(lon).sum()),
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
    }
    summary["raw_files"] = {"n_unique": int(np.unique(ds["raw_file"].values).size)}

    floors = tuple(float(f) for f in zcfg.get("floors_db", [-100, -90, -80]))
    hist_range = zcfg.get("histogram_range_db", [-130, 0])
    stats = sv_statistics(
        ds,
        floors,
        (float(hist_range[0]), float(hist_range[1])),
        float(zcfg.get("histogram_bin_db", 1.0)),
        n_workers=int(cfg.get("n_workers", 32)),
    )
    freqs = ds["frequency"].values
    n_pings, n_range = stats["n_pings"], stats["n_range"]
    total = float(n_pings * n_range)

    per_freq = []
    for i, f in enumerate(freqs):
        row: dict[str, Any] = {
            "frequency_hz": float(f),
            "nan_fraction": float(stats["nan_by_range"][i].sum() / total),
            "zero_fraction": float(stats["zero_by_range"][i].sum() / total),
            "negative_fraction": float(stats["negative_by_range"][i].sum() / total),
        }
        for f_db in floors:
            row[f"at_or_below_{int(f_db)}dB_fraction"] = float(
                stats[f"floor{int(f_db)}_by_range"][i].sum() / total
            )
        nan_frac_by_range = stats["nan_by_range"][i] / n_pings
        ok = np.nonzero(nan_frac_by_range < 0.5)[0]
        row["last_range_m_with_under_50pct_missing"] = float(rng[ok.max()]) if ok.size else np.nan
        pings_all_nan = int((stats["nan_by_ping"][i] == n_range).sum())
        row["pings_entirely_missing"] = pings_all_nan
        row["pings_entirely_missing_fraction"] = pings_all_nan / n_pings
        h = stats["hist"][i]
        edges = stats["hist_edges"]
        if h.sum() > 0:
            c = np.cumsum(h) / h.sum()
            row["db_p05_in_hist_range"] = float(edges[np.searchsorted(c, 0.05)])
            row["db_median_in_hist_range"] = float(edges[np.searchsorted(c, 0.5)])
            row["db_p95_in_hist_range"] = float(edges[np.searchsorted(c, 0.95)])
            row["fraction_in_hist_range"] = float(h.sum() / total)
        per_freq.append(row)
    pd.DataFrame(per_freq).to_csv(out_dir / "sv_per_frequency.csv", index=False)
    outputs.append(out_dir / "sv_per_frequency.csv")
    summary["sv_per_frequency"] = per_freq

    cols: dict[str, Any] = {"range_m": rng}
    for i, f in enumerate(freqs):
        cols[f"nan_frac_{int(f)}"] = stats["nan_by_range"][i] / n_pings
        for f_db in floors:
            cols[f"floor{int(f_db)}_frac_{int(f)}"] = stats[f"floor{int(f_db)}_by_range"][i] / n_pings
    pd.DataFrame(cols).to_csv(out_dir / "sv_by_range.csv", index=False)
    outputs.append(out_dir / "sv_by_range.csv")

    hist_cols: dict[str, Any] = {"db_low": stats["hist_edges"][:-1], "db_high": stats["hist_edges"][1:]}
    for i, f in enumerate(freqs):
        hist_cols[f"count_{int(f)}"] = stats["hist"][i]
    pd.DataFrame(hist_cols).to_csv(out_dir / "sv_db_histogram.csv", index=False)
    outputs.append(out_dir / "sv_db_histogram.csv")

    per_ping: dict[str, Any] = {"ping_time": pt}
    for i, f in enumerate(freqs):
        per_ping[f"nan_{int(f)}"] = stats["nan_by_ping"][i].astype(np.uint16)
        for f_db in floors:
            per_ping[f"floor{int(f_db)}_{int(f)}"] = stats[f"floor{int(f_db)}_by_ping"][i].astype(np.uint16)
    pd.DataFrame(per_ping).to_parquet(out_dir / "sv_per_ping.parquet", index=False)
    outputs.append(out_dir / "sv_per_ping.parquet")

    _dump_json(summary, out_dir / "summary.json")
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "audit-zarr", cfg, inputs, outputs)
    return out_dir
