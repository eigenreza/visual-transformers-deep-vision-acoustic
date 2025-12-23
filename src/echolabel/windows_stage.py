"""Stage 7 commands: build the window dataset per geometry and make the splits.

``build-windows`` reads, per transect, the preprocessed store, the alignment
mapping and the camera series of one detector run, cuts the patches of every
geometry at the lag-corrected vessel time of each window, and writes the
arrays to the scratch directory and the metadata and targets to the project
directory. ``make-splits`` assigns the windows of every geometry to the
leave-one-transect-out folds, the three-transect variant and the
within-transect blocks, checks the leakage rules, and writes the assignment.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import resolve_path
from .manifest import manifest_is_current, write_manifest
from .windows.build import Geometry, TransectPatches, band_top, window_grid
from .windows.splits import (
    block_splits,
    buffer_by_transect,
    buffer_by_window,
    buffer_seconds,
    check_splits,
    loto_folds,
    split_counts,
)
from .windows.targets import build_targets, log1p_targets, summarize_targets

log = logging.getLogger(__name__)


def _classes_from_series(series: pd.DataFrame, suffix: str, exclude: list[str]) -> list[str]:
    names = sorted(c[: -len(suffix)] for c in series.columns if c.endswith(suffix) and not c.startswith("n_"))
    return [c for c in names if c not in exclude]


def _write_arrays(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> None:
    """Write the patch arrays of one geometry as a Zarr group chunked by window."""
    import zarr

    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    for name, arr in arrays.items():
        chunks = (min(256, arr.shape[0]),) + tuple(arr.shape[1:])
        a = g.create_array(name, shape=arr.shape, chunks=chunks, dtype=arr.dtype)
        a[:] = arr
    g.attrs.update(attrs)


def build_geometry(
    geometry: Geometry,
    patches: TransectPatches,
    mapping: pd.DataFrame,
    series: pd.DataFrame,
    exclusion_depth_m: float,
    wcfg: dict[str, Any],
    tcfg: dict[str, Any],
    context_patches: TransectPatches | None = None,
    frames: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Patches, metadata and targets of one geometry for one transect.

    ``frames`` is the per-frame count table of the transect; with it the
    per-frame target forms named in the target configuration are built.
    """
    grid = window_grid(mapping, geometry.stride_s, geometry.length_s)
    n_time = int(round(geometry.length_s / patches.time_step_s))
    n_depth = int(round(geometry.height_m / patches.depth_step_m))
    # Nominal number of pings of a window, from the median ping interval rounded to whole
    # pings per time cell; a window with fewer than the configured share of it is dropped.
    expected_pings = n_time * patches.ping_factor
    keep_rows, sv, valid, hard, depth_cells = [], [], [], [], []
    ctx_sv, ctx_valid, ctx_depth = [], [], []
    half = np.timedelta64(int(geometry.length_s / 2 * 1e9), "ns")
    for i, row in grid.iterrows():
        d = row["camera_depth_m"]
        if not np.isfinite(d):
            continue
        top = band_top(float(d), geometry.height_m, geometry.anchor, exclusion_depth_m)
        tv = np.datetime64(row["time_vessel"], "ns")
        patch = patches.patch(tv - half, tv + half, top, geometry.height_m, n_time, n_depth)
        if patch is None or patch["n_pings"] < wcfg.get("min_ping_fraction", 0.8) * expected_pings:
            continue
        if context_patches is not None and geometry.context:
            c_n_time = int(round(geometry.length_s / context_patches.time_step_s))
            c_n_depth = int(
                round(float(geometry.context.get("depth_max_m", 500)) / context_patches.depth_step_m)
            )
            ctx = context_patches.patch(
                tv - half,
                tv + half,
                float(context_patches.depth.min()),
                float(geometry.context.get("depth_max_m", 500)),
                c_n_time,
                c_n_depth,
            )
            if ctx is None:
                continue
            ctx_sv.append(ctx["sv_db"])
            ctx_valid.append(ctx["valid"])
            ctx_depth.append(ctx["depth_cells"])
        keep_rows.append(i)
        sv.append(patch["sv_db"])
        valid.append(patch["valid"])
        hard.append(patch["hard"])
        depth_cells.append(patch["depth_cells"])
        grid.loc[i, "band_top_m"] = top
        grid.loc[i, "n_pings"] = patch["n_pings"]
    meta = grid.loc[keep_rows].reset_index(drop=True)
    if meta.empty:
        return {"meta": meta, "targets": pd.DataFrame(), "arrays": {}}
    forms = dict(tcfg.get("forms", {"hard": "_res", "weighted": "_weighted", "expected": "_exp"}))
    classes = _classes_from_series(series, next(iter(forms.values())), list(tcfg.get("exclude", [])))
    mapped = str(meta["lag_distribution"].iloc[0]) if "lag_distribution" in meta else "normal"
    if geometry.target_span_s is not None:
        span_s = float(geometry.target_span_s)
        distribution = mapped if geometry.target_lag == "distribution" else "fixed"
    else:
        distribution, span_s = mapped, geometry.length_s
    targets, offsets_record = build_targets(
        series,
        meta,
        span_s,
        forms,
        classes,
        {k: list(v) for k, v in (tcfg.get("merged") or {}).items()},
        tuple(float(x) for x in tcfg.get("lag_offsets_sd", (-2, -1, 0, 1, 2))),
        distribution=distribution,
        n_uniform_offsets=int(tcfg.get("n_uniform_offsets", 9)),
        frames=frames,
        frame_forms=tuple(tcfg.get("frame_forms", ())),
        frame_suffix=str(tcfg.get("frame_suffix", "_res")),
    )
    targets = pd.concat([targets, log1p_targets(targets, tuple(f"{f}_" for f in forms))], axis=1)
    meta["target_distribution"] = distribution
    arrays = {
        "sv_db": np.stack(sv).astype(np.float16),
        "valid": np.stack(valid).astype(np.uint8),
        "hard": np.stack(hard).astype(np.uint8),
        "depth_cells": np.stack(depth_cells).astype(np.float32),
    }
    if ctx_sv:
        arrays["context_sv_db"] = np.stack(ctx_sv).astype(np.float16)
        arrays["context_valid"] = np.stack(ctx_valid).astype(np.uint8)
        arrays["context_depth_cells"] = np.stack(ctx_depth).astype(np.float32)
    return {
        "meta": meta,
        "targets": targets,
        "arrays": arrays,
        "classes": classes,
        "lag_offsets": offsets_record,
    }


def _load_store(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    ds = xr.open_zarr(str(path))
    sv = ds["sv_db"].values
    mask = ds["mask"].values
    return (
        sv,
        mask,
        ds["ping_time"].values,
        ds["depth"].values,
        ds["frequency"].values,
        float(ds.attrs.get("draft_m", 8.5)),
        ds["surface_exclusion_m"].values if "surface_exclusion_m" in ds else None,
    )


def run_tag(experiment: str, seed: int, variant: str = "") -> str:
    """Name of a window run: detector experiment and seed, with the lag-model variant as a suffix."""
    return f"{experiment}_seed{seed}" + (f"_{variant}" if variant else "")


def run_build_windows(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    variant: str = "",
) -> Path:
    """Build every geometry of the window dataset for one detector run.

    ``variant`` names a lag model whose mapping (``mapping_<variant>.parquet``)
    replaces the configured one; its windows are written under a run tag
    with the variant as a suffix.
    """
    wcfg, tcfg = cfg.get("windows", {}), cfg.get("targets", {})
    project = resolve_path(cfg, "project_dir")
    tag = run_tag(experiment, seed, variant or "")
    scratch = resolve_path(cfg, "scratch_dir", "windows", tag)
    target = out_dir / tag
    transects = list(cfg.get("transects", []))
    interval_s = int(cfg.get("interval_s", 10))
    inputs: dict[str, str | Path] = {}
    for t in transects:
        inputs[f"mapping_{t}"] = Path(
            project,
            "runs",
            "alignment",
            f"{experiment}_seed{seed}",
            t,
            "Left",
            f"mapping_{variant}.parquet" if variant else "mapping.parquet",
        )
        inputs[f"series_{t}"] = Path(
            project,
            "runs",
            "camera_series",
            f"{experiment}_seed{seed}",
            t,
            "Left",
            f"series_{interval_s}s.parquet",
        )
        if tcfg.get("frame_forms"):
            inputs[f"frames_{t}"] = Path(
                project, "runs", "inference", f"{experiment}_seed{seed}", t, "Left", "frames.parquet"
            )
    run_cfg = {**cfg, "experiment": experiment, "seed": seed, "variant": variant or ""}
    if not force and manifest_is_current(target, run_cfg, inputs):
        log.info("window dataset is up to date in %s", target)
        return target
    missing = [str(p) for p in inputs.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"inputs missing: {missing[:3]}")
    target.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    geometries = [Geometry.from_config(name, g) for name, g in wcfg.get("geometries", {}).items()]
    primary_hz = float(wcfg.get("primary_frequency_hz", 38000))
    per_geometry: dict[str, dict[str, Any]] = {
        g.name: {"meta": [], "targets": [], "arrays": [], "lag_offsets": {}} for g in geometries
    }
    summary: dict[str, Any] = {"experiment": experiment, "seed": seed, "transects": {}, "geometries": {}}
    outputs: list[str | Path] = []
    for t in transects:
        t_start = time.time()
        store_path = resolve_path(cfg, "scratch_dir", "acoustics", "transects", f"{t}.zarr")
        sv, mask, ping_time, depth, freqs, draft_m, exclusions = _load_store(store_path)
        ds_pos = int(np.argmin(np.abs(freqs - primary_hz)))
        excl_depth = float(exclusions[ds_pos]) + draft_m if exclusions is not None else draft_m + 5.0
        patches = TransectPatches(
            sv,
            mask,
            ping_time,
            depth,
            float(wcfg.get("time_step_s", 2.0)),
            float(wcfg.get("depth_step_m", 0.5)),
        )
        mapping = pd.read_parquet(inputs[f"mapping_{t}"])
        series = pd.read_parquet(inputs[f"series_{t}"])
        frames = pd.read_parquet(inputs[f"frames_{t}"]) if f"frames_{t}" in inputs else None
        summary["transects"][t] = {
            "n_pings": patches.n_pings,
            "exclusion_depth_m": excl_depth,
            "n_intervals": int(len(mapping)),
        }
        for g in geometries:
            ctx = None
            if g.context:
                ctx = TransectPatches(
                    sv,
                    mask,
                    ping_time,
                    depth,
                    float(g.context.get("time_step_s", 10.0)),
                    float(g.context.get("depth_step_m", 5.0)),
                )
            built = build_geometry(g, patches, mapping, series, excl_depth, wcfg, tcfg, ctx, frames)
            if built["meta"].empty:
                log.warning("%s %s: no windows", t, g.name)
                continue
            meta = built["meta"]
            meta.insert(0, "transect", t)
            meta.insert(1, "geometry", g.name)
            meta["length_s"] = g.length_s
            meta["height_m"] = g.height_m
            meta["target_span_s"] = g.target_span_s if g.target_span_s is not None else g.length_s
            per_geometry[g.name]["lag_offsets"] = built["lag_offsets"]
            meta["anchor"] = g.anchor
            per_geometry[g.name]["meta"].append(meta)
            per_geometry[g.name]["targets"].append(built["targets"])
            per_geometry[g.name]["arrays"].append(built["arrays"])
            summary["transects"][t][g.name] = int(len(meta))
        log.info("%s: %d pings, windows built in %.0f s", t, patches.n_pings, time.time() - t_start)
    frame_min = float(wcfg.get("min_frame_fraction", 0.5))
    flag_col = f"camera_in_exclusion_{int(primary_hz)}"
    usable_col = f"band_usable_fraction_{int(primary_hz)}"
    for g in geometries:
        parts = per_geometry[g.name]
        if not parts["meta"]:
            continue
        meta = pd.concat(parts["meta"], ignore_index=True)
        targets = pd.concat(parts["targets"], ignore_index=True)
        meta.insert(0, "window_id", np.arange(len(meta)))
        meta["frame_fraction"] = (
            targets["frame_fraction"].to_numpy() if "frame_fraction" in targets else np.nan
        )
        inside = meta[flag_col].fillna(1.0) if flag_col in meta else pd.Series(1.0, index=meta.index)
        usable = meta[usable_col].fillna(0.0) if usable_col in meta else pd.Series(1.0, index=meta.index)
        meta["flag_camera_in_exclusion"] = inside > 0
        meta["flag_band_unusable"] = usable < 0.5
        meta["flag_camera_gap"] = meta["frame_fraction"].fillna(0.0) < frame_min
        surface = (
            meta["camera_surface"].fillna(0.0)
            if "camera_surface" in meta
            else pd.Series(0.0, index=meta.index)
        )
        meta["flag_camera_surface"] = surface > 0
        meta["unflagged"] = ~(
            meta["flag_camera_in_exclusion"]
            | meta["flag_band_unusable"]
            | meta["flag_camera_gap"]
            | meta["flag_camera_surface"]
        )
        arrays = {k: np.concatenate([a[k] for a in parts["arrays"]]) for k in parts["arrays"][0]}
        zpath = scratch / f"{g.name}.zarr"
        _write_arrays(
            zpath,
            arrays,
            {
                "geometry": g.name,
                "length_s": g.length_s,
                "height_m": g.height_m,
                "anchor": g.anchor,
                "time_step_s": patches.time_step_s,
                "depth_step_m": patches.depth_step_m,
                "frequencies_hz": [float(f) for f in freqs],
            },
        )
        gdir = target / g.name
        gdir.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(gdir / "meta.parquet", index=False)
        targets.insert(0, "window_id", meta["window_id"].to_numpy())
        targets.to_parquet(gdir / "targets.parquet", index=False)
        with open(gdir / "lag_offsets.json", "w", encoding="utf-8") as fh:
            json.dump(parts["lag_offsets"], fh, indent=2)
        classes = _classes_from_series(series, "_res", list(tcfg.get("exclude", [])))
        summary["geometries"][g.name] = {
            "n_windows": int(len(meta)),
            "n_unflagged": int(meta["unflagged"].sum()),
            "per_transect": meta.groupby("transect")["unflagged"]
            .agg(["size", "sum"])
            .rename(columns={"size": "windows", "sum": "unflagged"})
            .to_dict("index"),
            "shape": list(arrays["sv_db"].shape),
            "bytes": int(sum(a.nbytes for a in arrays.values())),
            "store": str(zpath),
            "targets_hard_at_lag": summarize_targets(
                targets, classes + list((tcfg.get("merged") or {}).keys()), "hard"
            ),
        }
        outputs += [gdir / "meta.parquet", gdir / "targets.parquet", gdir / "lag_offsets.json"]
        log.info(
            "%s: %d windows (%d unflagged), %.2f GB",
            g.name,
            len(meta),
            int(meta["unflagged"].sum()),
            summary["geometries"][g.name]["bytes"] / 1e9,
        )
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    outputs.append(target / "summary.json")
    write_manifest(target, "build-windows", run_cfg, inputs, outputs)
    return target


def run_make_splits(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    variant: str = "",
) -> Path:
    """Assign the windows of every geometry to the protocols and check the leakage rules."""
    scfg = cfg.get("splits", {})
    target = out_dir / run_tag(experiment, seed, variant or "")
    geometries = list(cfg.get("windows", {}).get("geometries", {}))
    inputs: dict[str, str | Path] = {
        g: target / g / "meta.parquet" for g in geometries if (target / g / "meta.parquet").is_file()
    }
    if not inputs:
        raise FileNotFoundError(f"no window metadata under {target}")
    run_cfg = {**cfg, "experiment": experiment, "seed": seed, "stage": "splits", "variant": variant or ""}
    marker = target / "splits_summary.json"
    if not force and marker.is_file() and manifest_is_current(target / "splits", run_cfg, inputs):
        log.info("splits are up to date in %s", target)
        return target
    (target / "splits").mkdir(parents=True, exist_ok=True)
    clean_excluded = str(scfg.get("clean_protocol_excludes", "20180509T0018Z"))
    summary: dict[str, Any] = {}
    outputs: list[str | Path] = []
    for g, meta_path in inputs.items():
        meta = pd.read_parquet(meta_path)
        # Every window carries its own buffer, from the lag uncertainty of its own targets.
        buffers = buffer_by_window(meta, float(scfg.get("lag_sd_multiple", 2)))
        buffer = buffer_seconds(meta, float(scfg.get("lag_sd_multiple", 2)))
        buffer_transects = buffer_by_transect(meta, float(scfg.get("lag_sd_multiple", 2)))
        pieces = []
        # Long windows lose a window length plus a buffer at the block boundary, so they get a larger block.
        long_geometry = buffer > float(scfg.get("long_buffer_s", 600))
        val_fraction = float(
            scfg.get("loto_val_fraction_long", 0.4) if long_geometry else scfg.get("loto_val_fraction", 0.3)
        )
        val_position = str(scfg.get("loto_val_position", "end"))
        loto = loto_folds(meta, val_fraction, buffers, val_position)
        loto["protocol"] = "loto_all"
        pieces.append(loto)
        unflagged = meta[meta["unflagged"]]
        loto_u = loto_folds(unflagged, val_fraction, buffers, val_position)
        loto_u["protocol"] = "loto_unflagged"
        pieces.append(loto_u)
        clean = unflagged[unflagged["transect"] != clean_excluded]
        loto_c = loto_folds(clean, val_fraction, buffers, val_position)
        loto_c["protocol"] = "clean_three_transects"
        pieces.append(loto_c)
        block_s = float(scfg.get("block_s", 1200))
        within_run = buffer < block_s / 2.0
        if within_run:
            blocks = block_splits(
                meta,
                block_s,
                list(scfg.get("block_pattern", ["train", "train", "val", "train", "train", "test"])),
                buffers,
            )
            blocks["protocol"] = "within_transect"
            pieces.append(blocks)
        else:
            # A buffer wider than half a block leaves no windows; the geometry is not split within transects.
            log.info("%s: buffer %.0f s exceeds half a block; within-transect protocol skipped", g, buffer)
        assignment = pd.concat(pieces, ignore_index=True)
        violations = pd.concat(
            [
                check_splits(meta, assignment[assignment["protocol"] == p], buffers).assign(protocol=p)
                for p in assignment["protocol"].unique()
            ],
            ignore_index=True,
        )
        if not violations.empty:
            violations.to_csv(target / "splits" / f"{g}_violations.csv", index=False)
            raise RuntimeError(
                f"leakage rule violated for geometry {g}; see {target / 'splits' / f'{g}_violations.csv'}"
            )
        assignment.to_parquet(target / "splits" / f"{g}.parquet", index=False)
        counts = split_counts(assignment, meta)
        counts.to_csv(target / "splits" / f"{g}_counts.csv", index=False)
        outputs += [target / "splits" / f"{g}.parquet", target / "splits" / f"{g}_counts.csv"]
        summary[g] = {
            "buffer_s": buffer,
            "buffer_s_by_transect": buffer_transects,
            "loto_val_fraction": val_fraction,
            "within_transect_run": within_run,
            "n_windows": int(len(meta)),
            "n_unflagged": int(meta["unflagged"].sum()),
            "protocols": {
                p: {
                    "folds": int(a["fold"].nunique()),
                    "assigned": int(a["split"].notna().sum()),
                    "dropped": int(a["split"].isna().sum()),
                }
                for p, a in assignment.groupby("protocol")
            },
        }
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    outputs.append(marker)
    write_manifest(target / "splits", "make-splits", run_cfg, inputs, outputs)
    return target
