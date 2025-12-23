"""Stage 4: camera time and depth series.

The Deep Vision metadata gives a depth per frame at about five frames per
second. The series carries dropouts (exact zeros) and isolated jumps of tens of
metres between consecutive frames, which are physically impossible for a towed
body. Depth is cleaned with a rolling median and a vertical speed bound
derived from the data, rejected values are flagged and counted, short gaps are
interpolated and long gaps left empty. Frames and detections are then
aggregated to a base time resolution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .data.transects import parse_deepvision_xml
from .manifest import write_manifest

log = logging.getLogger(__name__)


def clean_depth(
    timestamps: pd.Series,
    depth: pd.Series,
    trend_window_s: float = 30.0,
    smooth_window_s: float = 30.0,
    speed_quantile: float = 0.95,
    speed_step_s: float = 10.0,
    outlier_mads: float = 3.0,
    max_gap_s: float = 10.0,
    min_valid_in_window: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a per-frame depth series.

    Exact zeros are dropouts. A centred rolling median over ``trend_window_s``
    seconds of the remaining readings is the physical trend: long enough for
    the sensor's scatter to average out, so that its speed is a plausible
    vertical speed of the towed body. The speed bound is the
    ``speed_quantile`` of the trend's absolute speed over steps of
    ``speed_step_s`` seconds, long enough that the sensor's quantisation step
    does not dominate the speed and timestamp jitter plays no role. A
    reading is rejected when it departs from the trend by more than
    ``outlier_mads`` times the robust scale of the residuals plus the distance
    the camera can move at the speed bound within half a trend window. The
    cleaned depth is a centred rolling median over ``smooth_window_s`` seconds
    of the consistent readings, interpolated in time across gaps shorter than
    ``max_gap_s`` and left empty across longer ones. The result has
    ``depth_raw``, ``depth_trend``, ``depth_clean``, ``residual``,
    ``reading_flag`` (``consistent``, ``rejected``, ``dropout``) and
    ``depth_source`` (``median``, ``interpolated``, ``none``), with a summary.
    """
    t = pd.to_datetime(timestamps).reset_index(drop=True)
    d = pd.to_numeric(depth, errors="coerce").reset_index(drop=True).astype(float)
    dropout = d.isna() | (d <= 0)
    work = d.where(~dropout)
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    frame_dt = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.2
    window = max(3, int(round(trend_window_s / max(frame_dt, 1e-3))) | 1)
    trend = work.rolling(window, center=True, min_periods=min(min_valid_in_window, window)).median()
    step = max(1, int(round(speed_step_s / max(frame_dt, 1e-3))))
    tr = trend.to_numpy()
    dt = seconds[step:] - seconds[:-step]
    dz = np.abs(tr[step:] - tr[:-step])
    ok = np.isfinite(dz) & (dt > 0.5 * speed_step_s)
    speed = dz[ok] / dt[ok]
    speed_bound = float(np.quantile(speed, speed_quantile)) if speed.size else 0.0
    residual = work - trend
    mad = float(1.4826 * np.nanmedian(np.abs(residual.to_numpy())))
    tolerance = outlier_mads * mad + speed_bound * trend_window_s / 2
    rejected = (~dropout) & residual.abs().gt(tolerance)
    reading_flag = pd.Series("consistent", index=d.index, dtype=object)
    reading_flag[dropout] = "dropout"
    reading_flag[rejected] = "rejected"

    smooth = max(3, int(round(smooth_window_s / max(frame_dt, 1e-3))) | 1)
    consistent_values = work.where(~rejected)
    clean = consistent_values.rolling(
        smooth, center=True, min_periods=min(min_valid_in_window, smooth)
    ).median()
    source = pd.Series(np.where(clean.notna(), "median", "none"), index=d.index, dtype=object)
    valid_idx = np.flatnonzero(clean.notna().to_numpy())
    if valid_idx.size >= 2:
        interp = np.interp(seconds, seconds[valid_idx], clean.to_numpy()[valid_idx])
        missing = clean.isna().to_numpy()
        gap_len = np.zeros(len(clean))
        i, n = 0, len(clean)
        while i < n:
            if missing[i]:
                j = i
                while j < n and missing[j]:
                    j += 1
                left = seconds[i - 1] if i > 0 else seconds[i]
                right = seconds[j] if j < n else seconds[j - 1]
                gap_len[i:j] = right - left
                i = j
            else:
                i += 1
        fill = missing & (gap_len <= max_gap_s)
        fill[: valid_idx[0]] = False
        fill[valid_idx[-1] + 1 :] = False
        clean = pd.Series(np.where(fill, interp, clean.to_numpy()), index=d.index)
        source[fill] = "interpolated"
    out = pd.DataFrame(
        {
            "timestamp": t,
            "depth_raw": d,
            "depth_trend": trend,
            "depth_clean": clean,
            "residual": residual,
            "reading_flag": reading_flag,
            "depth_source": source,
        }
    )
    consistent = residual[reading_flag == "consistent"]
    cl = clean.to_numpy()
    dt_s = seconds[smooth:] - seconds[:-smooth]
    dz_s = np.abs(cl[smooth:] - cl[:-smooth])
    ok_s = np.isfinite(dz_s) & (dt_s > 0)
    clean_speed = dz_s[ok_s] / dt_s[ok_s]
    summary = {
        "n_frames": int(len(d)),
        "frame_interval_s": frame_dt,
        "trend_window_frames": int(window),
        "smooth_window_frames": int(smooth),
        "speed_bound_m_per_s": speed_bound,
        "residual_scale_m": mad,
        "tolerance_m": float(tolerance),
        "reading_flags": {str(k): int(v) for k, v in reading_flag.value_counts().items()},
        "depth_sources": {str(k): int(v) for k, v in source.value_counts().items()},
        "fraction_rejected": float(rejected.mean()),
        "fraction_with_depth": float(clean.notna().mean()),
        "raw_scatter_iqr_m": float(consistent.quantile(0.75) - consistent.quantile(0.25))
        if len(consistent)
        else None,
        "clean_speed_p95_m_per_s": float(np.quantile(clean_speed, 0.95)) if clean_speed.size else None,
        "depth_clean_quantiles_m": {
            f"q{int(q * 100):02d}": float(v) for q, v in clean.quantile([0.05, 0.5, 0.95]).items()
        },
    }
    return out, summary


def aggregate_frames(
    frames: pd.DataFrame,
    detections: pd.DataFrame | None,
    classes: list[str],
    interval_s: float,
    count_columns: list[str],
    extra_weighted: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Aggregate per-frame depth and detection counts to fixed intervals.

    ``frames`` needs ``timestamp``, ``depth_clean`` and the per-frame count
    columns; ``detections`` (optional) needs ``timestamp``, ``class_name`` and
    ``score_calibrated`` and yields confidence-weighted counts per class;
    ``extra_weighted`` maps a column suffix to a further detection table
    summed the same way. Each row of the result is one interval starting at
    ``time``.
    """
    f = frames.copy()
    f["time"] = pd.to_datetime(f["timestamp"]).dt.floor(f"{int(interval_s * 1000)}ms")
    agg: dict[str, Any] = {
        "n_frames": ("timestamp", "size"),
        "n_frames_with_depth": ("depth_clean", "count"),
        "depth_mean": ("depth_clean", "mean"),
        "depth_median": ("depth_clean", "median"),
        "depth_min": ("depth_clean", "min"),
        "depth_max": ("depth_clean", "max"),
    }
    if "depth_raw" in f.columns:
        # Raw readings without the dropouts, kept so that the alignment can be run on them as well.
        raw = pd.to_numeric(f["depth_raw"], errors="coerce")
        f["depth_raw_nonzero"] = raw.where(raw > 0)
        agg["depth_raw_median"] = ("depth_raw_nonzero", "median")
        agg["depth_raw_mean"] = ("depth_raw_nonzero", "mean")
    for col in count_columns:
        if col in f.columns:
            agg[col] = (col, "sum")
    series = f.groupby("time").agg(**agg)
    tables = {"_weighted": detections, **(extra_weighted or {})}
    for suffix, table in tables.items():
        if table is None or table.empty:
            continue
        dets = table.copy()
        dets["time"] = pd.to_datetime(dets["timestamp"]).dt.floor(f"{int(interval_s * 1000)}ms")
        weighted = dets.groupby(["time", "class_name"])["score_calibrated"].sum().unstack(fill_value=0.0)
        weighted = weighted.reindex(columns=classes, fill_value=0.0)
        weighted.columns = [f"{c}{suffix}" for c in weighted.columns]
        series = series.join(weighted, how="left").fillna({c: 0.0 for c in weighted.columns})
    full = pd.date_range(series.index.min(), series.index.max(), freq=f"{int(interval_s * 1000)}ms")
    series = series.reindex(full)
    series.index.name = "time"
    series["n_frames"] = series["n_frames"].fillna(0).astype(int)
    if "static_frame" in series.columns:
        # Share of the interval's frames that show fixed structure, which marks time at the surface.
        series["frames_static_fraction"] = series["static_frame"] / series["n_frames"].where(
            series["n_frames"] > 0
        )
    return series.reset_index()


def run_build_camera_series(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    transect: str = "",
    camera: str = "Left",
) -> Path:
    """Clean the depth series of one transect, join it to inference tables and aggregate to intervals."""
    ccfg = cfg.get("camera_series", {})
    interval_s = float(ccfg.get("interval_s", 10.0))
    project = resolve_path(cfg, "project_dir")
    transect_dir = resolve_path(cfg, "source_data", transect)
    infer_dir = Path(project, "runs", "inference", f"{experiment}_seed{seed}", transect, camera)
    target = Path(project, "runs", "camera_series", f"{experiment}_seed{seed}", transect, camera)
    target.mkdir(parents=True, exist_ok=True)

    meta, _ = parse_deepvision_xml(transect_dir / "deepvision.xml")
    meta = meta.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    cleaned, depth_summary = clean_depth(
        meta["timestamp"],
        meta["depth_num"],
        trend_window_s=float(ccfg.get("trend_window_s", 30.0)),
        smooth_window_s=float(ccfg.get("smooth_window_s", 30.0)),
        speed_quantile=float(ccfg.get("speed_quantile", 0.95)),
        outlier_mads=float(ccfg.get("outlier_mads", 3.0)),
        max_gap_s=float(ccfg.get("max_gap_s", 10.0)),
    )
    cleaned.to_parquet(target / "depth_per_frame.parquet", index=False)

    frames = pd.read_parquet(infer_dir / "frames.parquet")
    frames["timestamp"] = pd.to_datetime(frames["timestamp"])
    joined = pd.merge_asof(
        frames.sort_values("timestamp"),
        cleaned[["timestamp", "depth_raw", "depth_clean", "reading_flag", "depth_source"]].sort_values(
            "timestamp"
        ),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=int(ccfg.get("join_tolerance_ms", 50))),
    )
    detections = pd.read_parquet(infer_dir / "detections.parquet")
    for col, default in (("static", False), ("kept_resolved", True), ("rank", 1)):
        if col not in detections.columns:
            detections[col] = default
    classes = sorted(detections["class_name"].dropna().unique().tolist())
    keep = ["timestamp", "class_name", "score_calibrated"]
    # Confidence-weighted counts sum the calibrated scores of the resolved objects; the sum over
    # every stored first hypothesis above the floor is kept as a diagnostic.
    objects = detections.loc[detections["kept_resolved"] & ~detections["static"], keep]
    first = detections.loc[(detections["rank"] == 1) & ~detections["static"], keep]
    if "n_static" in joined.columns:
        joined["static_frame"] = (joined["n_static"] > 0).astype(int)
    count_columns = [
        c
        for c in joined.columns
        if "_ge" in c
        or c.endswith(("_op", "_res", "_exp", "_hyp"))
        or c.startswith("n_")
        or c == "static_frame"
    ]
    series = aggregate_frames(joined, objects, classes, interval_s, count_columns, {"_weighted_floor": first})
    series.insert(0, "camera", camera)
    series.insert(0, "transect", transect)
    series.to_parquet(target / f"series_{int(interval_s)}s.parquet", index=False)

    summary = {
        "experiment": experiment,
        "seed": seed,
        "transect": transect,
        "camera": camera,
        "interval_s": interval_s,
        "depth": depth_summary,
        "n_frames_joined": int(joined["depth_source"].notna().sum()),
        "n_frames_unjoined": int(joined["depth_source"].isna().sum()),
        "n_intervals": int(len(series)),
        "n_intervals_without_frames": int((series["n_frames"] == 0).sum()),
        "n_intervals_without_depth": int(series["depth_mean"].isna().sum()),
    }
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    write_manifest(
        target,
        "build-camera-series",
        {**cfg, "experiment": experiment, "seed": seed, "transect": transect, "camera": camera},
        {"metadata": transect_dir / "deepvision.xml", "frames": infer_dir / "frames.parquet"},
        [
            target / "depth_per_frame.parquet",
            target / f"series_{int(interval_s)}s.parquet",
            target / "summary.json",
        ],
    )
    return target
