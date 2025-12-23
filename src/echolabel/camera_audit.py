"""Stage 4 quality control: raw against cleaned camera depth per transect.

The behaviour of the depth cleaner is examined per transect: how
far the raw readings depart from the cleaned series, whether departures are
isolated updates or persistent offsets, whether they lean to one side, and
whether the cleaner rejects readings on both sides of the trend.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .camera import clean_depth
from .config import resolve_path
from .data.transects import parse_deepvision_xml
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def update_sequence(raw: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse consecutive identical non-zero readings into sensor updates: values and their first times."""
    nz = raw > 0
    r = raw[nz]
    t = timestamps[nz]
    if r.size == 0:
        return r, t
    change = np.r_[True, r[1:] != r[:-1]]
    return r[change], t[change]


def _runs(flags: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True values."""
    lengths = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            lengths.append(j - i)
            i = j
        else:
            i += 1
    return np.asarray(lengths)


def classify_long_runs(
    raw: np.ndarray,
    trend: np.ndarray,
    speed: np.ndarray,
    far: np.ndarray,
    min_frames: int = 10,
    ramp_speed_m_s: float = 0.15,
) -> dict[str, Any]:
    """Sort the long runs of far readings into stale holds, ramp lag and noise bursts.

    A run is a stale hold when the sensor repeats one value throughout it; it
    is ramp lag when the trend moves faster than ``ramp_speed_m_s`` and the
    readings sit on the side the trend would lag on (deeper during a descent,
    shallower during an ascent); the rest are noise bursts.
    """
    out = {
        "min_frames": min_frames,
        "n_runs": 0,
        "frames": 0,
        "stale_hold": 0,
        "ramp_lag": 0,
        "noise_burst": 0,
    }
    i, n = 0, len(far)
    while i < n:
        if not far[i]:
            i += 1
            continue
        j = i
        while j < n and far[j]:
            j += 1
        if j - i >= min_frames:
            seg_raw = raw[i:j]
            seg_res = seg_raw - trend[i:j]
            seg_speed = np.nanmedian(speed[i:j]) if np.isfinite(speed[i:j]).any() else np.nan
            out["n_runs"] += 1
            out["frames"] += j - i
            if np.unique(seg_raw).size == 1:
                out["stale_hold"] += j - i
            elif (
                np.isfinite(seg_speed)
                and abs(seg_speed) > ramp_speed_m_s
                and np.sign(np.nanmedian(seg_res)) == np.sign(seg_speed)
            ):
                out["ramp_lag"] += j - i
            else:
                out["noise_burst"] += j - i
        i = j
    frames = max(out["frames"], 1)
    out["fraction_stale_hold"] = out["stale_hold"] / frames
    out["fraction_ramp_lag"] = out["ramp_lag"] / frames
    out["fraction_noise_burst"] = out["noise_burst"] / frames
    return out


def residual_by_speed(
    residual: np.ndarray, speed: np.ndarray, consistent: np.ndarray
) -> list[dict[str, Any]]:
    """Median residual of the consistent readings per bin of signed trend speed (descending positive)."""
    edges = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf]
    labels = ["ascending fast", "ascending", "level", "descending", "descending fast"]
    rows = []
    ok = consistent & np.isfinite(residual) & np.isfinite(speed)
    for lo, hi, name in zip(edges[:-1], edges[1:], labels, strict=True):
        sel = ok & (speed > lo) & (speed <= hi)
        rows.append(
            {
                "speed_bin": name,
                "n": int(sel.sum()),
                "median_residual_m": round(float(np.median(residual[sel])), 2) if sel.any() else None,
                "q25_m": round(float(np.quantile(residual[sel], 0.25)), 2) if sel.any() else None,
                "q75_m": round(float(np.quantile(residual[sel], 0.75)), 2) if sel.any() else None,
            }
        )
    return rows


def trend_speed(trend: np.ndarray, seconds: np.ndarray, step_s: float = 10.0) -> np.ndarray:
    """Signed vertical speed of the trend over centred steps of ``step_s`` seconds, positive downward."""
    frame_dt = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.2
    k = max(1, int(round(step_s / max(frame_dt, 1e-3) / 2)))
    out = np.full(len(trend), np.nan)
    if len(trend) > 2 * k:
        dz = trend[2 * k :] - trend[: -2 * k]
        dt = seconds[2 * k :] - seconds[: -2 * k]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k:-k] = np.where(dt > 0, dz / dt, np.nan)
    return out


def window_sensitivity(
    raw: np.ndarray,
    seconds: np.ndarray,
    windows_s: tuple[float, ...] = (60.0, 30.0, 20.0),
    rejected: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Median residual by speed bin for several trend windows, on readings the cleaner did not reject."""
    frame_dt = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.2
    work = pd.Series(np.where(raw > 0, raw, np.nan))
    consistent = (raw > 0) if rejected is None else ((raw > 0) & ~rejected)
    rows = []
    for w_s in windows_s:
        w = max(3, int(round(w_s / max(frame_dt, 1e-3))) | 1)
        tr = work.rolling(w, center=True, min_periods=min(10, w)).median().to_numpy()
        sp = trend_speed(tr, seconds)
        for row in residual_by_speed(raw - tr, sp, consistent):
            rows.append({"trend_window_s": w_s, **row})
    return rows


def cleaning_diagnostics(
    cleaned: pd.DataFrame,
    timestamps: pd.Series,
    far_threshold_m: float = 5.0,
    spike_m: float = 5.0,
    depth_bins: tuple[float, ...] = (0, 20, 50, 100, 200, 300, 500),
) -> dict[str, Any]:
    """Statistics of raw against cleaned depth; see the module description."""
    raw = cleaned["depth_raw"].astype(float).to_numpy()
    clean = cleaned["depth_clean"].astype(float).to_numpy()
    trend = cleaned["depth_trend"].astype(float).to_numpy()
    flag = cleaned["reading_flag"].to_numpy()
    t = pd.to_datetime(timestamps).to_numpy()
    nz = raw > 0
    both = nz & np.isfinite(clean)
    diff = raw[both] - clean[both]
    residual = raw - trend
    out: dict[str, Any] = {
        "n_frames": int(len(raw)),
        "dropout_fraction": float((~nz).mean()),
        "raw_nonzero_quantiles_m": {
            str(q): round(float(v), 2)
            for q, v in zip(QUANTILES, np.quantile(raw[nz], QUANTILES), strict=True)
        },
        "cleaned_quantiles_m": {
            str(q): round(float(v), 2)
            for q, v in zip(QUANTILES, np.nanquantile(clean, QUANTILES), strict=True)
        },
        "raw_minus_cleaned": {
            "quantiles_m": {
                str(q): round(float(v), 2)
                for q, v in zip(QUANTILES, np.quantile(diff, QUANTILES), strict=True)
            },
            "fraction_abs_gt_2m": float((np.abs(diff) > 2).mean()),
            "fraction_abs_gt_10m": float((np.abs(diff) > 10).mean()),
            "fraction_deeper_gt_2m": float((diff > 2).mean()),
            "fraction_shallower_gt_2m": float((diff < -2).mean()),
            "fraction_deeper_gt_10m": float((diff > 10).mean()),
            "fraction_shallower_gt_10m": float((diff < -10).mean()),
        },
        "rejected": {
            "n": int((flag == "rejected").sum()),
            "deeper_than_trend": int(((flag == "rejected") & (residual > 0)).sum()),
            "shallower_than_trend": int(((flag == "rejected") & (residual < 0)).sum()),
        },
    }
    values, times = update_sequence(raw, t)
    if values.size > 3:
        du = np.diff(values)
        mid, prev, nxt = values[1:-1], values[:-2], values[2:]
        agree = np.abs(prev - nxt) < 3
        out["sensor_updates"] = {
            "n": int(values.size),
            "median_interval_s": float(
                np.median(np.diff(times).astype("timedelta64[ms]").astype(float)) / 1000
            ),
            "abs_step_quantiles_m": {
                str(q): round(float(v), 2)
                for q, v in zip(
                    (0.5, 0.75, 0.9, 0.99), np.quantile(np.abs(du), (0.5, 0.75, 0.9, 0.99)), strict=True
                )
            },
            "corr_consecutive_steps": float(np.corrcoef(du[:-1], du[1:])[0, 1]) if du.size > 2 else None,
            "single_update_spikes_deeper": int(
                ((mid - prev > spike_m) & (mid - nxt > spike_m) & agree).sum()
            ),
            "single_update_spikes_shallower": int(
                ((prev - mid > spike_m) & (nxt - mid > spike_m) & agree).sum()
            ),
        }
    rows = []
    sel = nz & np.isfinite(trend)
    cut = pd.cut(trend[sel], list(depth_bins))
    for b, g in pd.Series(residual[sel]).groupby(cut, observed=True):
        rows.append(
            {
                "trend_depth_bin": str(b),
                "n": int(len(g)),
                "q05": round(float(g.quantile(0.05)), 2),
                "q50": round(float(g.quantile(0.5)), 2),
                "q95": round(float(g.quantile(0.95)), 2),
                "fraction_deeper_gt_5m": float((g > far_threshold_m).mean()),
                "fraction_shallower_gt_5m": float((g < -far_threshold_m).mean()),
            }
        )
    out["residual_by_trend_depth"] = rows
    pos = residual[sel & (residual > far_threshold_m)]
    neg = residual[sel & (residual < -far_threshold_m)]
    out["excursions"] = {
        "deeper_median_m": float(np.median(pos)) if pos.size else None,
        "deeper_median_ratio_to_depth": float(np.median(pos / trend[sel & (residual > far_threshold_m)]))
        if pos.size
        else None,
        "shallower_median_m": float(np.median(-neg)) if neg.size else None,
    }
    far = (np.abs(residual) > far_threshold_m) & nz
    runs = _runs(far)
    seconds = (pd.to_datetime(timestamps) - pd.to_datetime(timestamps).iloc[0]).dt.total_seconds().to_numpy()
    speed = trend_speed(trend, seconds)
    out["long_far_runs"] = classify_long_runs(raw, trend, speed, far)
    out["residual_by_trend_speed"] = residual_by_speed(residual, speed, flag == "consistent")
    out["trend_window_sensitivity"] = window_sensitivity(raw, seconds, rejected=(flag == "rejected"))
    out["far_runs"] = {
        "threshold_m": far_threshold_m,
        "n_runs": int(runs.size),
        "frames_in_runs": int(runs.sum()) if runs.size else 0,
        "run_length_frames_q50": float(np.quantile(runs, 0.5)) if runs.size else None,
        "run_length_frames_q90": float(np.quantile(runs, 0.9)) if runs.size else None,
        "fraction_of_far_frames_in_runs_ge_10": float(runs[runs >= 10].sum() / runs.sum())
        if runs.size
        else None,
    }
    fr = pd.Series(np.where(nz, raw, np.nan))
    frame_dt = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.2
    w = max(5, int(round(30.0 / max(frame_dt, 1e-3))) | 1)
    gap = fr.rolling(w, center=True, min_periods=10).quantile(0.8) - fr.rolling(
        w, center=True, min_periods=10
    ).quantile(0.2)
    two_sided = fr.rolling(w, center=True, min_periods=10).apply(
        lambda s: ((s - np.nanmedian(s) > 5).mean() > 0.2) and ((np.nanmedian(s) - s > 5).mean() > 0.2),
        raw=True,
    )
    out["scatter_30s"] = {
        "p80_minus_p20_quantiles_m": {
            str(q): round(float(v), 2)
            for q, v in zip((0.25, 0.5, 0.75, 0.9), gap.quantile([0.25, 0.5, 0.75, 0.9]), strict=True)
        },
        "fraction_windows_two_sided": float(np.nanmean(two_sided.to_numpy()))
        if two_sided.notna().any()
        else None,
    }
    return out


def choose_excerpt(
    cleaned: pd.DataFrame, timestamps: pd.Series, minutes: float = 20.0
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end of the window with the most frames more than 10 m off the cleaned depth."""
    t = pd.to_datetime(timestamps)
    far = ((cleaned["depth_raw"] - cleaned["depth_clean"]).abs() > 10) & (cleaned["depth_raw"] > 0)
    s = pd.Series(far.to_numpy(dtype=float), index=t)
    counts = s.rolling(f"{int(minutes * 60)}s").sum()
    if counts.empty:
        return t.iloc[0], t.iloc[-1]
    end = counts.idxmax()
    start = end - pd.Timedelta(minutes=minutes)
    return max(start, t.iloc[0]), end


def depth_figure(
    cleaned: pd.DataFrame,
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.to_datetime(timestamps)
    raw = cleaned["depth_raw"].astype(float).where(cleaned["depth_raw"] > 0)
    clean = cleaned["depth_clean"]
    sel = (t >= start) & (t < end)
    detail_end = start + pd.Timedelta(minutes=2)
    sel2 = (t >= start) & (t < detail_end)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(t[sel], raw[sel], ".", ms=2, color="0.6", label="raw")
    ax.plot(t[sel], cleaned["depth_trend"][sel], color="tab:blue", lw=0.8, label="trend")
    ax.plot(t[sel], clean[sel], color="tab:red", lw=1.2, label="cleaned")
    ax.invert_yaxis()
    ax.set_ylabel("depth (m)")
    ax.set_title(f"{title}: 20 minute excerpt")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(t[sel2], raw[sel2], ".-", ms=3, lw=0.5, color="0.4", label="raw, every frame")
    ax.plot(t[sel2], clean[sel2], color="tab:red", lw=1.2, label="cleaned")
    ax.invert_yaxis()
    ax.set_title("first two minutes of the excerpt")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    diff = (raw - clean).dropna()
    ax.hist(diff.clip(-40, 40), bins=160, color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("raw minus cleaned depth (m)")
    ax.set_ylabel("frames")
    ax.set_title("raw minus cleaned")
    ax = axes[1, 1]
    ax.hist(raw.dropna().clip(0, 400), bins=200, color="0.4", label="raw")
    ax.hist(clean.dropna().clip(0, 400), bins=200, color="tab:red", alpha=0.5, label="cleaned")
    ax.set_yscale("log")
    ax.set_xlabel("depth (m)")
    ax.set_title("depth histograms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_audit_camera_depth(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Diagnose the depth cleaning on every inference transect and write figures and a summary."""
    ccfg = cfg.get("camera_series", {})
    transects = list(cfg.get("inference", {}).get("transects", []))
    inputs: dict[str, str | Path] = {
        t: resolve_path(cfg, "source_data", t, "deepvision.xml") for t in transects
    }
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("camera depth audit is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(inspect.signature(clean_depth).parameters) - {"timestamps", "depth"}
    kwargs = {k: v for k, v in ccfg.items() if k in allowed}
    acfg = ccfg.get("audit", {})
    summaries: dict[str, Any] = {}
    outputs: list[str | Path] = []
    for transect in transects:
        meta, _ = parse_deepvision_xml(inputs[transect])
        meta = meta.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        cleaned, stats = clean_depth(meta["timestamp"], meta["depth_num"], **kwargs)
        diag = cleaning_diagnostics(cleaned, meta["timestamp"], float(acfg.get("far_threshold_m", 5.0)))
        diag["cleaning"] = {
            k: v for k, v in stats.items() if k in ("speed_bound_m_s", "tolerance_m", "reading_flags")
        }
        start, end = choose_excerpt(cleaned, meta["timestamp"], float(acfg.get("excerpt_minutes", 20)))
        diag["excerpt"] = [str(start), str(end)]
        fig_path = out_dir / f"{transect}_depth.png"
        depth_figure(cleaned, meta["timestamp"], start, end, transect, fig_path)
        outputs.append(fig_path)
        summaries[transect] = diag
        log.info(
            "%s: %.0f percent of frames differ by more than 2 m",
            transect,
            100 * diag["raw_minus_cleaned"]["fraction_abs_gt_2m"],
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "audit-camera-depth", cfg, inputs, outputs)
    return out_dir
