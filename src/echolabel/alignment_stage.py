"""Stage 6 command: lag between the camera and the echosounder for one transect.

The command reads the preprocessed transect store and the camera interval
series, derives the physical prior from vessel speed, correlates camera
signals with the backscatter at the camera depth over a lag grid, bootstraps
the lag over time blocks, tests a piecewise lag against a constant one on
held-out segments, runs the plateau and event tests that look for a lag in
the along-track variation alone, and writes a mapping per interval whose
lag distribution follows the configured lag model.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .acoustics.preprocess import HARD_BITS, MASK_SURFACE
from .acoustics.store import frequency_position
from .alignment.lag import (
    band_series,
    band_usable_fraction,
    best_lag,
    block_bootstrap_lag,
    echogram_at_intervals,
    event_lags,
    find_plateaus,
    held_out_lag_check,
    lag_bounds,
    lag_bounds_per_interval,
    lag_correlation,
    lag_mapping,
    paired_series,
    piecewise_lags,
    plateau_lag_curve,
    setback_from_warp,
    vessel_speed,
)
from .config import resolve_path
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def build_signals(
    series: pd.DataFrame, spec: dict[str, list[str]], exclude: list[str]
) -> dict[str, np.ndarray]:
    """Camera signals as sums of the series columns matching each pattern list; empty matches are skipped."""
    out: dict[str, np.ndarray] = {}
    for name, patterns in spec.items():
        cols: list[str] = []
        for pat in patterns:
            cols += [
                c for c in series.columns if fnmatch.fnmatch(c, pat) and c not in exclude and c not in cols
            ]
        if not cols:
            log.warning("signal %s matches no column", name)
            continue
        out[name] = series[cols].sum(axis=1, min_count=1).to_numpy(dtype=float)
    return out


def exclusion_depth(ds: xr.Dataset, position: int, draft_m: float) -> float:
    """Depth of the surface exclusion of one channel, from the stored value or from the surface bits."""
    if "surface_exclusion_m" in ds:
        return float(ds["surface_exclusion_m"].isel(frequency=position).values) + draft_m
    surf = (ds["mask"].isel(frequency=position).values & MASK_SURFACE) > 0
    rng = ds["range"].values
    everywhere = surf.all(axis=0)
    if not everywhere.any():
        return draft_m
    return float(rng[everywhere].max() + np.median(np.diff(rng))) + draft_m


def speed_at(speed: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Speed interpolated onto the given times."""
    s = speed.dropna(subset=["speed_m_s"])
    if s.empty:
        return np.full(len(times), np.nan)
    x = s["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    return np.interp(times.astype("datetime64[ns]").astype(np.int64), x, s["speed_m_s"].to_numpy(dtype=float))


def _figures(
    curves: pd.DataFrame,
    boots: dict[tuple[str, int], dict[str, Any]],
    primary: tuple[str, int],
    interval_s: float,
    overlay: dict[str, np.ndarray],
    out_dir: Path,
    title: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if "mode" in curves.columns:
        nearest = curves[curves["mode"] == "nearest"]
        curves = curves[curves["mode"] == "strict"]
    else:
        nearest = curves.iloc[0:0]
    signals = sorted(curves["signal"].unique())
    fig, axes = plt.subplots(len(signals), 1, figsize=(9, 2.8 * len(signals)), sharex=True, squeeze=False)
    for ax, sig in zip(axes[:, 0], signals, strict=True):
        for f, g in curves[curves["signal"] == sig].groupby("frequency_hz"):
            ax.plot(g["lag_s"] / 60, g["corr"], lw=1.2, label=f"{int(f / 1000)} kHz")
        key = (sig, primary[1])
        alt = nearest[(nearest["signal"] == sig) & (nearest["frequency_hz"] == primary[1])]
        if not alt.empty:
            ax.plot(
                alt["lag_s"] / 60, alt["corr"], lw=1, ls="--", color="k", label="camera inside exclusion kept"
            )
        if key in boots:
            b = boots[key]
            lo, hi = np.nanpercentile(b["curves"], [2.5, 97.5], axis=0)
            g = curves[(curves["signal"] == sig) & (curves["frequency_hz"] == primary[1])]
            ax.fill_between(g["lag_s"] / 60, lo, hi, color="0.8", label="bootstrap 95% band")
            ax.axvline(b["lag_median"] * interval_s / 60, color="k", lw=0.8, ls="--")
        ax.set_ylabel("rank correlation")
        ax.set_title(f"{title}: {sig}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("lag (minutes, camera after vessel)")
    fig.tight_layout()
    p = out_dir / "lag_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    if overlay:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        t = overlay["time"]
        for name, color in (("camera", "tab:blue"), ("acoustic", "tab:orange")):
            x = overlay[name]
            z = (x - np.nanmean(x)) / np.nanstd(x)
            ax.plot(t, z, color=color, lw=1, label=overlay[f"{name}_label"])
        ax.set_ylabel("standardised value")
        ax.set_xlabel("camera time (UTC)")
        ax.set_title(f"{title}: series at the estimated lag")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / "overlay.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def run_align_transect(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    transect: str = "",
    camera: str = "Left",
) -> Path:
    """Estimate the camera lag of one transect with its uncertainty and write the mapping."""
    acfg = cfg.get("alignment", {})
    interval_s = float(cfg.get("interval_s", 10.0))
    project = resolve_path(cfg, "project_dir")
    store = resolve_path(cfg, "scratch_dir", "acoustics", "transects", f"{transect}.zarr")
    series_path = Path(
        project,
        "runs",
        "camera_series",
        f"{experiment}_seed{seed}",
        transect,
        camera,
        f"series_{int(interval_s)}s.parquet",
    )
    target = out_dir / f"{experiment}_seed{seed}" / transect / camera
    inputs: dict[str, str | Path] = {"store": store, "series": series_path}
    run_cfg = {**cfg, "experiment": experiment, "seed": seed, "transect": transect, "camera": camera}
    if not force and manifest_is_current(target, run_cfg, inputs):
        log.info("alignment of %s is up to date in %s", transect, target)
        return target
    if not series_path.is_file():
        raise FileNotFoundError(f"camera series not found: {series_path}")
    target.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(str(store))
    series = pd.read_parquet(series_path).sort_values("time").reset_index(drop=True)
    interval_start = series["time"].to_numpy().astype("datetime64[ns]")
    depth_columns = dict(acfg.get("depth_columns", {"cleaned": "depth_median"}))
    primary_depth = str(acfg.get("primary_depth", next(iter(depth_columns))))
    camera_depth = series[depth_columns[primary_depth]].to_numpy(dtype=float)
    ping_time = ds["ping_time"].values
    depth = ds["depth"].values
    draft_m = float(ds.attrs.get("draft_m", 8.5))

    # Physical prior from vessel speed within the camera span.
    speed = vessel_speed(
        ping_time, ds["latitude"].values, ds["longitude"].values, float(acfg.get("speed_window_s", 60))
    )
    span = (speed["time"] >= interval_start[0]) & (speed["time"] <= interval_start[-1])
    warp = acfg.get("warp_length_m", [500, 1500])
    depth_median = float(np.nanmedian(camera_depth))
    setback = (
        float(setback_from_warp(warp[0], depth_median)),
        float(setback_from_warp(warp[1], depth_median)),
    )
    prior = lag_bounds(speed.loc[span, "speed_m_s"].to_numpy(), setback)
    prior.update({"warp_length_m": [float(w) for w in warp], "camera_depth_m_median": depth_median})
    speed.loc[span].to_csv(target / "speed.csv", index=False)
    per_interval = lag_bounds_per_interval(
        speed_at(speed, interval_start),
        camera_depth,
        (float(warp[0]), float(warp[1])),
        float(acfg.get("lag_max_s", 900)),
    )
    prior["per_interval"] = {
        "lag_prior_lo_s_median": float(np.nanmedian(per_interval["lag_prior_lo_s"])),
        "lag_prior_hi_s_median": float(np.nanmedian(per_interval["lag_prior_hi_s"])),
        "lag_prior_hi_s_q05_q95": [
            float(np.nanquantile(per_interval["lag_prior_hi_s"], 0.05)),
            float(np.nanquantile(per_interval["lag_prior_hi_s"], 0.95)),
        ],
    }

    lags = np.arange(
        int(round(float(acfg.get("lag_min_s", 0)) / interval_s)),
        int(round(float(acfg.get("lag_max_s", 900)) / interval_s)) + 1,
    )
    # The echogram starts ``offset`` intervals before the camera series, so that every camera
    # interval has an acoustic partner at every lag of the grid.
    offset = int(lags[-1])
    step = np.timedelta64(int(round(interval_s * 1000)), "ms")
    interval_ext = np.concatenate(
        [interval_start[0] - step * np.arange(offset, 0, -1), interval_start]
    ).astype("datetime64[ns]")
    signals = build_signals(
        series, acfg.get("signals", {"all_weighted": ["*_weighted"]}), acfg.get("exclude_columns", [])
    )
    frequencies = [float(f) for f in acfg.get("frequencies_hz", [38000])]
    primary = (
        str(acfg.get("primary_signal", next(iter(signals)))),
        int(acfg.get("primary_frequency_hz", 38000)),
    )
    half_width = float(acfg.get("band_half_width_m", 5))
    smooth = int(acfg.get("smooth_intervals", 3))
    method = str(acfg.get("method", "spearman"))
    mask_all = ds["mask"]
    min_usable = float(acfg.get("min_band_usable_fraction", 0.5))
    pcfg = dict(acfg.get("plateau", {}))
    ecfg = dict(acfg.get("events", {}))
    plateau_rows: list[dict[str, Any]] = []
    plateau_curve_rows: list[dict[str, Any]] = []
    raw_bands: dict[int, np.ndarray] = {}
    plateaus = find_plateaus(
        camera_depth,
        int(round(float(pcfg.get("min_minutes", 10)) * 60.0 / interval_s)),
        float(pcfg.get("span_m", 10.0)),
    )
    plateau_span = float(pcfg.get("span_m", 10.0))
    primary_excl = exclusion_depth(ds, frequency_position(ds, float(primary[1])), draft_m)
    surface_frac = (
        series["frames_static_fraction"].fillna(0.0).to_numpy(dtype=float)
        if "frames_static_fraction" in series.columns
        else np.zeros(len(series))
    )

    def usable_plateaus(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # Plateaus must sit below the primary surface exclusion by a band half width and not at the surface.
        return [
            (s, e)
            for s, e in runs
            if np.nanmedian(camera_depth[s:e]) >= primary_excl + half_width and surface_frac[s:e].mean() < 0.1
        ]

    plateaus = usable_plateaus(plateaus)
    if not plateaus and pcfg.get("span_m_fallback"):
        plateau_span = float(pcfg["span_m_fallback"])
        plateaus = usable_plateaus(
            find_plateaus(
                camera_depth, int(round(float(pcfg.get("min_minutes", 10)) * 60.0 / interval_s)), plateau_span
            )
        )
    curve_rows, boot_rows, seg_rows, held_rows = [], [], [], []
    flags: dict[str, np.ndarray] = {}
    exclusion_stats: dict[str, Any] = {}
    primary_channel: tuple[np.ndarray, np.ndarray, float] | None = None
    boots: dict[tuple[str, int], dict[str, Any]] = {}
    overlay: dict[str, Any] = {}
    primary_result: dict[str, Any] = {}
    for f in frequencies:
        i = frequency_position(ds, f)
        db = ds["sv_db"].isel(frequency=i).values.astype(np.float32)
        hard = (mask_all.isel(frequency=i).values & HARD_BITS) > 0
        echogram, centres = echogram_at_intervals(ping_time, db, hard, depth, interval_ext, interval_s, 1.0)
        excl_depth = exclusion_depth(ds, i, draft_m)
        if int(f) == primary[1]:
            primary_channel = (echogram, centres, excl_depth)
        inside = np.isfinite(camera_depth) & (camera_depth < excl_depth)
        key = str(int(f))
        flags[f"camera_in_exclusion_{key}"] = inside
        exclusion_stats[key] = {
            "exclusion_depth_m": excl_depth,
            "intervals_camera_inside": int(inside.sum()),
            "intervals_with_depth": int(np.isfinite(camera_depth).sum()),
        }
        with np.errstate(divide="ignore"):
            raw_bands[int(f)] = np.vstack(
                [
                    10.0
                    * np.log10(
                        np.where(
                            (
                                b := band_series(
                                    echogram,
                                    centres,
                                    camera_depth,
                                    half_width,
                                    int(k),
                                    excl_depth,
                                    min_usable,
                                    offset,
                                )
                            )
                            > 0,
                            b,
                            np.nan,
                        )
                    )
                    for k in lags
                ]
            )
        for name, signal in signals.items():
            for p_idx, (p_start, p_end) in enumerate(plateaus):
                prior_p = (
                    float(np.nanmedian(per_interval["lag_prior_lo_s"][p_start:p_end])),
                    float(np.nanmedian(per_interval["lag_prior_hi_s"][p_start:p_end])),
                )
                if np.nansum(signal[p_start:p_end]) <= 0:
                    continue
                res = plateau_lag_curve(
                    signal,
                    raw_bands[int(f)],
                    lags,
                    p_start,
                    p_end,
                    int(round(float(pcfg.get("detrend_s", 180)) / interval_s)),
                    method,
                    int(pcfg.get("block_intervals", 12)),
                    int(pcfg.get("n_boot", 100)),
                    int(cfg.get("seed", 0)),
                    prior_p,
                    interval_s,
                    int(pcfg.get("n_null", 0)),
                )
                curve_p = res.pop("curve")
                plateau_rows.append(
                    {
                        "plateau": p_idx,
                        "signal": name,
                        "frequency_hz": int(f),
                        "depth_m": float(np.nanmedian(camera_depth[p_start:p_end])),
                        "prior_lo_s": prior_p[0],
                        "prior_hi_s": prior_p[1],
                        **res,
                    }
                )
                for k, c in zip(lags, curve_p, strict=True):
                    plateau_curve_rows.append(
                        {
                            "plateau": p_idx,
                            "signal": name,
                            "frequency_hz": int(f),
                            "lag_s": float(k * interval_s),
                            "corr": float(c),
                        }
                    )
            cam, acoustic = paired_series(
                signal,
                echogram,
                centres,
                camera_depth,
                half_width,
                lags,
                smooth,
                True,
                excl_depth,
                min_usable,
                offset,
            )
            curve = lag_correlation(cam, acoustic, lags, method)
            curve["signal"], curve["frequency_hz"] = name, int(f)
            curve["lag_s"], curve["mode"] = curve["lag_intervals"] * interval_s, "strict"
            curve_rows.append(curve)
            if (name, int(f)) == primary:
                # Sensitivity: keep intervals with the camera inside the exclusion; use the cells below.
                cam_n, acoustic_n = paired_series(
                    signal, echogram, centres, camera_depth, half_width, lags, smooth, True, None, 0.0, offset
                )
                curve_n = lag_correlation(cam_n, acoustic_n, lags, method)
                curve_n["signal"], curve_n["frequency_hz"] = name, int(f)
                curve_n["lag_s"], curve_n["mode"] = curve_n["lag_intervals"] * interval_s, "nearest"
                curve_rows.append(curve_n)
                lag_n, corr_n = best_lag(curve_n)
                exclusion_stats[key]["nearest_mode"] = {"lag_s": lag_n * interval_s, "corr": corr_n}
            lag_best, corr_best = best_lag(curve)
            boot = block_bootstrap_lag(
                cam,
                acoustic,
                lags,
                block=int(acfg.get("block_intervals", 30)),
                n_boot=int(acfg.get("n_boot", 200)),
                seed=int(cfg.get("seed", 0)),
                method=method,
            )
            boots[(name, int(f))] = boot
            boot_rows.append(
                {
                    "signal": name,
                    "frequency_hz": int(f),
                    "lag_best_s": lag_best * interval_s,
                    "corr_best": corr_best,
                    "n_pairs": int(curve.loc[curve["lag_intervals"] == lag_best, "n"].iloc[0]),
                    **{
                        k: (v * interval_s).tolist() if isinstance(v, np.ndarray) else v * interval_s
                        for k, v in boot.items()
                        if k.startswith("lag_")
                    },
                }
            )
            segments = piecewise_lags(
                cam,
                acoustic,
                lags,
                int(acfg.get("segment_intervals", 120)),
                int(acfg.get("min_pairs", 40)),
                method,
            )
            held = held_out_lag_check(segments, lags, lag_best)
            for _, row in segments.iterrows():
                seg_rows.append(
                    {
                        "signal": name,
                        "frequency_hz": int(f),
                        "segment": int(row["segment"]),
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "lag_s": row["lag_intervals"] * interval_s
                        if np.isfinite(row["lag_intervals"])
                        else np.nan,
                        "corr": row["corr"],
                        "n": int(row["n"]),
                    }
                )
            if not held.empty:
                held["signal"], held["frequency_hz"] = name, int(f)
                held_rows.append(held)
            if (name, int(f)) == primary:
                gain = float(held["gain_varying"].mean()) if not held.empty else float("nan")
                varying = bool(
                    np.isfinite(gain) and gain > float(acfg.get("varying_lag_gain_threshold", 0.02))
                )
                flags[f"band_usable_fraction_{key}"] = band_usable_fraction(
                    echogram, centres, camera_depth, half_width, lag_best, offset
                )
                if "frames_static_fraction" in series.columns:
                    # Intervals whose frames mostly show the housing were spent at the surface.
                    flags["camera_surface"] = (
                        series["frames_static_fraction"].fillna(0.0).to_numpy(dtype=float) > 0.5
                    )
                # Prior-restricted peak of the primary channel and the spread of its bootstrap peaks
                # inside the prior: the whole-series lag model.
                lag_s_grid = lags * interval_s
                prior_mask = (lag_s_grid >= float(np.nanmedian(per_interval["lag_prior_lo_s"]))) & (
                    lag_s_grid <= float(np.nanmedian(per_interval["lag_prior_hi_s"]))
                )
                series_curve = curve["corr"].to_numpy(dtype=float)
                series_ok = prior_mask & np.isfinite(series_curve)
                if series_ok.any():
                    j_s = int(np.flatnonzero(series_ok)[np.argmax(series_curve[series_ok])])
                    boot_peaks = []
                    for row_c in boot["curves"]:
                        ok_c = prior_mask & np.isfinite(row_c)
                        if ok_c.any():
                            boot_peaks.append(lag_s_grid[np.flatnonzero(ok_c)[np.argmax(row_c[ok_c])]])
                    series_model = {
                        "lag_s": float(lag_s_grid[j_s]),
                        "corr": float(series_curve[j_s]),
                        "lag_sd_s": float(np.std(boot_peaks, ddof=1))
                        if len(boot_peaks) > 1
                        else float("nan"),
                        "lag_q025_s": float(np.quantile(boot_peaks, 0.025)) if boot_peaks else float("nan"),
                        "lag_q975_s": float(np.quantile(boot_peaks, 0.975)) if boot_peaks else float("nan"),
                    }
                else:
                    series_model = {"lag_s": float("nan"), "corr": float("nan"), "lag_sd_s": float("nan")}
                primary_result = {
                    "signal": name,
                    "frequency_hz": int(f),
                    "lag_s": lag_best * interval_s,
                    "corr": corr_best,
                    "lag_ci95_s": [boot["lag_q025"] * interval_s, boot["lag_q975"] * interval_s],
                    "lag_sd_s": boot["lag_sd"] * interval_s,
                    "held_out_gain_varying": gain,
                    "varying_lag_used": varying,
                    "n_pairs": int(curve.loc[curve["lag_intervals"] == lag_best, "n"].iloc[0]),
                    "within_prior": bool(
                        np.nanmin(per_interval["lag_prior_lo_s"])
                        <= lag_best * interval_s
                        <= np.nanmax(per_interval["lag_prior_hi_s"])
                    ),
                    "prior_restricted": series_model,
                }
                a = band_series(echogram, centres, camera_depth, half_width, lag_best, offset=offset)
                with np.errstate(divide="ignore"):
                    overlay = {
                        "time": pd.to_datetime(interval_start),
                        "camera": cam,
                        "acoustic": 10 * np.log10(np.where(a > 0, a, np.nan)),
                        "camera_label": f"{name} (smoothed)",
                        "acoustic_label": f"{int(f / 1000)} kHz, lag {lag_best * interval_s:.0f} s",
                    }
    # The same estimate with each camera depth series, on the primary signal and channel.
    depth_rows: list[dict[str, Any]] = []
    p_signal = signals.get(primary[0])
    if primary_channel is not None and p_signal is not None:
        p_echo, p_centres, p_excl = primary_channel
        for dname, col in depth_columns.items():
            if col not in series:
                continue
            d_alt = series[col].to_numpy(dtype=float)
            cam_d, ac_d = paired_series(
                p_signal, p_echo, p_centres, d_alt, half_width, lags, smooth, True, p_excl, min_usable, offset
            )
            curve_d = lag_correlation(cam_d, ac_d, lags, method)
            lag_d, corr_d = best_lag(curve_d)
            boot_d = block_bootstrap_lag(
                cam_d,
                ac_d,
                lags,
                block=int(acfg.get("block_intervals", 30)),
                n_boot=int(acfg.get("n_boot", 200)),
                seed=int(cfg.get("seed", 0)),
                method=method,
            )
            depth_rows.append(
                {
                    "depth_series": dname,
                    "column": col,
                    "lag_best_s": lag_d * interval_s,
                    "corr_best": corr_d,
                    "n_pairs": int(curve_d.loc[curve_d["lag_intervals"] == lag_d, "n"].iloc[0]),
                    "lag_q025_s": boot_d["lag_q025"] * interval_s,
                    "lag_q975_s": boot_d["lag_q975"] * interval_s,
                    "intervals_inside_exclusion": int((np.isfinite(d_alt) & (d_alt < p_excl)).sum()),
                    "depth_median_m": float(np.nanmedian(d_alt)) if np.isfinite(d_alt).any() else None,
                }
            )
        pd.DataFrame(depth_rows).to_csv(target / "depth_comparison.csv", index=False)
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(target / "curves.csv", index=False)
    plateau_table = pd.DataFrame(plateau_rows)
    plateau_table.to_csv(target / "plateaus.csv", index=False)
    pd.DataFrame(plateau_curve_rows).to_csv(target / "plateau_curves.csv", index=False)
    burst_col = str(ecfg.get("signal_column", "krill_res"))
    events = (
        event_lags(
            series[burst_col].to_numpy(dtype=float),
            raw_bands,
            plateaus,
            per_interval["lag_prior_lo_s"].to_numpy(dtype=float),
            per_interval["lag_prior_hi_s"].to_numpy(dtype=float),
            int(ecfg.get("n_per_transect", 5)),
            int(round(float(ecfg.get("min_separation_s", 300)) / interval_s)),
            int(ecfg.get("smooth_intervals", 3)),
            interval_s,
        )
        if burst_col in series.columns and raw_bands
        else pd.DataFrame()
    )
    events.to_csv(target / "events.csv", index=False)
    event_values = (
        events[[c for c in events.columns if c.startswith("lag_s_")]].to_numpy(dtype=float).ravel()
        if not events.empty
        else np.array([])
    )
    event_values = event_values[np.isfinite(event_values)]
    plateau_values = (
        plateau_table["lag_prior_s"].to_numpy(dtype=float) if not plateau_table.empty else np.array([])
    )
    plateau_values = plateau_values[np.isfinite(plateau_values)]
    pooled = np.concatenate([plateau_values, event_values])
    min_sd = float(acfg.get("min_lag_sd_s", 60.0))
    lo_i = per_interval["lag_prior_lo_s"].to_numpy(dtype=float)
    hi_i = per_interval["lag_prior_hi_s"].to_numpy(dtype=float)
    models: dict[str, dict[str, Any]] = {
        "prior": {
            "distribution": "uniform",
            "lag_s_median": float(np.nanmedian((lo_i + hi_i) / 2.0)),
            "width_s_median": float(np.nanmedian(hi_i - lo_i)),
        },
        "plateau_event": {
            "distribution": "normal",
            "n_plateau_peaks": int(plateau_values.size),
            "n_event_lags": int(event_values.size),
            "lag_s": float(np.median(pooled)) if pooled.size else float("nan"),
            "mad_s": float(np.median(np.abs(pooled - np.median(pooled)))) if pooled.size else float("nan"),
            "lag_sd_s": float(max(1.4826 * np.median(np.abs(pooled - np.median(pooled))), min_sd))
            if pooled.size
            else float("nan"),
            "plateau_peaks_s": plateau_values.tolist(),
            "event_lags_s": event_values.tolist(),
        },
        "series": {"distribution": "normal", **primary_result.get("prior_restricted", {})},
    }
    lag_model = str(acfg.get("lag_model", "prior"))

    def model_parameters(name: str) -> tuple[np.ndarray | float, float, str] | None:
        """Centre, spread and distribution of one lag model, or None when the model has no estimate."""
        if name == "prior":
            return (lo_i + hi_i) / 2.0, float(np.nanmedian(hi_i - lo_i)) / np.sqrt(12.0), "uniform"
        if name == "plateau_event" and pooled.size:
            return float(models[name]["lag_s"]), float(models[name]["lag_sd_s"]), "normal"
        if name == "series" and np.isfinite(models["series"].get("lag_s", float("nan"))):
            return (
                float(models["series"]["lag_s"]),
                float(max(models["series"].get("lag_sd_s", min_sd), min_sd)),
                "normal",
            )
        return None

    variants: dict[str, pd.DataFrame] = {}
    for name in ("prior", "plateau_event", "series"):
        params = model_parameters(name)
        if params is None:
            continue
        centre, sd, distribution = params
        table = lag_mapping(
            interval_start,
            camera_depth,
            centre,
            sd,
            None,
            interval_s,
            flags={**flags, "lag_prior_lo_s": lo_i, "lag_prior_hi_s": hi_i},
        )
        table["lag_lo_s"] = lo_i
        table["lag_hi_s"] = hi_i
        table["lag_distribution"] = distribution
        table["lag_model"] = name
        table.to_parquet(target / f"mapping_{name}.parquet", index=False)
        variants[name] = table
    chosen = lag_model if lag_model in variants else "prior"
    mapping = variants[chosen]
    mapping.to_parquet(target / "mapping.parquet", index=False)
    models["chosen"] = {
        "lag_model": chosen,
        "distribution": str(mapping["lag_distribution"].iloc[0]),
        "lag_s_median": float(np.nanmedian(mapping["lag_s"])),
        "lag_sd_s": float(mapping["lag_sd_s"].iloc[0]),
        "variants_written": sorted(variants),
    }
    pd.DataFrame(boot_rows).to_csv(target / "bootstrap.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(target / "segments.csv", index=False)
    if held_rows:
        pd.concat(held_rows, ignore_index=True).to_csv(target / "held_out.csv", index=False)
    outputs: list[str | Path] = [
        target / "curves.csv",
        target / "bootstrap.csv",
        target / "segments.csv",
        target / "plateaus.csv",
        target / "plateau_curves.csv",
        target / "events.csv",
        target / "mapping.parquet",
        *[target / f"mapping_{name}.parquet" for name in variants],
    ]
    outputs += _figures(curves, boots, primary, interval_s, overlay, target, transect)
    summary = {
        "transect": transect,
        "camera": camera,
        "experiment": experiment,
        "seed": seed,
        "interval_s": interval_s,
        "n_intervals": int(len(series)),
        "n_intervals_with_depth": int(np.isfinite(camera_depth).sum()),
        "prior": prior,
        "surface_exclusion": exclusion_stats,
        "depth_series_comparison": depth_rows,
        "primary_depth_series": primary_depth,
        "lag_grid_s": [float(lags[0] * interval_s), float(lags[-1] * interval_s)],
        "primary": primary_result,
        "by_signal_and_frequency": boot_rows,
        "plateaus": [
            {
                "start": int(s),
                "end": int(e),
                "minutes": float((e - s) * interval_s / 60.0),
                "depth_m": float(np.nanmedian(camera_depth[s:e])),
            }
            for s, e in plateaus
        ],
        "plateau_span_m": plateau_span,
        "lag_models": models,
    }
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    outputs.append(target / "summary.json")
    write_manifest(target, "align-transect", run_cfg, inputs, outputs)
    return target
