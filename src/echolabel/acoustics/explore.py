"""Exploration of the backscatter around each camera transect.

For every transect window the store is read at full resolution and all
frequencies, converted to decibels, and summarised: per-range quantiles per
frequency (noise floor and signal levels against range), near-surface levels,
a search for a bottom echo, impulsive-noise pings, and a coarse echogram at a
few seconds by a few metres for figures. These summaries decide the masks
and filters of the preprocessing stage.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .preprocess import to_db

log = logging.getLogger(__name__)

__all__ = [
    "bottom_candidates",
    "coarse_echogram",
    "explore_window",
    "impulsive_pings",
    "range_profile",
    "to_db",
]


def range_profile(
    db: np.ndarray, rng: np.ndarray, freqs: np.ndarray, quantiles=(0.1, 0.5, 0.95, 0.99)
) -> pd.DataFrame:
    """Quantiles of dB and the missing fraction per range sample and frequency; ``db`` has shape (F, P, R)."""
    rows = []
    for i, f in enumerate(freqs):
        block = db[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qs = np.nanquantile(block, quantiles, axis=0)
        nan_frac = np.isnan(block).mean(axis=0)
        table = pd.DataFrame({"frequency_hz": float(f), "range_m": rng, "nan_fraction": nan_frac})
        for q, values in zip(quantiles, qs, strict=True):
            table[f"db_q{int(q * 100):02d}"] = values
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def bottom_candidates(
    db: np.ndarray, rng: np.ndarray, min_range_m: float = 50.0, strong_db: float = -35.0
) -> pd.DataFrame:
    """Per ping, the strongest return beyond ``min_range_m`` and whether it exceeds ``strong_db``.

    ``db`` is one frequency with shape (P, R). A bottom echo shows as a strong
    return at a range that drifts slowly from ping to ping.
    """
    sel = rng >= min_range_m
    block = np.where(np.isnan(db[:, sel]), -np.inf, db[:, sel])
    idx = block.argmax(axis=1)
    peak = block[np.arange(block.shape[0]), idx]
    peak = np.where(np.isfinite(peak), peak, np.nan)
    return pd.DataFrame({"peak_db": peak, "peak_range_m": rng[sel][idx], "strong": peak >= strong_db})


def impulsive_pings(
    db: np.ndarray,
    rng: np.ndarray,
    lo_m: float = 50.0,
    hi_m: float = 400.0,
    z: float = 5.0,
    window: int = 51,
) -> np.ndarray:
    """Pings whose mean level between ``lo_m`` and ``hi_m`` exceeds the local median by ``z`` scales."""
    sel = (rng >= lo_m) & (rng <= hi_m)
    with np.errstate(all="ignore"):
        level = np.nanmean(db[:, sel], axis=1)
    s = pd.Series(level)
    med = s.rolling(window, center=True, min_periods=5).median()
    mad = 1.4826 * (s - med).abs().rolling(window, center=True, min_periods=5).median()
    return ((s - med) > z * mad.clip(lower=0.5)).to_numpy()


def coarse_echogram(
    sv: np.ndarray,
    ping_time: np.ndarray,
    depth: np.ndarray,
    time_step_s: float,
    depth_step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean linear sv in time by depth cells, returned in dB with the cell edges.

    ``sv`` has shape (F, P, R), ``depth`` is per range sample. Cells without
    any finite value are NaN.
    """
    t_ns = ping_time.astype("datetime64[ns]").astype(np.int64)
    t0 = t_ns.min()
    t_bin = ((t_ns - t0) // int(time_step_s * 1e9)).astype(int)
    d_bin = np.floor((depth - depth.min()) / depth_step_m).astype(int)
    n_t, n_d = t_bin.max() + 1, d_bin.max() + 1
    out = np.full((sv.shape[0], n_t, n_d), np.nan, dtype=np.float32)
    for i in range(sv.shape[0]):
        block = sv[i]
        finite = np.isfinite(block)
        vals = np.where(finite, block, 0.0)
        sums = np.zeros((n_t, n_d))
        counts = np.zeros((n_t, n_d))
        for d_idx in range(n_d):
            cols = d_bin == d_idx
            if not cols.any():
                continue
            s_col = vals[:, cols].sum(axis=1)
            c_col = finite[:, cols].sum(axis=1)
            np.add.at(sums[:, d_idx], t_bin, s_col)
            np.add.at(counts[:, d_idx], t_bin, c_col)
        with np.errstate(all="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            out[i] = 10.0 * np.log10(np.maximum(mean, 1e-13))
            out[i][counts == 0] = np.nan
    time_edges = np.datetime64(int(t0), "ns") + (np.arange(n_t + 1) * int(time_step_s * 1e9)).astype(
        "timedelta64[ns]"
    )
    depth_edges = depth.min() + np.arange(n_d + 1) * depth_step_m
    return out, time_edges, depth_edges


def explore_window(
    ds: xr.Dataset,
    start: str,
    end: str,
    draft_m: float,
    floor_db: float,
    time_step_s: float,
    depth_step_m: float,
    reference_hz: float,
) -> dict[str, Any]:
    """Load one time window at all frequencies and compute every summary."""
    from .store import frequency_position

    sub = ds.sel(ping_time=slice(start, end))
    sv = sub["sv"].transpose("frequency", "ping_time", "range").values
    freqs = sub["frequency"].values
    rng = sub["range"].values
    ping_time = sub["ping_time"].values
    depth = rng + draft_m
    db = to_db(sv, floor_db)
    profile = range_profile(db, rng, freqs)
    ref = frequency_position(sub, reference_hz)
    bottom = bottom_candidates(db[ref], rng)
    bottom["ping_time"] = ping_time
    impulsive = impulsive_pings(db[ref], rng)
    near_surface = {float(f): float(np.nanmedian(db[i][:, depth <= 20.0])) for i, f in enumerate(freqs)}
    coarse, t_edges, d_edges = coarse_echogram(sv, ping_time, depth, time_step_s, depth_step_m)
    summary = {
        "start": str(ping_time[0]),
        "end": str(ping_time[-1]),
        "n_pings": int(ping_time.size),
        "frequencies_hz": [float(f) for f in freqs],
        "reference_hz": reference_hz,
        "draft_m": draft_m,
        "near_surface_median_db_0_20m": near_surface,
        "bottom": {
            "fraction_pings_with_strong_peak": float(np.nanmean(bottom["strong"].astype(float))),
            "peak_range_m_median": float(np.nanmedian(bottom.loc[bottom["strong"], "peak_range_m"]))
            if bottom["strong"].any()
            else None,
            "peak_db_median": float(np.nanmedian(bottom["peak_db"])),
        },
        "impulsive_pings": int(impulsive.sum()),
        "impulsive_fraction": float(impulsive.mean()),
        "coarse_shape": list(coarse.shape),
    }
    return {
        "summary": summary,
        "profile": profile,
        "bottom": bottom,
        "impulsive": impulsive,
        "coarse": coarse,
        "time_edges": t_edges,
        "depth_edges": d_edges,
        "freqs": freqs,
    }
