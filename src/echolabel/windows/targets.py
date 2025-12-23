"""Targets per window from the camera series.

A window covers a span of camera time; its targets are the sums of the
camera series over the intervals inside that span, in every count form (hard
counts after cross-class resolution, confidence-weighted counts, expected
counts), per class, for a merged class, and as presence. The lag uncertainty
enters as offsets of the camera span. Under a normal lag distribution the
sums are stored at several multiples of the lag standard deviation and
their normal-weighted average is the expected target; under a uniform
distribution over the lag support the offsets are equally spaced over the
support with equal weights and the expected target is the exact average
over the support; with fixed targets every offset is the camera span
itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def interval_sums(
    series: pd.DataFrame, centres: np.ndarray, length_s: float | np.ndarray, columns: list[str]
) -> np.ndarray:
    """Sum of ``columns`` over the intervals whose start lies in [centre - length/2, centre + length/2).

    ``series`` needs a ``time`` column on a regular grid; missing intervals
    count as zero; ``length_s`` may vary per centre. Returns an array of
    shape (len(centres), len(columns)).
    """
    t = series["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(t)
    t = t[order]
    values = series[columns].to_numpy(dtype=float)[order]
    values = np.nan_to_num(values, nan=0.0)
    cum = np.vstack([np.zeros((1, len(columns))), np.cumsum(values, axis=0)])
    c = np.asarray(centres).astype("datetime64[ns]").astype(np.int64)
    half = (np.broadcast_to(np.asarray(length_s, dtype=float), c.shape) / 2.0 * 1e9).astype(np.int64)
    i0 = np.searchsorted(t, c - half, "left")
    i1 = np.searchsorted(t, c + half, "left")
    return cum[i1] - cum[i0]


FRAME_FORMS = ("framemean", "framemedian", "occupancy", "occupied")


def frame_statistics(
    frames: pd.DataFrame, centres: np.ndarray, length_s: float, columns: list[str]
) -> dict[str, np.ndarray]:
    """Per-frame statistics of ``columns`` over the frames in [centre - length/2, centre + length/2).

    ``frames`` needs a ``timestamp`` column and one count per frame in every
    column. Returns the mean count per frame, the median count per frame,
    the occupancy (share of frames with a positive count) and the number of
    occupied frames, each of shape (len(centres), len(columns)), and the
    number of frames per centre. A span without frames gives zeros.
    """
    t = frames["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(t)
    t = t[order]
    values = np.nan_to_num(frames[columns].to_numpy(dtype=float)[order], nan=0.0)
    c = np.asarray(centres).astype("datetime64[ns]").astype(np.int64)
    half = int(length_s / 2.0 * 1e9)
    i0 = np.searchsorted(t, c - half, "left")
    i1 = np.searchsorted(t, c + half, "left")
    shape = (len(c), len(columns))
    mean, median, occupancy = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    occupied = np.zeros(shape)
    for n in range(len(c)):
        if i1[n] > i0[n]:
            block = values[i0[n] : i1[n]]
            mean[n] = block.mean(axis=0)
            median[n] = np.median(block, axis=0)
            occupancy[n] = (block > 0).mean(axis=0)
            occupied[n] = (block > 0).sum(axis=0)
    return {
        "framemean": mean,
        "framemedian": median,
        "occupancy": occupancy,
        "occupied": occupied,
        "n_frames": i1 - i0,
    }


def build_targets(
    series: pd.DataFrame,
    grid: pd.DataFrame,
    length_s: float,
    forms: dict[str, str],
    classes: list[str],
    merged: dict[str, list[str]] | None = None,
    lag_offsets_sd: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    frames_per_second: float = 4.9,
    distribution: str = "normal",
    n_uniform_offsets: int = 9,
    frames: pd.DataFrame | None = None,
    frame_forms: tuple[str, ...] = (),
    frame_suffix: str = "_res",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Targets per window of ``grid`` and the record of the lag offsets used.

    ``grid`` needs ``time_camera`` and ``lag_sd_s``; a uniform distribution
    also needs ``lag_lo_s``, ``lag_hi_s`` and ``lag_s``. For every form and
    class the table holds one column per lag offset, ``<form>_<class>_k<i>``,
    the expected value under the lag distribution, ``<form>_<class>``, and
    the value at the central lag, ``<form>_<class>_at_lag``. Merged classes
    are sums of their members. Presence is the hard count at the central lag
    being positive. Frame coverage is the number of camera frames in the span
    over the number a full span would hold. ``distribution`` is ``normal``
    (offsets in multiples of the standard deviation, normal weights),
    ``uniform`` (equally spaced offsets over the support, equal weights, exact
    expectation) or ``fixed`` (no offsets). With a frame table (``timestamp``
    and one count column per class, ``<class><frame_suffix>``) the forms named
    in ``frame_forms`` are added in the same layout: the mean and the median
    count per frame of the span and the share of its frames with at least
    one object, whose expected value is the weighted mean over the offsets.
    """
    merged = merged or {}
    out: dict[str, np.ndarray] = {}
    centres = grid["time_camera"].to_numpy().astype("datetime64[ns]")
    n = len(grid)
    if distribution == "uniform":
        lo = grid["lag_lo_s"].to_numpy(dtype=float) - grid["lag_s"].to_numpy(dtype=float)
        hi = grid["lag_hi_s"].to_numpy(dtype=float) - grid["lag_s"].to_numpy(dtype=float)
        fractions = np.linspace(0.0, 1.0, int(n_uniform_offsets))
        shifts = [lo + (hi - lo) * f for f in fractions]  # seconds, per window
        weights = np.full(len(fractions), 1.0 / len(fractions))
        centre_index = int(np.argmin(np.abs(fractions - 0.5)))
        record = {"distribution": "uniform", "fractions": fractions.tolist(), "weights": weights.tolist()}
    elif distribution == "fixed":
        shifts = [np.zeros(n)]
        weights = np.ones(1)
        centre_index = 0
        record = {"distribution": "fixed", "weights": [1.0]}
    else:
        offsets = np.asarray(lag_offsets_sd, dtype=float)
        sd = grid["lag_sd_s"].to_numpy(dtype=float)
        shifts = [sd * z for z in offsets]
        weights = norm.pdf(offsets)
        weights = weights / weights.sum()
        centre_index = int(np.argmin(np.abs(offsets)))
        record = {"distribution": "normal", "offsets_sd": offsets.tolist(), "weights": weights.tolist()}
    all_names = list(classes) + list(merged)
    for form, suffix in forms.items():
        cols = [f"{c}{suffix}" for c in classes]
        present = [c for c in cols if c in series.columns]
        per_offset = []
        for k, shift in enumerate(shifts):
            shifted = centres + (np.asarray(shift) * 1e9).astype("timedelta64[ns]")
            sums = interval_sums(series, shifted, length_s, present) if present else np.zeros((n, 0))
            table = pd.DataFrame(sums, columns=[c[: -len(suffix)] for c in present], index=grid.index)
            for name, members in merged.items():
                table[name] = table[[m for m in members if m in table.columns]].sum(axis=1)
            table = table.reindex(columns=all_names, fill_value=0.0)
            per_offset.append(table)
            for name in all_names:
                out[f"{form}_{name}_k{k}"] = table[name].to_numpy(dtype=np.float32)
        if distribution == "uniform":
            # Exact average over the support: the sum over the span widened by the support width,
            # scaled back to one span.
            width = hi - lo
            mid = centres + (((lo + hi) / 2.0) * 1e9).astype("timedelta64[ns]")
            wide = interval_sums(series, mid, width + length_s, present) if present else np.zeros((n, 0))
            table = pd.DataFrame(wide, columns=[c[: -len(suffix)] for c in present], index=grid.index)
            for name, members in merged.items():
                table[name] = table[[m for m in members if m in table.columns]].sum(axis=1)
            expected = table.reindex(columns=all_names, fill_value=0.0).mul(
                length_s / (width + length_s), axis=0
            )
        else:
            expected = sum(w * tbl for w, tbl in zip(weights, per_offset, strict=True))
        for name in all_names:
            out[f"{form}_{name}"] = expected[name].to_numpy(dtype=np.float32)
            out[f"{form}_{name}_at_lag"] = per_offset[centre_index][name].to_numpy(dtype=np.float32)
    hard = next(iter(forms))
    if frames is not None and len(frames) and frame_forms:
        # Per-frame forms: the same offsets and weights as the sums, on the frame table.
        table = frames[["timestamp"]].copy()
        for c in classes:
            col = f"{c}{frame_suffix}"
            table[c] = frames[col].to_numpy(dtype=float) if col in frames.columns else 0.0
        for name, members in merged.items():
            table[name] = table[[m for m in members if m in table.columns]].sum(axis=1)
        per_offset = []
        for shift in shifts:
            shifted = centres + (np.asarray(shift) * 1e9).astype("timedelta64[ns]")
            per_offset.append(frame_statistics(table, shifted, length_s, all_names))
        for form in frame_forms:
            for j, name in enumerate(all_names):
                for k, stats in enumerate(per_offset):
                    out[f"{form}_{name}_k{k}"] = stats[form][:, j].astype(np.float32)
                expected = sum(w * stats[form][:, j] for w, stats in zip(weights, per_offset, strict=True))
                out[f"{form}_{name}"] = np.asarray(expected, dtype=np.float32)
                out[f"{form}_{name}_at_lag"] = per_offset[centre_index][form][:, j].astype(np.float32)
    for name in all_names:
        out[f"presence_{name}"] = (out[f"{hard}_{name}_at_lag"] > 0).astype(np.int8)
    if "n_frames" in series.columns:
        frames = interval_sums(series, centres, length_s, ["n_frames"])[:, 0]
        out["n_frames"] = frames.astype(np.int32)
        out["frame_fraction"] = (frames / (frames_per_second * length_s)).astype(np.float32)
    return pd.DataFrame(out, index=grid.index), record


def log1p_targets(targets: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """log(1 + y) of every count column that starts with one of ``prefixes``."""
    cols = [c for c in targets.columns if c.startswith(prefixes) and not c.startswith("presence_")]
    logged = np.log1p(targets[cols].clip(lower=0))
    logged.columns = [f"log1p_{c}" for c in cols]
    return logged


def summarize_targets(targets: pd.DataFrame, classes: list[str], form: str = "hard") -> dict[str, Any]:
    """Presence rate, mean and 95th percentile per class for one form at the estimated lag."""
    out: dict[str, Any] = {}
    for name in classes:
        col = f"{form}_{name}_at_lag"
        if col in targets:
            v = targets[col].to_numpy(dtype=float)
            out[name] = {
                "presence_rate": float((v > 0).mean()) if v.size else None,
                "mean": float(v.mean()) if v.size else None,
                "q95": float(np.quantile(v, 0.95)) if v.size else None,
            }
    return out
