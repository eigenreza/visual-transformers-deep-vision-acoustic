"""Lag between the vessel-mounted echosounder and the towed camera.

The camera trails the vessel, so the two sensors observe the same water at
different times. A physical prior bounds the lag from vessel speed and the
horizontal distance between vessel and camera. The empirical estimate
correlates a camera-derived series with the backscatter at the camera depth
over a grid of lags; a moving block bootstrap gives its uncertainty, and a
piecewise estimate with a held-out check tells whether the lag drifts within
a transect. All series live on a common grid of fixed intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres between points given in degrees."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlam = np.deg2rad(lon2) - np.deg2rad(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def vessel_speed(
    ping_time: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, window_s: float = 60.0
) -> pd.DataFrame:
    """Vessel speed over ground in m/s from positions ``window_s`` apart, assigned to the midpoint time."""
    t = ping_time.astype("datetime64[ns]").astype(np.int64) / 1e9
    j = np.searchsorted(t, t + window_s)
    valid = j < len(t)
    i = np.flatnonzero(valid)
    j = j[valid]
    dist = haversine_m(latitude[i], longitude[i], latitude[j], longitude[j])
    dt = t[j] - t[i]
    mid = ((t[i] + t[j]) / 2 * 1e9).astype("datetime64[ns]")
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, dist / dt, np.nan)
    return pd.DataFrame({"time": mid, "speed_m_s": speed})


def setback_from_warp(warp_length_m: float, depth_m: np.ndarray | float) -> np.ndarray | float:
    """Horizontal vessel-to-camera distance for a straight warp of given length at a given depth."""
    return np.sqrt(
        np.maximum(np.asarray(warp_length_m, dtype=float) ** 2 - np.asarray(depth_m, dtype=float) ** 2, 0.0)
    )


def lag_bounds(speed_m_s: np.ndarray, setback_m: tuple[float, float]) -> dict[str, float]:
    """Lag range in seconds implied by a setback range and the observed speed quantiles."""
    s = np.asarray(speed_m_s, dtype=float)
    s = s[np.isfinite(s) & (s > 0.1)]
    q10, q50, q90 = np.quantile(s, [0.1, 0.5, 0.9])
    lo, hi = setback_m
    return {
        "speed_m_s_q10": float(q10),
        "speed_m_s_median": float(q50),
        "speed_m_s_q90": float(q90),
        "setback_m": [float(lo), float(hi)],
        "lag_s_at_median_speed": [float(lo / q50), float(hi / q50)],
        "lag_s_min": float(lo / q90),
        "lag_s_max": float(hi / q10),
    }


def lag_bounds_per_interval(
    speed_m_s: np.ndarray,
    depth_m: np.ndarray,
    warp_length_m: tuple[float, float],
    max_lag_s: float | None = None,
) -> pd.DataFrame:
    """Lag bounds per interval from camera depth and speed; the setback is at most sqrt(L^2 - d^2).

    The lower bound uses the shorter warp and is zero where the camera is
    deeper than that warp is long; the upper bound uses the longer warp.
    Both are capped at ``max_lag_s`` when given: where the vessel nearly
    stops the geometric bound grows without limit, but the record kept
    before each camera interval, and the lag grid, are finite.
    """
    s = np.asarray(speed_m_s, dtype=float)
    d = np.asarray(depth_m, dtype=float)
    lo = setback_from_warp(warp_length_m[0], d)
    hi = setback_from_warp(warp_length_m[1], d)
    with np.errstate(invalid="ignore", divide="ignore"):
        lag_lo = np.where(s > 0.1, lo / s, np.nan)
        lag_hi = np.where(s > 0.1, hi / s, np.nan)
    if max_lag_s is not None:
        lag_hi = np.minimum(lag_hi, float(max_lag_s))
        lag_lo = np.minimum(lag_lo, lag_hi)
    return pd.DataFrame(
        {"speed_m_s": s, "camera_depth_m": d, "lag_prior_lo_s": lag_lo, "lag_prior_hi_s": lag_hi}
    )


def echogram_at_intervals(
    ping_time: np.ndarray,
    db: np.ndarray,
    hard_mask: np.ndarray,
    depth: np.ndarray,
    interval_start: np.ndarray,
    interval_s: float,
    depth_step_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean linear backscatter per interval and depth cell, with the cell centres.

    ``db`` and ``hard_mask`` (True where a sample must not be used) have shape
    (P, R); ``depth`` has shape (R,). Intervals start at ``interval_start``
    and last ``interval_s``. Cells without usable samples are NaN.
    """
    t = ping_time.astype("datetime64[ns]").astype(np.int64) / 1e9
    t0 = interval_start.astype("datetime64[ns]").astype(np.int64) / 1e9
    n_t = len(t0)
    idx = np.floor((t - t0[0]) / interval_s).astype(int)
    in_range = (idx >= 0) & (idx < n_t)
    cell = np.floor((depth - depth.min()) / depth_step_m).astype(int)
    n_d = int(cell.max()) + 1
    centres = depth.min() + (np.arange(n_d) + 0.5) * depth_step_m
    with np.errstate(all="ignore"):
        lin = np.where(hard_mask | ~np.isfinite(db), 0.0, 10.0 ** (db.astype(np.float32) / 10.0))
    usable = (~hard_mask & np.isfinite(db)).astype(np.float32)
    # Collapse range samples to depth cells, then pings to intervals.
    sums_d = np.zeros((db.shape[0], n_d), dtype=np.float64)
    cnt_d = np.zeros((db.shape[0], n_d), dtype=np.float64)
    for c in range(n_d):
        cols = cell == c
        if cols.any():
            sums_d[:, c] = lin[:, cols].sum(axis=1)
            cnt_d[:, c] = usable[:, cols].sum(axis=1)
    sums = np.zeros((n_t, n_d))
    cnt = np.zeros((n_t, n_d))
    np.add.at(sums, idx[in_range], sums_d[in_range])
    np.add.at(cnt, idx[in_range], cnt_d[in_range])
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    return mean, centres


def band_series(
    echogram: np.ndarray,
    centres: np.ndarray,
    camera_depth: np.ndarray,
    half_width_m: float,
    lag_intervals: int = 0,
    exclusion_depth_m: float | None = None,
    min_usable_fraction: float = 0.5,
    offset: int = 0,
) -> np.ndarray:
    """Backscatter at the camera depth per camera interval, read ``lag_intervals`` earlier.

    The value for camera interval ``t`` is the mean of the usable cells within
    ``half_width_m`` of the camera depth at ``t``, taken from the echogram
    interval ``offset + t - lag_intervals`` (the vessel passed first; the
    echogram may start ``offset`` intervals before the camera series).
    Intervals are NaN without a camera depth, when the camera depth lies
    above ``exclusion_depth_m`` (inside the surface exclusion of the
    channel), or when fewer than ``min_usable_fraction`` of the band cells
    are usable.
    """
    n_t = len(camera_depth)
    out = np.full(n_t, np.nan)
    for t in range(n_t):
        src = offset + t - lag_intervals
        d = camera_depth[t]
        if src < 0 or src >= echogram.shape[0] or not np.isfinite(d):
            continue
        if exclusion_depth_m is not None and d < exclusion_depth_m:
            continue
        sel = np.abs(centres - d) <= half_width_m
        vals = echogram[src, sel]
        finite = np.isfinite(vals)
        if sel.sum() and finite.any() and finite.sum() >= min_usable_fraction * sel.sum():
            out[t] = vals[finite].mean()
    return out


def band_usable_fraction(
    echogram: np.ndarray,
    centres: np.ndarray,
    camera_depth: np.ndarray,
    half_width_m: float,
    lag_intervals: int = 0,
    offset: int = 0,
) -> np.ndarray:
    """Fraction of usable cells in the band around the camera depth per interval, ``lag_intervals`` back."""
    n_t = len(camera_depth)
    out = np.full(n_t, np.nan)
    for t in range(n_t):
        src = offset + t - lag_intervals
        d = camera_depth[t]
        if src < 0 or src >= echogram.shape[0] or not np.isfinite(d):
            continue
        sel = np.abs(centres - d) <= half_width_m
        if sel.sum():
            out[t] = np.isfinite(echogram[src, sel]).mean()
    return out


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return pd.Series(x).rolling(window, center=True, min_periods=max(1, window // 2)).mean().to_numpy()


def _rank_corr(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 8:
        return float("nan"), n
    s = pd.DataFrame({"a": a[ok], "b": b[ok]})
    if s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return float("nan"), n
    return float(s["a"].corr(s["b"], method=method)), n


def paired_series(
    camera_signal: np.ndarray,
    echogram: np.ndarray,
    centres: np.ndarray,
    camera_depth: np.ndarray,
    half_width_m: float,
    lags: np.ndarray,
    smooth: int = 3,
    log_acoustic: bool = True,
    exclusion_depth_m: float | None = None,
    min_usable_fraction: float = 0.5,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed camera series (T,) and, per lag, the smoothed acoustic series paired with it (L, T)."""
    cam = _smooth(np.asarray(camera_signal, dtype=float), smooth)
    rows = []
    for k in lags:
        a = band_series(
            echogram,
            centres,
            camera_depth,
            half_width_m,
            int(k),
            exclusion_depth_m,
            min_usable_fraction,
            offset,
        )
        if log_acoustic:
            with np.errstate(divide="ignore"):
                a = 10.0 * np.log10(np.where(a > 0, a, np.nan))
        rows.append(_smooth(a, smooth))
    return cam, np.vstack(rows)


def lag_correlation(
    cam: np.ndarray, acoustic: np.ndarray, lags: np.ndarray, method: str = "spearman"
) -> pd.DataFrame:
    """Correlation between the camera series and the acoustic series at every lag."""
    rows = []
    for k, a in zip(lags, acoustic, strict=True):
        r, n = _rank_corr(cam, a, method)
        rows.append({"lag_intervals": int(k), "corr": r, "n": n})
    return pd.DataFrame(rows)


def best_lag(curve: pd.DataFrame) -> tuple[int, float]:
    """Lag with the highest correlation; NaN correlations are ignored."""
    ok = curve.dropna(subset=["corr"])
    if ok.empty:
        return 0, float("nan")
    row = ok.loc[ok["corr"].idxmax()]
    return int(row["lag_intervals"]), float(row["corr"])


def block_bootstrap_lag(
    cam: np.ndarray,
    acoustic: np.ndarray,
    lags: np.ndarray,
    block: int = 30,
    n_boot: int = 200,
    seed: int = 0,
    method: str = "spearman",
) -> dict[str, Any]:
    """Moving block bootstrap of the best lag: resample blocks of consecutive intervals, keep the pairing.

    Returns the best lag per resample, the correlation curve per resample
    and summary quantiles of the lag.
    """
    rng = np.random.default_rng(seed)
    n_t = len(cam)
    n_blocks = int(np.ceil(n_t / block))
    starts_all = np.arange(0, max(n_t - block + 1, 1))
    lag_samples = np.zeros(n_boot, dtype=int)
    curves = np.full((n_boot, len(lags)), np.nan)
    for b in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n_t)) for s in starts])[:n_t]
        c = cam[idx]
        best, best_r = 0, -np.inf
        for j, a in enumerate(acoustic):
            r, _ = _rank_corr(c, a[idx], method)
            curves[b, j] = r
            if np.isfinite(r) and r > best_r:
                best, best_r = int(lags[j]), r
        lag_samples[b] = best
    q = np.quantile(lag_samples, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "lag_samples": lag_samples,
        "curves": curves,
        "lag_q025": float(q[0]),
        "lag_q25": float(q[1]),
        "lag_median": float(q[2]),
        "lag_q75": float(q[3]),
        "lag_q975": float(q[4]),
        "lag_sd": float(lag_samples.std(ddof=1)) if n_boot > 1 else 0.0,
    }


def piecewise_lags(
    cam: np.ndarray,
    acoustic: np.ndarray,
    lags: np.ndarray,
    segment: int = 120,
    min_pairs: int = 40,
    method: str = "spearman",
) -> pd.DataFrame:
    """Best lag per segment of consecutive intervals, with the whole correlation curve per segment."""
    n_t = len(cam)
    rows = []
    for s_idx, start in enumerate(range(0, n_t, segment)):
        end = min(start + segment, n_t)
        sl = slice(start, end)
        curve = [(_rank_corr(cam[sl], a[sl], method)) for a in acoustic]
        corr = np.array([c[0] for c in curve])
        n = np.array([c[1] for c in curve])
        usable = np.isfinite(corr) & (n >= min_pairs)
        if not usable.any():
            rows.append(
                {
                    "segment": s_idx,
                    "start": start,
                    "end": end,
                    "lag_intervals": np.nan,
                    "corr": np.nan,
                    "n": int(n.max()),
                    "curve": corr,
                }
            )
            continue
        j = int(np.nanargmax(np.where(usable, corr, -np.inf)))
        rows.append(
            {
                "segment": s_idx,
                "start": start,
                "end": end,
                "lag_intervals": int(lags[j]),
                "corr": float(corr[j]),
                "n": int(n[j]),
                "curve": corr,
            }
        )
    return pd.DataFrame(rows)


def held_out_lag_check(segments: pd.DataFrame, lags: np.ndarray, global_lag: int) -> pd.DataFrame:
    """Held-out correlation per segment at the constant lag and at the lag predicted from its neighbours.

    The neighbour prediction is the mean best lag of the adjacent segments
    (rounded to the grid); the segment's own curve is only read, never used
    to choose its lag. A positive mean difference favours a varying lag.
    """
    lag_list = list(lags)
    rows = []
    for i, row in segments.iterrows():
        curve = row["curve"]
        neighbours = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(segments) and np.isfinite(segments.loc[j, "lag_intervals"]):
                neighbours.append(segments.loc[j, "lag_intervals"])
        if not neighbours or not np.isfinite(curve).any():
            continue
        predicted = int(lags[np.argmin(np.abs(lags - np.mean(neighbours)))])
        rows.append(
            {
                "segment": int(row["segment"]),
                "lag_constant": int(global_lag),
                "lag_neighbours": predicted,
                "corr_constant": float(curve[lag_list.index(global_lag)])
                if global_lag in lag_list
                else np.nan,
                "corr_neighbours": float(curve[lag_list.index(predicted)]),
                "corr_own_best": float(row["corr"]),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["gain_varying"] = table["corr_neighbours"] - table["corr_constant"]
    return table


def detrend(x: np.ndarray, window: int) -> np.ndarray:
    """``x`` minus its centred running mean over ``window`` intervals (NaN kept where ``x`` is NaN)."""
    if window <= 1:
        return np.asarray(x, dtype=float)
    trend = pd.Series(x).rolling(window, center=True, min_periods=max(3, window // 3)).mean().to_numpy()
    return np.asarray(x, dtype=float) - trend


def find_plateaus(depth: np.ndarray, min_intervals: int = 60, span_m: float = 10.0) -> list[tuple[int, int]]:
    """Runs of at least ``min_intervals`` intervals whose depth stays within a range of ``span_m``.

    Runs are grown greedily from left to right; a run ends at the first
    interval that would widen the range beyond ``span_m`` or has no depth.
    """
    out: list[tuple[int, int]] = []
    n = len(depth)
    i = 0
    while i < n:
        if not np.isfinite(depth[i]):
            i += 1
            continue
        lo = hi = float(depth[i])
        j = i + 1
        while j < n and np.isfinite(depth[j]) and max(hi, depth[j]) - min(lo, depth[j]) <= span_m:
            lo, hi = min(lo, float(depth[j])), max(hi, float(depth[j]))
            j += 1
        if j - i >= min_intervals:
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def plateau_lag_curve(
    cam_signal: np.ndarray,
    acoustic: np.ndarray,
    lags: np.ndarray,
    start: int,
    end: int,
    detrend_intervals: int = 18,
    method: str = "spearman",
    block: int = 12,
    n_boot: int = 100,
    seed: int = 0,
    prior_s: tuple[float, float] | None = None,
    interval_s: float = 10.0,
    n_null: int = 0,
) -> dict[str, Any]:
    """Lag correlation inside one plateau after detrending both series, with a block bootstrap of the peak.

    ``acoustic`` holds the band series per lag (L, T) in dB; it and the camera
    signal (log1p of counts) lose a centred running mean over
    ``detrend_intervals`` so that only along-track variation remains. The
    bootstrap peak is restricted to ``prior_s`` when given. With ``n_null``
    the search is repeated on that many block permutations of the camera
    series; the quantiles of the maximum correlation under this null, the
    share of null maxima at or above the observed one, and the quartiles of
    the lag at which the null maximum falls are returned.
    """
    cam = detrend(np.log1p(np.asarray(cam_signal, dtype=float)), detrend_intervals)[start:end]
    ac = np.vstack([detrend(row, detrend_intervals) for row in acoustic])[:, start:end]
    curve = np.array([_rank_corr(cam, ac[j], method)[0] for j in range(len(lags))])
    lag_s = lags * interval_s
    inside = (
        np.ones(len(lags), dtype=bool) if prior_s is None else (lag_s >= prior_s[0]) & (lag_s <= prior_s[1])
    )

    def peak(c: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        m = mask & np.isfinite(c)
        if not m.any():
            return float("nan"), float("nan")
        j = int(np.flatnonzero(m)[np.argmax(c[m])])
        return float(lag_s[j]), float(c[j])

    rng = np.random.default_rng(seed)
    n = end - start
    starts_all = np.arange(0, max(n - block + 1, 1))
    peaks = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.choice(starts_all, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        c = np.array([_rank_corr(cam[idx], ac[j, idx], method)[0] for j in range(len(lags))])
        peaks[b] = peak(c, inside)[0]
    lag_all, corr_all = peak(curve, np.ones(len(lags), dtype=bool))
    lag_prior, corr_prior = peak(curve, inside)
    # Null calibration of the search: the camera series is permuted in blocks, which keeps its
    # short-range structure and breaks its alignment with the echogram, and the maximum over the
    # same lags is taken again. The generator is separate so that the bootstrap above is unchanged.
    null = np.full(n_null, np.nan)
    null_lag = np.full(n_null, np.nan)
    if n_null and np.isfinite(corr_prior):
        rng_null = np.random.default_rng(seed + 7919)
        pieces = [np.arange(k, min(k + block, n)) for k in range(0, n, block)]
        searched = np.flatnonzero(inside)
        for b in range(n_null):
            idx = np.concatenate([pieces[k] for k in rng_null.permutation(len(pieces))])
            values = np.array([_rank_corr(cam[idx], ac[j], method)[0] for j in searched], dtype=float)
            if np.isfinite(values).any():
                best = int(np.nanargmax(values))
                null[b], null_lag[b] = values[best], lag_s[searched[best]]
    null_finite = null[np.isfinite(null)]
    null_q = np.quantile(null_finite, [0.5, 0.95, 0.99]) if null_finite.size else np.full(3, np.nan)
    p_null = (
        float((1 + (null_finite >= corr_prior).sum()) / (1 + null_finite.size))
        if null_finite.size
        else float("nan")
    )
    null_lag_finite = null_lag[np.isfinite(null_lag)]
    null_lag_q = (
        np.quantile(null_lag_finite, [0.25, 0.5, 0.75]) if null_lag_finite.size else np.full(3, np.nan)
    )
    finite = peaks[np.isfinite(peaks)]
    q = np.quantile(finite, [0.025, 0.5, 0.975]) if finite.size else np.full(3, np.nan)
    return {
        "start": int(start),
        "end": int(end),
        "n": int(np.isfinite(cam).sum()),
        "lag_best_s": lag_all,
        "corr_best": corr_all,
        "lag_prior_s": lag_prior,
        "corr_prior": corr_prior,
        "lag_q025_s": float(q[0]),
        "lag_median_s": float(q[1]),
        "lag_q975_s": float(q[2]),
        "n_null": int(null_finite.size),
        "null_q50": float(null_q[0]),
        "null_q95": float(null_q[1]),
        "null_q99": float(null_q[2]),
        "p_null": p_null,
        "null_lag_q25_s": float(null_lag_q[0]),
        "null_lag_q50_s": float(null_lag_q[1]),
        "null_lag_q75_s": float(null_lag_q[2]),
        "curve": curve,
    }


def event_lags(
    burst_signal: np.ndarray,
    acoustic_by_channel: dict[int, np.ndarray],
    plateaus: list[tuple[int, int]],
    lag_lo_s: np.ndarray,
    lag_hi_s: np.ndarray,
    n_events: int = 5,
    min_separation: int = 30,
    smooth: int = 3,
    interval_s: float = 10.0,
) -> pd.DataFrame:
    """Implied lag per detection burst: the strongest patch at the burst depth within the prior window.

    Bursts are the ``n_events`` largest values of the smoothed ``burst_signal``
    inside the plateaus, at least ``min_separation`` intervals apart. For each
    burst and channel the band series at the burst's own depth is read at lags
    between the interval's prior bounds (``acoustic_by_channel[f][k, t]`` is
    the band value at camera interval ``t`` read ``k`` intervals earlier), and
    the lag of its maximum after smoothing is the implied lag.
    """
    signal = _smooth(np.asarray(burst_signal, dtype=float), smooth)
    inside = np.zeros(len(signal), dtype=bool)
    for start, end in plateaus:
        inside[start:end] = True
    candidates = np.where(inside & np.isfinite(signal), signal, -np.inf)
    chosen: list[int] = []
    for idx in np.argsort(candidates)[::-1]:
        if not np.isfinite(candidates[idx]) or candidates[idx] <= 0:
            break
        if all(abs(int(idx) - c) >= min_separation for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == n_events:
            break
    rows = []
    for b in chosen:
        k0 = int(np.floor(lag_lo_s[b] / interval_s))
        k1 = int(np.ceil(lag_hi_s[b] / interval_s))
        row: dict[str, Any] = {
            "interval": b,
            "burst": float(signal[b]),
            "lag_lo_s": float(lag_lo_s[b]),
            "lag_hi_s": float(lag_hi_s[b]),
        }
        for f, acoustic in acoustic_by_channel.items():
            ks = np.arange(max(k0, 0), min(k1, acoustic.shape[0] - 1) + 1)
            profile = _smooth(acoustic[ks, b], smooth) if ks.size else np.array([])
            if ks.size == 0 or np.isfinite(profile).sum() < 3:
                row[f"lag_s_{int(f)}"] = np.nan
                row[f"peak_db_{int(f)}"] = np.nan
                continue
            j = int(np.nanargmax(profile))
            row[f"lag_s_{int(f)}"] = float(ks[j] * interval_s)
            row[f"peak_db_{int(f)}"] = float(profile[j])
        rows.append(row)
    return pd.DataFrame(rows)


def lag_mapping(
    interval_start: np.ndarray,
    camera_depth: np.ndarray,
    lag_s: float | np.ndarray,
    lag_sd_s: float,
    segment_lags_s: pd.DataFrame | None = None,
    interval_s: float = 10.0,
    flags: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per camera interval: the vessel time that saw the same water, the lag used, its uncertainty, flags."""
    t = pd.to_datetime(interval_start)
    lag = np.broadcast_to(np.asarray(lag_s, dtype=float), (len(t),)).copy()
    if segment_lags_s is not None and not segment_lags_s.empty:
        for _, row in segment_lags_s.iterrows():
            if np.isfinite(row["lag_s"]):
                lag[int(row["start"]) : int(row["end"])] = float(row["lag_s"])
    table = pd.DataFrame(
        {
            "time_camera": t,
            "time_vessel": t - pd.to_timedelta(lag, unit="s"),
            "camera_depth_m": camera_depth,
            "lag_s": lag,
            "lag_sd_s": float(lag_sd_s),
            "interval_s": interval_s,
        }
    )
    for name, values in (flags or {}).items():
        table[name] = values
    return table
