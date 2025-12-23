"""Whole-store statistics of the backscattering data.

All heavy reductions over ``sv`` are expressed as one dask graph and computed
together, so the store is read once. Per-range profiles reveal the usable
range of each channel; per-ping counts reveal dropped or empty pings.
"""

from __future__ import annotations

import logging
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def ping_time_summary(
    ping_time: np.ndarray, gap_thresholds_s: tuple[float, ...]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Interval percentiles, monotonicity and gap table of the ping time axis."""
    t = ping_time.astype("datetime64[ms]").astype(np.int64)
    d = np.diff(t) / 1000.0
    summary: dict[str, Any] = {
        "n_pings": int(t.size),
        "first": str(ping_time[0]),
        "last": str(ping_time[-1]),
        "monotonic_increasing": bool((d > 0).all()),
        "n_nonpositive_steps": int((d <= 0).sum()),
        "interval_s": {f"p{q}": float(np.percentile(d, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)},
        "gaps": {
            str(thr): {"count": int((d > thr).sum()), "total_hours": float(d[d > thr].sum() / 3600)}
            for thr in gap_thresholds_s
        },
    }
    thr0 = min(gap_thresholds_s)
    idx = np.nonzero(d > thr0)[0]
    gaps = (
        pd.DataFrame({"start": ping_time[idx], "end": ping_time[idx + 1], "seconds": d[idx]})
        .sort_values("seconds", ascending=False)
        .reset_index(drop=True)
    )
    return summary, gaps


def transect_coverage(ds: xr.Dataset, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Number of pings and interval statistics inside each named time window."""
    rows = []
    for name, (start, end) in windows.items():
        sel = ds["ping_time"].sel(ping_time=slice(start, end)).values
        if sel.size < 2:
            rows.append({"transect": name, "start": start, "end": end, "n_pings": int(sel.size)})
            continue
        d = np.diff(sel.astype("datetime64[ms]").astype(np.int64)) / 1000.0
        rows.append(
            {
                "transect": name,
                "start": start,
                "end": end,
                "n_pings": int(sel.size),
                "first_ping": str(sel[0]),
                "last_ping": str(sel[-1]),
                "median_interval_s": float(np.median(d)),
                "p99_interval_s": float(np.percentile(d, 99)),
                "max_gap_s": float(d.max()),
                "n_gaps_over_5s": int((d > 5).sum()),
            }
        )
    return pd.DataFrame(rows)


def transducer_draft_summary(ds: xr.Dataset) -> tuple[dict[str, Any], pd.DataFrame]:
    """Constancy of the transducer draft across pings and channels."""
    draft = ds["transducer_draft"].transpose("frequency", "ping_time").values
    freqs = ds["frequency"].values
    rows = []
    for i, f in enumerate(freqs):
        v = draft[i]
        finite = v[np.isfinite(v)]
        uniq, counts = np.unique(finite, return_counts=True)
        rows.append(
            {
                "frequency_hz": float(f),
                "n_nan": int(v.size - finite.size),
                "min": float(finite.min()) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
                "n_unique": int(uniq.size),
                "values": ";".join(f"{u:g}:{c}" for u, c in zip(uniq[:10], counts[:10], strict=True)),
            }
        )
    table = pd.DataFrame(rows)
    finite_all = draft[np.isfinite(draft)]
    col_max = np.nanmax(draft, axis=0)
    col_min = np.nanmin(draft, axis=0)
    summary = {
        "constant_over_cruise": bool(np.unique(finite_all).size == 1),
        "identical_across_channels": bool(
            np.all(col_max[np.isfinite(col_max)] == col_min[np.isfinite(col_min)])
        ),
        "unique_values": np.unique(finite_all)[:20].tolist(),
    }
    return summary, table


def sv_statistics(
    ds: xr.Dataset,
    floors_db: tuple[float, ...],
    histogram_range_db: tuple[float, float],
    histogram_bin_db: float,
    n_workers: int = 32,
) -> dict[str, Any]:
    """One pass over ``sv``: missing values, floor fractions, zeros, negatives and dB histograms.

    Returns numpy arrays keyed by name. Per-range arrays have shape
    (frequency, range); per-ping arrays have shape (frequency, ping_time);
    histograms have shape (frequency, n_bins) with edges under ``hist_edges``.
    """
    sv = ds["sv"].data
    if not isinstance(sv, da.Array):
        sv = da.from_array(sv)
    tasks: dict[str, da.Array] = {}
    isnan = da.isnan(sv)
    tasks["nan_by_range"] = isnan.sum(axis=1)
    tasks["nan_by_ping"] = isnan.sum(axis=2)
    tasks["zero_by_range"] = (sv == 0).sum(axis=1)
    tasks["negative_by_range"] = (sv < 0).sum(axis=1)
    for f_db in floors_db:
        thr = 10.0 ** (f_db / 10.0)
        mask = sv <= thr
        tasks[f"floor{int(f_db)}_by_range"] = mask.sum(axis=1)
        tasks[f"floor{int(f_db)}_by_ping"] = mask.sum(axis=2)
    lo, hi = histogram_range_db
    n_bins = int(round((hi - lo) / histogram_bin_db))
    db = 10.0 * da.log10(da.maximum(sv, 1e-30))
    hists = [da.histogram(db[i], bins=n_bins, range=(lo, hi))[0] for i in range(sv.shape[0])]
    tasks["hist"] = da.stack(hists)
    log.info("computing sv statistics over %s with %d threads", sv.shape, n_workers)
    with dask.config.set(scheduler="threads", num_workers=n_workers):
        (results,) = dask.compute(tasks)
    results["hist_edges"] = np.linspace(lo, hi, n_bins + 1)
    results["n_pings"] = sv.shape[1]
    results["n_range"] = sv.shape[2]
    return results
