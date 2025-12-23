"""Leakage-free splits of the window dataset.

Two protocols. Leave-one-transect-out: one transect is the test set, the
last share of every other transect is the validation block and the rest is
training. Within-transect blocks: contiguous blocks of fixed length are
assigned to training, validation and test in a repeating pattern. In both
protocols every boundary between different splits is surrounded by a buffer
on both sides, windows whose span touches a buffer are discarded, and no
window straddles a boundary. A checker asserts these properties on the
resulting assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _spans(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of every window in seconds of vessel time."""
    t = meta["time_vessel"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
    half = meta["length_s"].to_numpy(dtype=float) / 2.0
    return t - half, t + half


def buffer_by_window(meta: pd.DataFrame, lag_sd_multiple: float) -> pd.Series:
    """Buffer per window: its length plus the lag uncertainty of its targets, indexed like ``meta``.

    The uncertainty is ``lag_sd_multiple`` standard deviations under a normal
    lag distribution, the window's own support width under a uniform one, and
    nothing for fixed targets, whose window already spans the support.
    """
    length = meta["length_s"].to_numpy(dtype=float)
    by_sd = lag_sd_multiple * meta["lag_sd_s"].to_numpy(dtype=float)
    kind = (
        meta["target_distribution"].astype(str).to_numpy()
        if "target_distribution" in meta
        else np.full(len(meta), "normal")
    )
    width = (
        (meta["lag_hi_s"] - meta["lag_lo_s"]).to_numpy(dtype=float)
        if {"lag_lo_s", "lag_hi_s"} <= set(meta.columns)
        else np.full(len(meta), np.nan)
    )
    lag_part = np.where(kind == "fixed", 0.0, np.where(kind == "uniform", width, by_sd))
    lag_part = np.where(np.isfinite(lag_part), lag_part, by_sd)
    return pd.Series(length + lag_part, index=meta.index, name="buffer_s")


def buffer_seconds(meta: pd.DataFrame, lag_sd_multiple: float) -> float:
    """Median buffer of the windows, used to summarise and to decide whether a protocol can run."""
    return float(np.median(buffer_by_window(meta, lag_sd_multiple).to_numpy())) if len(meta) else 0.0


def buffer_by_transect(meta: pd.DataFrame, lag_sd_multiple: float) -> dict[str, float]:
    """Median buffer per transect."""
    return {str(t): buffer_seconds(sub, lag_sd_multiple) for t, sub in meta.groupby("transect", sort=True)}


def _buffer_rows(buffer_s: float | dict[str, float] | pd.Series, transect: str, rows: pd.Index) -> np.ndarray:
    """Buffer per row of a transect's windows from a scalar, a per-transect dict or a per-window series."""
    if isinstance(buffer_s, pd.Series):
        return buffer_s.loc[rows].to_numpy(dtype=float)
    value = float(buffer_s[str(transect)]) if isinstance(buffer_s, dict) else float(buffer_s)
    return np.full(len(rows), value)


def _assign_blocks(
    meta: pd.DataFrame, boundaries: np.ndarray, labels: list[str], buffer_s: float | np.ndarray
) -> pd.Series:
    """Split label per window from block boundaries (sorted seconds) and the label of each block.

    A window is dropped (label ``None``) when its span crosses a boundary or
    comes within its buffer (``buffer_s``, one value or one per window) of a
    boundary between blocks of different labels.
    """
    start, end = _spans(meta)
    buffers = np.broadcast_to(np.asarray(buffer_s, dtype=float), (len(meta),))
    out = np.array([None] * len(meta), dtype=object)
    for i in range(len(meta)):
        k = int(np.searchsorted(boundaries, start[i], "right"))
        if k == 0 or k > len(labels):
            continue
        lo = boundaries[k - 1]
        hi = boundaries[k] if k < len(boundaries) else np.inf
        if end[i] > hi:
            continue  # straddles the next boundary
        label = labels[k - 1]
        near_lo = k >= 2 and labels[k - 2] != label and start[i] < lo + buffers[i]
        near_hi = k < len(labels) and labels[k] != label and end[i] > hi - buffers[i]
        out[i] = None if (near_lo or near_hi) else label
    return pd.Series(out, index=meta.index)


def loto_folds(
    meta: pd.DataFrame,
    val_fraction: float,
    buffer_s: float | dict[str, float] | pd.Series,
    val_position: str = "end",
) -> pd.DataFrame:
    """Leave-one-transect-out assignment: one row per window and fold with its split or ``None``.

    ``buffer_s`` is one length, a length per transect, or a series with one
    length per window indexed like ``meta``.
    The validation block of a training transect holds ``val_fraction`` of its
    span, centred on the middle of the transect (``val_position`` middle) or
    at its end; the ends of a tow are spent at the surface, so the middle
    keeps usable validation windows.
    """
    rows = []
    transects = sorted(meta["transect"].unique())
    start, end = _spans(meta)
    for test_transect in transects:
        split = pd.Series([None] * len(meta), index=meta.index, dtype=object)
        split[meta["transect"] == test_transect] = "test"
        for transect in transects:
            if transect == test_transect:
                continue
            sel = (meta["transect"] == transect).to_numpy()
            t0, t1 = start[sel].min(), end[sel].max()
            sub = meta[sel]
            if val_position == "middle":
                mid, half = (t0 + t1) / 2.0, val_fraction * (t1 - t0) / 2.0
                boundaries = np.array([t0, mid - half, mid + half, t1 + 1.0])
                block_labels = ["train", "val", "train"]
            else:
                boundaries = np.array([t0, t1 - val_fraction * (t1 - t0), t1 + 1.0])
                block_labels = ["train", "val"]
            labels = _assign_blocks(
                sub, boundaries, block_labels, _buffer_rows(buffer_s, transect, sub.index)
            )
            split.loc[sub.index] = labels
        rows.append(
            pd.DataFrame({"window_id": meta["window_id"], "fold": f"test_{test_transect}", "split": split})
        )
    return pd.concat(rows, ignore_index=True)


def block_splits(
    meta: pd.DataFrame,
    block_s: float,
    pattern: list[str],
    buffer_s: float | dict[str, float] | pd.Series,
) -> pd.DataFrame:
    """Within-transect assignment by contiguous blocks in a repeating pattern, buffered at label changes."""
    start, end = _spans(meta)
    split = pd.Series([None] * len(meta), index=meta.index, dtype=object)
    for transect in sorted(meta["transect"].unique()):
        sel = (meta["transect"] == transect).to_numpy()
        t0, t1 = start[sel].min(), end[sel].max()
        n_blocks = int(np.ceil((t1 - t0) / block_s))
        boundaries = t0 + np.arange(n_blocks + 1) * block_s
        labels = [pattern[i % len(pattern)] for i in range(n_blocks)]
        sub = meta[sel]
        split.loc[sub.index] = _assign_blocks(
            sub, boundaries, labels, _buffer_rows(buffer_s, transect, sub.index)
        )
    return pd.DataFrame({"window_id": meta["window_id"], "fold": "within_transect", "split": split})


def check_splits(
    meta: pd.DataFrame, assignment: pd.DataFrame, buffer_s: float | dict[str, float] | pd.Series
) -> pd.DataFrame:
    """Violations of the leakage rules per fold: overlapping or too close windows of different splits.

    Within each transect and fold, every pair of windows in different
    splits must be separated by at least ``buffer_s`` between the end of one
    and the start of the other. Returns one row per fold and transect with
    the smallest separation found and the number of violating pairs; an empty
    table means no violation.
    """
    start, end = _spans(meta)
    by_id = meta.set_index("window_id")
    rows = []
    for fold, a in assignment.groupby("fold"):
        a = a.dropna(subset=["split"])
        for transect in sorted(meta["transect"].unique()):
            ids = a[a["window_id"].map(by_id["transect"]) == transect]
            if ids.empty:
                continue
            pos = by_id.index.get_indexer(ids["window_id"])
            s, e = start[pos], end[pos]
            labels = ids["split"].to_numpy()
            buf = _buffer_rows(buffer_s, transect, meta.index[pos])
            min_gap, violations = np.inf, 0
            for u in SPLITS:
                for v in SPLITS:
                    if u >= v:
                        continue
                    su, eu = s[labels == u], e[labels == u]
                    sv, ev = s[labels == v], e[labels == v]
                    if su.size == 0 or sv.size == 0:
                        continue
                    gap = np.maximum(sv[None, :] - eu[:, None], su[:, None] - ev[None, :])
                    need = np.maximum(buf[labels == u][:, None], buf[labels == v][None, :])
                    min_gap = min(min_gap, float(gap.min()))
                    violations += int((gap < need).sum())
            if violations:
                rows.append(
                    {"fold": fold, "transect": transect, "min_gap_s": min_gap, "violating_pairs": violations}
                )
    return pd.DataFrame(rows, columns=["fold", "transect", "min_gap_s", "violating_pairs"])


def split_counts(assignment: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Number of windows per protocol, fold, split and transect, dropped windows included as ``dropped``."""
    a = assignment.merge(meta[["window_id", "transect"]], on="window_id")
    a["split"] = a["split"].fillna("dropped")
    keys = (["protocol"] if "protocol" in a.columns else []) + ["fold", "transect", "split"]
    return a.groupby(keys).size().rename("n_windows").reset_index()
