"""Aggregation of preprocessed backscatter to coarser cells.

Cells average the noise-corrected linear backscatter over the samples that
carry no hard mask bit (missing, surface, seabed, impulsive). Samples below
the signal-to-noise threshold keep their corrected value, which is zero or
close to it, so that empty water averages to the floor rather than being left
out. Each cell also records the fraction of its samples with no mask bit at
all and the hard bits that apply to at least half of its samples.
"""

from __future__ import annotations

import numpy as np

from .preprocess import HARD_BITS, MASK_BITS, to_db


def aggregate_cells(
    db: np.ndarray,
    mask: np.ndarray,
    ping_factor: int,
    range_factor: int,
    floor_db: float = -130.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean backscatter per cell in dB, the valid fraction (0 to 255) and the majority hard bits.

    ``db`` and ``mask`` have shape (F, P, R); the trailing pings and samples
    that do not fill a cell are dropped. Cells without any usable sample are
    NaN with valid fraction zero.
    """
    n_f, n_p, n_r = db.shape
    n_t, n_d = n_p // ping_factor, n_r // range_factor
    shape = (n_f, n_t, ping_factor, n_d, range_factor)
    cut = db[:, : n_t * ping_factor, : n_d * range_factor].astype(np.float32).reshape(shape)
    mcut = mask[:, : n_t * ping_factor, : n_d * range_factor].reshape(shape)
    with np.errstate(all="ignore"):
        lin = 10.0 ** (cut / 10.0)
    usable = ((mcut & HARD_BITS) == 0) & np.isfinite(lin)
    lin = np.where(usable, lin, 0.0)
    count = usable.sum(axis=(2, 4))
    total = lin.sum(axis=(2, 4))
    with np.errstate(all="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    out_db = to_db(mean, floor_db).astype(np.float16)
    clean = (mcut == 0).mean(axis=(2, 4))
    valid_fraction = np.round(clean * 255).astype(np.uint8)
    majority = np.zeros((n_f, n_t, n_d), dtype=np.uint8)
    half = ping_factor * range_factor / 2.0
    for bit in MASK_BITS.values():
        if bit & HARD_BITS:
            votes = ((mcut & bit) > 0).sum(axis=(2, 4))
            majority[votes >= half] |= bit
    return out_db, valid_fraction, majority


def aggregate_time_cells(
    db: np.ndarray,
    mask: np.ndarray,
    cell_of_ping: np.ndarray,
    n_time: int,
    range_factor: int,
    floor_db: float = -130.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells of a fixed duration, which hold a varying number of pings.

    ``db`` and ``mask`` have shape (F, P, R); ``cell_of_ping`` (P,) gives the
    time cell of every ping, non-decreasing and in [0, ``n_time``). Range
    samples are grouped ``range_factor`` at a time. Returns the mean
    backscatter in dB, the valid fraction (0 to 255) and the majority hard
    bits as ``aggregate_cells`` does, with shape (F, n_time, R // range_factor);
    a time cell without pings is missing.
    """
    n_f, n_p, n_r = db.shape
    n_d = n_r // range_factor
    cut = db[:, :, : n_d * range_factor].astype(np.float32).reshape(n_f, n_p, n_d, range_factor)
    mcut = mask[:, :, : n_d * range_factor].reshape(n_f, n_p, n_d, range_factor)
    with np.errstate(all="ignore"):
        lin = 10.0 ** (cut.astype(np.float64) / 10.0)
    usable = ((mcut & HARD_BITS) == 0) & np.isfinite(lin)
    lin = np.where(usable, lin, 0.0)
    edges = np.searchsorted(np.asarray(cell_of_ping), np.arange(n_time + 1), "left")

    def by_cell(per_ping: np.ndarray) -> np.ndarray:
        # Sums over the pings of every time cell from differences of the cumulative sum,
        # which gives zero for a cell without pings.
        cum = np.concatenate(
            [np.zeros((n_f, 1, n_d)), np.cumsum(per_ping.astype(np.float64), axis=1)], axis=1
        )
        return cum[:, edges[1:]] - cum[:, edges[:-1]]

    total = by_cell(lin.sum(axis=3))
    count = by_cell(usable.sum(axis=3))
    clean = by_cell((mcut == 0).sum(axis=3))
    samples = ((edges[1:] - edges[:-1]) * range_factor).astype(np.float64)[None, :, None]
    with np.errstate(all="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
        fraction = np.where(samples > 0, clean / np.maximum(samples, 1.0), 0.0)
    out_db = to_db(mean, floor_db).astype(np.float16)
    valid_fraction = np.round(np.clip(fraction, 0.0, 1.0) * 255).astype(np.uint8)
    majority = np.zeros((n_f, n_time, n_d), dtype=np.uint8)
    for bit in MASK_BITS.values():
        if bit & HARD_BITS:
            votes = by_cell(((mcut & bit) > 0).sum(axis=3))
            majority[(votes >= samples / 2.0) & (samples > 0)] |= bit
    majority[np.broadcast_to(samples == 0, majority.shape)] |= MASK_BITS["missing"]
    return out_db, valid_fraction, majority
