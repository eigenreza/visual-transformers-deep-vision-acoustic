"""Window geometry and patch extraction from a preprocessed transect store.

A window is a span of vessel time and a band of depth. Its patch holds, for
every channel, the mean corrected backscatter per cell over the hard-valid
samples, the share of valid samples per cell, and the absolute depth of every
cell row. Two anchors are supported: a band centred on the camera depth, and
a surface-anchored band whose top never rises above the surface exclusion
depth of the primary channel, so that shallow windows carry the same number
of usable cells as deep ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..acoustics.downsample import aggregate_time_cells


@dataclass(frozen=True)
class Geometry:
    """Window geometry in seconds and metres."""

    name: str
    length_s: float
    height_m: float
    stride_s: float
    anchor: str = "camera"
    context: dict[str, float] | None = None
    # Camera span of the targets; None means the window length. A window whose
    # acoustic span covers the whole lag support uses a short target span and
    # fixed targets, since every candidate vessel time is already inside it.
    target_span_s: float | None = None
    # Lag treatment of a short target span: ``fixed`` (no offsets) or
    # ``distribution`` (offsets and weights of the mapping's lag distribution,
    # as for a window whose targets span its whole length).
    target_lag: str = "fixed"

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> Geometry:
        return cls(
            name=name,
            length_s=float(cfg["length_s"]),
            height_m=float(cfg["height_m"]),
            stride_s=float(cfg["stride_s"]),
            anchor=str(cfg.get("anchor", "camera")),
            context=dict(cfg["context"]) if cfg.get("context") else None,
            target_span_s=float(cfg["target_span_s"]) if cfg.get("target_span_s") else None,
            target_lag=str(cfg.get("target_lag", "fixed")),
        )


def band_top(camera_depth_m: float, height_m: float, anchor: str, exclusion_depth_m: float) -> float:
    """Depth of the top of the band for a window whose camera sits at ``camera_depth_m``."""
    top = camera_depth_m - height_m / 2.0
    if anchor == "surface":
        return max(top, exclusion_depth_m)
    return top


class TransectPatches:
    """Patches cut from one transect store held in memory.

    ``sv_db`` has shape (F, P, R) in dB (float16 is accepted), ``mask`` the
    bit field of the same shape, ``ping_time`` shape (P,) and ``depth`` shape
    (R,), in metres below the surface.
    """

    def __init__(
        self,
        sv_db: np.ndarray,
        mask: np.ndarray,
        ping_time: np.ndarray,
        depth: np.ndarray,
        time_step_s: float,
        depth_step_m: float,
        floor_db: float = -130.0,
    ) -> None:
        self.sv_db = sv_db
        self.mask = mask
        self.t_ns = ping_time.astype("datetime64[ns]").astype(np.int64)
        self.depth = np.asarray(depth, dtype=float)
        self.time_step_s = float(time_step_s)
        self.depth_step_m = float(depth_step_m)
        self.floor_db = floor_db
        self.range_step_m = float(np.median(np.diff(self.depth)))
        self.ping_interval_s = float(np.median(np.diff(self.t_ns)) / 1e9)
        self.range_factor = max(1, int(round(self.depth_step_m / self.range_step_m)))
        self.ping_factor = max(1, int(round(self.time_step_s / self.ping_interval_s)))

    @property
    def n_pings(self) -> int:
        return int(self.t_ns.size)

    def ping_slice(self, start: np.datetime64, end: np.datetime64) -> tuple[int, int]:
        """Index range of the pings in [start, end)."""
        s = np.datetime64(start, "ns").astype(np.int64)
        e = np.datetime64(end, "ns").astype(np.int64)
        return int(np.searchsorted(self.t_ns, s, "left")), int(np.searchsorted(self.t_ns, e, "left"))

    def depth_slice(self, top_m: float, height_m: float) -> tuple[int, int]:
        """Index range of the range samples in [top_m, top_m + height_m)."""
        i0 = int(np.searchsorted(self.depth, top_m, "left"))
        i1 = int(np.searchsorted(self.depth, top_m + height_m, "left"))
        return i0, i1

    def patch(
        self,
        start: np.datetime64,
        end: np.datetime64,
        top_m: float,
        height_m: float,
        n_time: int,
        n_depth: int,
    ) -> dict[str, Any] | None:
        """One patch as cells of (time_step_s, depth_step_m); None when the window has no pings.

        Time cell ``k`` holds the pings from ``start + k * time_step_s`` to
        the next step, whatever the ping rate, so that the patch spans the
        whole window and a cell has the same duration in every transect; a
        cell without pings is missing. In depth the patch is padded or cropped
        to exactly ``n_depth`` cells; padded cells are missing (valid fraction
        zero). Row 0 lies at ``top_m``: a band that starts above the first
        range sample is padded at the top by the missing depth, so that the
        rows keep their depth.
        """
        p0, p1 = self.ping_slice(start, end)
        r0, r1 = self.depth_slice(top_m, height_m)
        if p1 <= p0 or r1 <= r0:
            return None
        samples_needed = n_depth * self.range_factor
        lead = 0
        if top_m < self.depth[0]:
            # Whole samples between the top of the band and the first sample, rounded down, so
            # that every sample falls in the row that holds its depth.
            lead = min(samples_needed - 1, int(np.floor((self.depth[0] - top_m) / self.range_step_m + 1e-6)))
        r1 = min(r1, r0 + samples_needed - lead)
        sv = self.sv_db[:, p0:p1, r0:r1]
        m = self.mask[:, p0:p1, r0:r1]
        pad_r = samples_needed - lead - (r1 - r0)
        if pad_r or lead:
            widths = ((0, 0), (0, 0), (lead, pad_r))
            sv = np.pad(sv.astype(np.float32), widths, constant_values=np.nan)
            m = np.pad(m, widths, constant_values=1)
        start_ns = np.datetime64(start, "ns").astype(np.int64)
        cell = (self.t_ns[p0:p1] - start_ns) // int(round(self.time_step_s * 1e9))
        db, valid, hard = aggregate_time_cells(
            sv, m, np.clip(cell, 0, n_time - 1), n_time, self.range_factor, self.floor_db
        )
        depth_cells = top_m + (np.arange(n_depth) + 0.5) * self.depth_step_m
        return {
            "sv_db": db,
            "valid": valid,
            "hard": hard,
            "depth_cells": depth_cells.astype(np.float32),
            "n_pings": int(p1 - p0),
            "n_samples": int(r1 - r0),
        }


def window_grid(mapping: pd.DataFrame, stride_s: float, length_s: float) -> pd.DataFrame:
    """Window centres on the camera interval grid: one window every ``stride_s`` along the mapping.

    Each row keeps the camera time of the centre, the vessel time from the
    mapping (centre minus lag), the lag and its standard deviation, the
    camera depth averaged over the window's intervals, and the mapping flags
    averaged over the same intervals.
    """
    m = mapping.sort_values("time_camera").reset_index(drop=True)
    t_cam = m["time_camera"].to_numpy().astype("datetime64[ns]")
    interval_s = (
        float(np.median(np.diff(t_cam).astype("timedelta64[ms]").astype(float)) / 1000.0)
        if len(m) > 1
        else 10.0
    )
    step = max(1, int(round(stride_s / interval_s)))
    half = int(round(length_s / 2.0 / interval_s))
    flag_cols = [
        c
        for c in m.columns
        if c.startswith(("camera_in_exclusion_", "band_usable_fraction_")) or c == "camera_surface"
    ]
    rows = []
    for i in range(half, len(m) - half, step):
        block = m.iloc[i - half : i + half] if half > 0 else m.iloc[i : i + 1]
        row = {
            "time_camera": t_cam[i],
            "time_vessel": np.datetime64(m.loc[i, "time_vessel"], "ns"),
            "lag_s": float(m.loc[i, "lag_s"]),
            "lag_sd_s": float(m.loc[i, "lag_sd_s"]),
            "lag_lo_s": float(m.loc[i, "lag_lo_s"]) if "lag_lo_s" in m else np.nan,
            "lag_hi_s": float(m.loc[i, "lag_hi_s"]) if "lag_hi_s" in m else np.nan,
            "lag_distribution": str(m.loc[i, "lag_distribution"]) if "lag_distribution" in m else "normal",
            "camera_depth_m": float(block["camera_depth_m"].mean())
            if block["camera_depth_m"].notna().any()
            else np.nan,
            "n_intervals": int(len(block)),
            "n_intervals_with_depth": int(block["camera_depth_m"].notna().sum()),
        }
        for c in flag_cols:
            row[c] = float(block[c].astype(float).mean()) if block[c].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
