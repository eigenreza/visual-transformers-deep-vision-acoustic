"""Detections along one camera transect: objects per species over time with the camera depth.

The input is the ten-second series written by the camera series stage
(``series_10s.parquet`` under ``<project_dir>/runs/camera_series/<experiment>_seed<seed>/
<transect>/<camera>/``). Resolved objects at the operating thresholds, after the removal of
fixed structure, are summed per two minutes and drawn on a symmetric log scale; the cleaned
camera depth is drawn on the right axis.

Usage: python scripts/make_transect_series_figure.py <series_10s.parquet> [--transect <name>]
       [--classes krill,herring,...] [--minutes 2] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
from figure_style import plt  # noqa: E402

DEFAULT_CLASSES = ["krill", "herring", "benthosema", "maurolicus", "bluewhiting", "gonatus"]


def count_column(series: pd.DataFrame, cls: str) -> str:
    """Resolved objects at the operating thresholds when present, otherwise the unresolved counts."""
    for suffix in ("_res", "_op"):
        if f"{cls}{suffix}" in series.columns:
            return f"{cls}{suffix}"
    raise SystemExit(f"no count column for {cls} in the series")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("series", type=Path)
    ap.add_argument("--transect", default=None, help="name in the title; the parent directory by default")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--minutes", type=float, default=2.0, help="length of the summed bins in minutes")
    ap.add_argument("--out", default=Path("docs/figures"), type=Path)
    args = ap.parse_args()
    fs.apply_style()

    series = pd.read_parquet(args.series)
    if "time" in series.columns:
        series = series.set_index("time")
    series.index = pd.to_datetime(series.index)
    transect = args.transect or args.series.resolve().parents[1].name
    classes = [c for c in args.classes.split(",") if c]
    rule = f"{int(round(args.minutes * 60))}s"
    counts = series[[count_column(series, c) for c in classes]].fillna(0).resample(rule).sum()
    counts.columns = classes
    depth = series["depth_mean"].resample(rule).mean()
    start = series.index.min()
    minutes = (counts.index - start).total_seconds() / 60.0
    depth_minutes = (depth.index - start).total_seconds() / 60.0

    fig, ax = plt.subplots(figsize=(fs.WIDTH, fs.WIDTH * 0.48))
    for cls in classes:
        ax.plot(minutes, counts[cls], color=fs.COLOURS[cls], label=fs.label(cls), linewidth=1.1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(0, max(10.0, float(counts.to_numpy().max()) * 1.6))
    ax.set_xlim(0, float(minutes.max()) + args.minutes)
    ax.set_xlabel(f"minutes since the start of the recording ({start:%Y-%m-%d %H:%M} UTC)")
    ax.set_ylabel(f"objects per {args.minutes:g} min at the operating thresholds")
    ax.grid(True, axis="y", color="0.92", linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(depth_minutes, depth, color="0.25", linewidth=1.0, linestyle=(0, (4, 2)), label="camera depth")
    ax2.set_ylim(max(float(np.nanmax(depth)) * 1.08, 10.0), 0)
    ax2.set_ylabel("camera depth (m)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + h2, labels + l2, loc="upper center", ncol=len(classes) + 1, bbox_to_anchor=(0.5, 1.12)
    )
    ax.set_title(f"transect {transect}", loc="left", fontsize=8, color="0.3", pad=22)
    fs.save(fig, args.out, "transect_series")


if __name__ == "__main__":
    main()
