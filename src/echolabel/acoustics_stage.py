"""Stage 5 commands: exploration of the backscatter around the camera transects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .acoustics.explore import explore_window
from .acoustics.store import open_sv_store
from .camera import clean_depth
from .config import resolve_path
from .data.transects import parse_deepvision_xml
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def _camera_depth(cfg: dict[str, Any], transect: str) -> pd.DataFrame | None:
    path = resolve_path(cfg, "source_data", transect, "deepvision.xml")
    if not path.is_file():
        return None
    meta, _ = parse_deepvision_xml(path)
    meta = meta.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    cleaned, _ = clean_depth(meta["timestamp"], meta["depth_num"])
    return cleaned


def _figure(
    coarse: np.ndarray,
    time_edges: np.ndarray,
    depth_edges: np.ndarray,
    freqs: np.ndarray,
    show_hz: list[float],
    camera: pd.DataFrame | None,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    panels = [f for f in show_hz if bool(np.any(np.abs(freqs - f) < 1))]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.6 * len(panels)), sharex=True, squeeze=False)
    for ax, f in zip(axes[:, 0], panels, strict=True):
        i = int(np.argmin(np.abs(freqs - f)))
        img = coarse[i].T
        mesh = ax.pcolormesh(
            mdates.date2num(time_edges.astype("datetime64[us]").astype(object)),
            depth_edges,
            img,
            vmin=-90,
            vmax=-50,
            cmap="viridis",
            shading="flat",
        )
        ax.invert_yaxis()
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{title}, {int(round(f / 1000))} kHz, mean sv per 10 s by 2 m")
        fig.colorbar(mesh, ax=ax, label="dB re 1 m$^{-1}$")
        if camera is not None and not camera.empty:
            cam = camera.dropna(subset=["depth_clean"])
            ax.plot(
                mdates.date2num(cam["timestamp"].dt.to_pydatetime()),
                cam["depth_clean"],
                color="white",
                lw=1.0,
                label="camera depth (cleaned)",
            )
            ax.legend(loc="lower right", fontsize=8)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1, 0].set_xlabel("time (UTC)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_audit_transect_acoustics(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Summarise the backscatter in every transect window and draw coarse echograms."""
    store = resolve_path(cfg, "source_data", cfg.get("store", "S2018105_sv.zarr"))
    windows = {k: (str(v[0]), str(v[1])) for k, v in cfg.get("transect_windows", {}).items()}
    inputs: dict[str, str | Path] = {"store": store}
    if not force and manifest_is_current(out_dir, cfg, inputs):
        log.info("acoustic exploration is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ecfg = cfg.get("exploration", {})
    ds = open_sv_store(store)
    outputs: list[str | Path] = []
    summaries: dict[str, Any] = {}
    profiles = []
    for transect, (start, end) in windows.items():
        log.info("exploring %s from %s to %s", transect, start, end)
        result = explore_window(
            ds,
            start,
            end,
            draft_m=float(cfg.get("draft_m", 8.5)),
            floor_db=float(cfg.get("floor_db", -130)),
            time_step_s=float(ecfg.get("coarse_time_step_s", 10)),
            depth_step_m=float(ecfg.get("coarse_depth_step_m", 2)),
            reference_hz=float(cfg.get("reference_hz", 38000)),
        )
        summaries[transect] = result["summary"]
        prof = result["profile"]
        prof.insert(0, "transect", transect)
        profiles.append(prof)
        result["bottom"].to_parquet(out_dir / f"{transect}_bottom.parquet", index=False)
        np.savez_compressed(
            out_dir / f"{transect}_coarse.npz",
            db=result["coarse"],
            time_edges=result["time_edges"].astype("datetime64[ns]").astype(np.int64),
            depth_edges=result["depth_edges"],
            frequencies_hz=result["freqs"],
        )
        outputs += [out_dir / f"{transect}_bottom.parquet", out_dir / f"{transect}_coarse.npz"]
        camera = _camera_depth(cfg, transect)
        fig_path = out_dir / f"{transect}_echogram.png"
        try:
            _figure(
                result["coarse"],
                result["time_edges"],
                result["depth_edges"],
                result["freqs"],
                [float(f) for f in ecfg.get("figure_frequencies_hz", [38000])],
                camera,
                transect,
                fig_path,
            )
            outputs.append(fig_path)
        except Exception as exc:  # pragma: no cover
            log.warning("figure for %s failed: %s", transect, exc)
    profile_table = pd.concat(profiles, ignore_index=True)
    profile_table.to_csv(out_dir / "range_profiles.csv", index=False)
    outputs.append(out_dir / "range_profiles.csv")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "audit-transect-acoustics", cfg, inputs, outputs)
    return out_dir
