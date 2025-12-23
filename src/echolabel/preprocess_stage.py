"""Stage 5 commands: preprocessing of the transect windows and of the whole cruise.

The transect command reads each camera transect window at full resolution,
applies the masks and the noise correction, and writes one store per transect
to the scratch directory together with quality-control figures and a summary
in the project directory. The cruise command applies the same processing to
every ping of the cruise, chunk by chunk in parallel, and writes a downsampled
store for self-supervised pretraining.
"""

from __future__ import annotations

import json
import logging
import math
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .acoustics.downsample import aggregate_cells
from .acoustics.preprocess import (
    HARD_BITS,
    MASK_BITS,
    MASK_LOW_SNR,
    block_summary,
    detect_bottom,
    noise_sv_db,
    preprocess_block,
    to_db,
)
from .acoustics.store import frequency_position, open_sv_store
from .acoustics_stage import _camera_depth
from .config import resolve_path
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)

AUX_1D = ("latitude", "longitude", "distance", "heave", "heading", "pitch", "roll")


def _params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(cfg.get("preprocessing", {}))
    params.setdefault("floor_db", float(cfg.get("floor_db", -130.0)))
    return params


def _ping_bounds(ds: xr.Dataset, start: str, end: str) -> tuple[int, int]:
    """Index range of the pings between two times (inclusive)."""
    t = ds["ping_time"].values
    i0 = int(np.searchsorted(t, np.datetime64(start), side="left"))
    i1 = int(np.searchsorted(t, np.datetime64(end), side="right"))
    return i0, i1


def _load_pings(ds: xr.Dataset, i0: int, i1: int) -> dict[str, Any]:
    """Load one ping range at all frequencies with the auxiliary series."""
    sub = ds.isel(ping_time=slice(i0, i1))
    out: dict[str, Any] = {
        "sv": sub["sv"].transpose("frequency", "ping_time", "range").values.astype(np.float32),
        "ping_time": sub["ping_time"].values,
        "pulse_length_s": None,
        "transducer_draft_m": None,
    }
    if "pulse_length" in sub:
        out["pulse_length_s"] = (
            sub["pulse_length"].transpose("ping_time", "frequency").values.astype(np.float32)
        )
    if "transducer_draft" in sub:
        out["transducer_draft_m"] = sub["transducer_draft"].transpose("frequency", "ping_time").values
    for name in AUX_1D:
        if name in sub:
            out[name] = sub[name].values
    return out


def _draft(block: dict[str, Any], default: float) -> tuple[float, bool]:
    """Median transducer draft in the block and whether it is constant across channels and pings."""
    draft = block.get("transducer_draft_m")
    if draft is None or not np.isfinite(draft).any():
        return float(default), True
    values = np.unique(np.round(draft[np.isfinite(draft)], 3))
    return float(np.nanmedian(draft)), bool(values.size == 1)


def _transect_dataset(
    result: dict[str, Any],
    block: dict[str, Any],
    rng: np.ndarray,
    freqs: np.ndarray,
    draft_m: float,
    params: dict[str, Any],
) -> xr.Dataset:
    dims3 = ("frequency", "ping_time", "range")
    data: dict[str, Any] = {
        "sv_db": (dims3, result["db"]),
        "mask": (dims3, result["mask"]),
        "noise_db": (("frequency", "ping_time"), result["noise_db"]),
        "noise_raw_db": (("frequency", "ping_time"), result["noise_raw_db"]),
        "impulsive_fraction": (("frequency", "ping_time"), result["impulsive_fraction"]),
        "bottom_range_m": (("ping_time",), result["bottom_range_m"]),
        "absorption_db_per_m": (("frequency",), result["absorption_db_per_m"].astype(np.float32)),
        "surface_exclusion_m": (("frequency",), result["surface_exclusion_m"].astype(np.float32)),
    }
    if block.get("pulse_length_s") is not None:
        data["pulse_length_s"] = (("ping_time", "frequency"), block["pulse_length_s"])
    if block.get("transducer_draft_m") is not None:
        data["transducer_draft_m"] = (
            ("frequency", "ping_time"),
            block["transducer_draft_m"].astype(np.float32),
        )
    for name in AUX_1D:
        if name in block:
            data[name] = (("ping_time",), block[name])
    coords = {
        "frequency": freqs,
        "ping_time": block["ping_time"],
        "range": rng,
        "depth": ("range", rng + draft_m),
    }
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(
        {
            "description": "Noise-corrected volume backscattering strength with per-sample mask bits.",
            "mask_bits": json.dumps(MASK_BITS),
            "draft_m": draft_m,
            "floor_db": float(params.get("floor_db", -130.0)),
            "sv_db_units": "dB re 1 m-1",
            "depth_definition": "range from the transducer plus the transducer draft",
        }
    )
    return ds


def _write_store(ds: xr.Dataset, path: Path, ping_chunk: int = 1024) -> None:
    n_r = ds.sizes["range"]
    n_f = ds.sizes["frequency"]
    encoding: dict[str, dict[str, Any]] = {
        "sv_db": {"chunks": (n_f, ping_chunk, n_r)},
        "mask": {"chunks": (n_f, ping_chunk, n_r), "_FillValue": None},
    }
    ds.to_zarr(str(path), mode="w", zarr_format=2, encoding=encoding, consolidated=True)


def _cell_edges(values: np.ndarray, factor: int, n_cells: int) -> np.ndarray:
    """Edges of ``n_cells`` cells of ``factor`` elements each, closed by the last element."""
    edges = values[: n_cells * factor : factor]
    last = values[min(n_cells * factor, len(values) - 1)]
    return np.append(edges, last)


def _processed_figure(
    result: dict[str, Any],
    block: dict[str, Any],
    rng: np.ndarray,
    freqs: np.ndarray,
    draft_m: float,
    fig_cfg: dict[str, Any],
    camera: pd.DataFrame | None,
    title: str,
    path: Path,
    floor_db: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ping_time = block["ping_time"]
    interval = float(np.median(np.diff(ping_time).astype("timedelta64[ms]").astype(float))) / 1000.0
    ping_factor = max(1, int(round(float(fig_cfg.get("time_step_s", 10)) / max(interval, 1e-3))))
    range_factor = max(1, int(round(float(fig_cfg.get("depth_step_m", 2)) / float(np.median(np.diff(rng))))))
    db, valid, _ = aggregate_cells(result["db"], result["mask"], ping_factor, range_factor, floor_db)
    n_t, n_d = db.shape[1], db.shape[2]
    t_edges = _cell_edges(ping_time, ping_factor, n_t)
    d_edges = _cell_edges(rng, range_factor, n_d) + draft_m
    panels = [f for f in fig_cfg.get("frequencies_hz", [38000]) if np.any(np.abs(freqs - f) < 1)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.6 * len(panels)), sharex=True, squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.55")
    x = mdates.date2num(t_edges.astype("datetime64[us]").astype(object))
    for ax, f in zip(axes[:, 0], panels, strict=True):
        i = int(np.argmin(np.abs(freqs - f)))
        img = np.ma.masked_where(valid[i].T == 0, db[i].T.astype(np.float32))
        mesh = ax.pcolormesh(x, d_edges, img, vmin=-90, vmax=-50, cmap=cmap, shading="flat")
        ax.invert_yaxis()
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{title}, {int(round(f / 1000))} kHz after preprocessing (grey: masked)")
        fig.colorbar(mesh, ax=ax, label="dB re 1 m$^{-1}$")
        bottom = result["bottom_range_m"]
        if np.isfinite(bottom).any():
            ax.plot(
                mdates.date2num(ping_time.astype("datetime64[us]").astype(object)),
                bottom + draft_m,
                color="red",
                lw=0.8,
                label="seabed",
            )
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


def _noise_figure(
    raw_db: np.ndarray,
    result: dict[str, Any],
    rng: np.ndarray,
    freqs: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Observed quantiles of the raw level against range with the modelled noise curve per frequency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_f = len(freqs)
    fig, axes = plt.subplots(1, n_f, figsize=(3.2 * n_f, 4), sharey=True)
    for i, (ax, f) in enumerate(zip(np.atleast_1d(axes), freqs, strict=True)):
        with np.errstate(all="ignore"):
            q = np.nanquantile(raw_db[i], [0.1, 0.5], axis=0)
        ax.plot(q[0], rng, color="0.3", lw=1, label="10th percentile")
        ax.plot(q[1], rng, color="0.7", lw=1, label="median")
        nd = result["noise_db"][i]
        if np.isfinite(nd).any():
            model = noise_sv_db(np.array([np.nanmedian(nd)]), rng, float(result["absorption_db_per_m"][i]))[0]
            ax.plot(model, rng, color="tab:red", lw=1.2, label="noise model")
        ax.set_title(f"{int(round(float(f) / 1000))} kHz")
        ax.set_xlabel("dB re 1 m$^{-1}$")
        ax.set_xlim(-130, -40)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("range (m)")
    np.atleast_1d(axes)[0].invert_yaxis()
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def camera_depth_summary(camera: pd.DataFrame, exclusion_depths_m: dict[str, float]) -> dict[str, Any]:
    """Camera depth percentiles (raw non-zero and cleaned) and time spent above each exclusion depth."""
    raw = camera["depth_raw"].astype(float)
    nonzero = raw[raw > 0]
    clean = camera["depth_clean"].dropna().astype(float)
    quantiles = [0.05, 0.5, 0.95]

    def q(s: pd.Series) -> list[float]:
        return [round(float(v), 1) for v in s.quantile(quantiles)] if len(s) else []

    return {
        "n_frames": int(len(camera)),
        "dropout_fraction": float((raw == 0).mean()) if len(raw) else None,
        "quantiles": quantiles,
        "raw_nonzero_m": q(nonzero),
        "cleaned_m": q(clean),
        "fraction_above_exclusion": {
            name: {
                "exclusion_depth_m": float(d),
                "raw_nonzero": float((nonzero < d).mean()) if len(nonzero) else None,
                "cleaned": float((clean < d).mean()) if len(clean) else None,
            }
            for name, d in exclusion_depths_m.items()
        },
    }


def camera_band_fractions(
    mask: np.ndarray,
    ping_time: np.ndarray,
    rng: np.ndarray,
    draft_m: float,
    freqs: np.ndarray,
    camera: pd.DataFrame,
    half_width_m: float = 25.0,
    tolerance_s: float = 5.0,
    exclusion_m: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Mask statistics per channel inside the depth band around the camera.

    For every ping with a cleaned camera depth within ``tolerance_s``, the
    samples whose depth lies within ``half_width_m`` of the camera depth are
    counted: the hard-valid fraction has none of the missing, surface, seabed
    or impulsive bits; the valid fraction has no bit at all, which also
    requires a signal-to-noise ratio above the threshold. With the surface
    exclusion range per channel (``exclusion_m``), the fraction of pings whose
    camera depth lies inside the excluded zone and the fraction of pings
    whose band has no, or less than half, usable samples are added. ``mask``
    has shape (F, P, R).
    """
    cam = camera.dropna(subset=["depth_clean"]).sort_values("timestamp")
    if cam.empty:
        return None
    t_cam = cam["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    d_cam = cam["depth_clean"].to_numpy(dtype=float)
    t_ping = ping_time.astype("datetime64[ns]").astype(np.int64)
    pos = np.clip(np.searchsorted(t_cam, t_ping), 1, len(t_cam) - 1)
    left, right = pos - 1, pos
    nearest = np.where(np.abs(t_cam[left] - t_ping) <= np.abs(t_cam[right] - t_ping), left, right)
    close = np.abs(t_cam[nearest] - t_ping) <= tolerance_s * 1e9
    if not close.any():
        return None
    depth_cam = d_cam[nearest[close]]
    depth = rng + draft_m
    band = np.abs(depth[None, :] - depth_cam[:, None]) <= half_width_m
    n_band = int(band.sum())
    out: dict[str, Any] = {
        "band_half_width_m": half_width_m,
        "n_pings_with_camera_depth": int(close.sum()),
        "n_pings": int(len(ping_time)),
        "camera_depth_m_median": float(np.median(depth_cam)),
        "per_frequency": {},
    }
    band_count = band.sum(axis=1)
    for i, f in enumerate(freqs):
        m_pings = mask[i][close]
        m = m_pings[band]
        usable_per_ping = (((m_pings & HARD_BITS) == 0) & band).sum(axis=1)
        stats = {
            "hard_valid_fraction": float(((m & HARD_BITS) == 0).mean()) if n_band else None,
            "valid_fraction": float((m == 0).mean()) if n_band else None,
            "low_snr_fraction": float(((m & MASK_LOW_SNR) > 0).mean()) if n_band else None,
            "pings_without_usable_band_fraction": float((usable_per_ping == 0).mean()),
            "pings_band_half_usable_fraction": float((usable_per_ping >= 0.5 * band_count).mean()),
        }
        if exclusion_m is not None:
            excl_depth = float(exclusion_m[i]) + draft_m
            stats["exclusion_depth_m"] = excl_depth
            stats["camera_inside_exclusion_fraction"] = float((depth_cam < excl_depth).mean())
        out["per_frequency"][str(int(f))] = stats
    return out


def _bottom_agreement(
    sv_ref: np.ndarray, rng: np.ndarray, bottom_cfg: dict[str, Any], constrained: np.ndarray, floor_db: float
) -> dict[str, Any]:
    """Compare the seabed from the default search with the one from the transect-specific search."""
    kwargs = {k: v for k, v in bottom_cfg.items() if k not in ("reference_hz", "margin_m")}
    generic = detect_bottom(to_db(sv_ref, floor_db), rng, **kwargs)
    both = np.isfinite(generic) & np.isfinite(constrained)
    return {
        "generic_detected_fraction": float(np.isfinite(generic).mean()),
        "constrained_detected_fraction": float(np.isfinite(constrained).mean()),
        "both_detected_fraction": float(both.mean()),
        "agree_within_10m_fraction": float((np.abs(generic[both] - constrained[both]) <= 10).mean())
        if both.any()
        else None,
    }


def run_preprocess_transects(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Preprocess every transect window and export it to the scratch directory."""
    store = resolve_path(cfg, "source_data", cfg.get("store", "S2018105_sv.zarr"))
    scratch = resolve_path(cfg, "scratch_dir", "acoustics", "transects")
    windows = {k: (str(v[0]), str(v[1])) for k, v in cfg.get("transect_windows", {}).items()}
    inputs: dict[str, str | Path] = {"store": store}
    stores_exist = all((scratch / f"{t}.zarr").exists() for t in windows)
    if not force and stores_exist and manifest_is_current(out_dir, cfg, inputs):
        log.info("transect preprocessing is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    params = _params(cfg)
    tcfg = cfg.get("transects", {})
    overrides = tcfg.get("bottom", {}) or {}
    fig_cfg = tcfg.get("figure", {})
    ds = open_sv_store(store)
    freqs = ds["frequency"].values
    rng = ds["range"].values
    ref = frequency_position(ds, float(params.get("bottom", {}).get("reference_hz", 38000)))
    summaries: dict[str, Any] = {}
    outputs: list[str | Path] = []
    for transect, (start, end) in windows.items():
        t_start = time.time()
        i0, i1 = _ping_bounds(ds, start, end)
        log.info("preprocessing %s: pings %d to %d", transect, i0, i1)
        block = _load_pings(ds, i0, i1)
        draft_m, constant = _draft(block, float(cfg.get("draft_m", 8.5)))
        if not constant:
            log.warning(
                "%s: transducer draft is not constant in the window; median %.2f m used", transect, draft_m
            )
        override = overrides.get(transect)
        result = preprocess_block(block["sv"], rng, freqs, params, block["pulse_length_s"], override)
        summary = block_summary(result, freqs)
        summary.update(
            {
                "start": str(block["ping_time"][0]),
                "end": str(block["ping_time"][-1]),
                "ping_index": [i0, i1],
                "draft_m": draft_m,
                "draft_constant": constant,
                "bottom_override": override,
            }
        )
        if override not in (None, False) and params.get("bottom"):
            summary["bottom_agreement"] = _bottom_agreement(
                block["sv"][ref], rng, params["bottom"], result["bottom_range_m"], params["floor_db"]
            )
        zpath = scratch / f"{transect}.zarr"
        _write_store(_transect_dataset(result, block, rng, freqs, draft_m, params), zpath)
        summary["store"] = str(zpath)
        camera = _camera_depth(cfg, transect)
        if camera is not None:
            summary["camera_band"] = camera_band_fractions(
                result["mask"],
                block["ping_time"],
                rng,
                draft_m,
                freqs,
                camera,
                half_width_m=float(tcfg.get("camera_band_half_width_m", 25.0)),
                exclusion_m=result["surface_exclusion_m"],
            )
            exclusion_depths = {
                str(int(f)): float(result["surface_exclusion_m"][i]) + draft_m for i, f in enumerate(freqs)
            }
            summary["camera_depth"] = camera_depth_summary(camera, exclusion_depths)
        fig_path = out_dir / f"{transect}_processed.png"
        _processed_figure(
            result, block, rng, freqs, draft_m, fig_cfg, camera, transect, fig_path, params["floor_db"]
        )
        outputs.append(fig_path)
        noise_path = out_dir / f"{transect}_noise.png"
        _noise_figure(to_db(block["sv"], params["floor_db"]), result, rng, freqs, transect, noise_path)
        outputs.append(noise_path)
        summary["seconds"] = round(time.time() - t_start, 1)
        summaries[transect] = summary
        log.info("%s done in %.0f s", transect, summary["seconds"])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    outputs.append(out_dir / "summary.json")
    write_manifest(out_dir, "preprocess-transects", cfg, inputs, outputs)
    return out_dir


# Whole-cruise downsampling. Workers hold their open handles in module state.
_W: dict[str, Any] = {}


def _init_worker(store: str, out_path: str, params: dict[str, Any], settings: dict[str, Any]) -> None:
    import dask

    # Workers read their chunks in their own processes; dask threads are not needed
    # and a pool forked from a process that already used them can deadlock.
    dask.config.set(scheduler="synchronous")
    _W["ds"] = open_sv_store(store)
    _W["group"] = zarr.open_group(out_path, mode="r+")
    _W["params"] = params
    _W["settings"] = settings
    _W["rng"] = _W["ds"]["range"].values
    _W["freqs"] = _W["ds"]["frequency"].values


def _process_chunk(k: int) -> dict[str, Any]:
    t_start = time.time()
    s = _W["settings"]
    ds, g, params, rng, freqs = _W["ds"], _W["group"], _W["params"], _W["rng"], _W["freqs"]
    pf, rf, overlap = s["ping_factor"], s["range_factor"], s["overlap_pings"]
    i0 = s["i_start"] + k * s["chunk_pings"]
    i1 = min(i0 + s["chunk_pings"], s["i_end"])
    o0, o1 = max(i0 - overlap, 0), min(i1 + overlap, s["n_total"])
    block = _load_pings(ds, o0, o1)
    result = preprocess_block(block["sv"], rng, freqs, params, block["pulse_length_s"])
    a, b = i0 - o0, i0 - o0 + (i1 - i0)
    trimmed = {
        "mask": result["mask"][:, a:b],
        "bottom_range_m": result["bottom_range_m"][a:b],
        "noise_db": result["noise_db"][:, a:b],
        "noise_raw_db": result["noise_raw_db"][:, a:b],
        "absorption_db_per_m": result["absorption_db_per_m"],
    }
    db, valid, hard = aggregate_cells(result["db"][:, a:b], trimmed["mask"], pf, rf, params["floor_db"])
    n_t = db.shape[1]
    t0 = (i0 - s["i_start"]) // pf
    t1 = t0 + n_t
    g["sv_db"][:, t0:t1, :] = db
    g["valid_fraction"][:, t0:t1, :] = valid
    g["hard_mask"][:, t0:t1, :] = hard
    g["noise_db"][:, t0:t1] = trimmed["noise_db"][:, : n_t * pf].reshape(len(freqs), n_t, pf)[:, :, 0]
    g["bottom_range_m"][t0:t1] = trimmed["bottom_range_m"][: n_t * pf].reshape(n_t, pf)[:, 0]
    summary = block_summary(trimmed, freqs)
    record: dict[str, Any] = {
        "chunk": k,
        "i0": i0,
        "i1": i1,
        "t0": t0,
        "t1": t1,
        "start": str(block["ping_time"][a]),
        "seconds": round(time.time() - t_start, 1),
        "bottom_detected_fraction": summary["bottom_detected_fraction"],
        "bottom_range_m_median": summary["bottom_range_m_median"],
    }
    for f, stats in summary["per_frequency"].items():
        for name, value in stats.items():
            record[f"{name}_{f}"] = value
        if block["pulse_length_s"] is not None:
            col = block["pulse_length_s"][:, int(np.argmin(np.abs(freqs - float(f))))]
            record[f"pulse_ms_{f}"] = float(np.nanmax(col) * 1000) if np.isfinite(col).any() else None
    progress = Path(s["progress_dir"])
    with open(progress / f"{k}.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    return record


def _cruise_skeleton(
    ds: xr.Dataset, out_path: Path, settings: dict[str, Any], params: dict[str, Any], ref: int
) -> xr.Dataset:
    """Create the output store with coordinates and empty arrays, chunked like the source."""
    import dask.array as da

    pf, rf = settings["ping_factor"], settings["range_factor"]
    i_start, i_end = settings["i_start"], settings["i_end"]
    rng = ds["range"].values
    n_d = len(rng) // rf
    centres = rng[: n_d * rf].reshape(n_d, rf).mean(axis=1)
    n_t = settings["n_cells"]
    first = np.arange(n_t) * pf + i_start
    sub = ds.isel(ping_time=first)
    freqs = ds["frequency"].values
    n_f = len(freqs)
    t_chunk = settings["chunk_pings"] // pf
    dims3 = ("frequency", "time", "range")
    data: dict[str, Any] = {
        "sv_db": (dims3, da.full((n_f, n_t, n_d), np.nan, dtype=np.float16, chunks=(n_f, t_chunk, n_d))),
        "valid_fraction": (dims3, da.zeros((n_f, n_t, n_d), dtype=np.uint8, chunks=(n_f, t_chunk, n_d))),
        "hard_mask": (dims3, da.zeros((n_f, n_t, n_d), dtype=np.uint8, chunks=(n_f, t_chunk, n_d))),
        "noise_db": (
            ("frequency", "time"),
            da.full((n_f, n_t), np.nan, dtype=np.float32, chunks=(n_f, t_chunk)),
        ),
        "bottom_range_m": (("time",), da.full((n_t,), np.nan, dtype=np.float32, chunks=(t_chunk,))),
    }
    if "pulse_length" in sub:
        data["pulse_length_s"] = (
            ("time", "frequency"),
            sub["pulse_length"].transpose("ping_time", "frequency").values.astype(np.float32),
        )
    if "transducer_draft" in sub:
        data["transducer_draft_m"] = (
            ("time",),
            sub["transducer_draft"].isel(frequency=ref).values.astype(np.float32),
        )
    for name in ("latitude", "longitude", "distance"):
        if name in sub:
            data[name] = (("time",), sub[name].values)
    coords = {"frequency": freqs, "time": sub["ping_time"].values, "range": centres}
    out = xr.Dataset(data, coords=coords)
    out.attrs.update(
        {
            "description": "Whole-cruise backscatter after preprocessing, averaged to cells.",
            "mask_bits": json.dumps(MASK_BITS),
            "ping_factor": pf,
            "range_step_m": float(rf * np.median(np.diff(rng))),
            "floor_db": float(params.get("floor_db", -130.0)),
            "source_ping_index": [i_start, i_end],
            "sv_db_units": "dB re 1 m-1",
            "range_definition": "cell centre range from the transducer; depth = range + transducer_draft_m",
            "valid_fraction_definition": "share of samples in the cell with no mask bit, scaled to 0 to 255",
        }
    )
    encoding = {"valid_fraction": {"_FillValue": None}, "hard_mask": {"_FillValue": None}}
    out.to_zarr(str(out_path), mode="w", zarr_format=2, encoding=encoding, compute=False, consolidated=True)
    return out


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_downsample_cruise(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Preprocess the whole cruise in parallel chunks and write the downsampled store."""
    ccfg = cfg.get("cruise", {})
    label = str(ccfg.get("label", "cruise"))
    store = resolve_path(cfg, "source_data", cfg.get("store", "S2018105_sv.zarr"))
    out_path = resolve_path(cfg, "scratch_dir", "acoustics", f"{label}.zarr")
    progress_dir = out_path.parent / f"{label}.progress"
    inputs: dict[str, str | Path] = {"store": store}
    if not force and out_path.exists() and manifest_is_current(out_dir, cfg, inputs):
        log.info("cruise downsampling is up to date in %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _params(cfg)
    ds = open_sv_store(store)
    n_total = ds.sizes["ping_time"]
    i_start, i_end = 0, n_total
    if ccfg.get("start") and ccfg.get("end"):
        i_start, i_end = _ping_bounds(ds, str(ccfg["start"]), str(ccfg["end"]))
    chunk_pings = int(ds["sv"].chunks[1][0])
    pf = int(ccfg.get("ping_factor", 1))
    rng = ds["range"].values
    rf = max(1, int(round(float(ccfg.get("range_step_m", 1.0)) / float(np.median(np.diff(rng))))))
    if chunk_pings % pf:
        raise ValueError(f"ping_factor {pf} must divide the source chunk length {chunk_pings}")
    n_chunks = math.ceil((i_end - i_start) / chunk_pings)
    n_cells = (i_end - i_start) // pf
    ref = frequency_position(ds, float(params.get("bottom", {}).get("reference_hz", 38000)))
    settings = {
        "i_start": i_start,
        "i_end": i_end,
        "n_total": n_total,
        "chunk_pings": chunk_pings,
        "ping_factor": pf,
        "range_factor": rf,
        "overlap_pings": int(ccfg.get("overlap_pings", 60)),
        "n_cells": n_cells,
        "progress_dir": str(progress_dir),
    }
    n_f, n_d = ds.sizes["frequency"], len(rng) // rf
    expected_bytes = n_f * n_cells * n_d * 4 + n_f * n_cells * 4
    log.info(
        "cruise product %s: %d cells by %d range cells at %d frequencies, %.1f GB before compression",
        label,
        n_cells,
        n_d,
        n_f,
        expected_bytes / 1e9,
    )
    fresh = force or not out_path.exists() or not progress_dir.exists()
    if fresh:
        if progress_dir.exists():
            for p in progress_dir.glob("*.json"):
                p.unlink()
        progress_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _cruise_skeleton(ds, out_path, settings, params, ref)
    todo = [k for k in range(n_chunks) if not (progress_dir / f"{k}.json").exists()]
    log.info("%d of %d chunks to process", len(todo), n_chunks)
    n_workers = int(ccfg.get("n_workers", min(int(cfg.get("n_workers", 16)), 16)))
    t_start = time.time()
    if todo:
        ctx = get_context("spawn")
        with ctx.Pool(
            min(n_workers, len(todo)),
            initializer=_init_worker,
            initargs=(str(store), str(out_path), params, settings),
        ) as pool:
            for n, rec in enumerate(pool.imap_unordered(_process_chunk, todo), start=1):
                if n % 10 == 0 or n == len(todo):
                    elapsed = time.time() - t_start
                    log.info(
                        "%d/%d chunks, %.0f s elapsed, last chunk %d in %.0f s",
                        n,
                        len(todo),
                        elapsed,
                        rec["chunk"],
                        rec["seconds"],
                    )
    records = [json.load(open(p, encoding="utf-8")) for p in sorted(progress_dir.glob("*.json"))]
    table = pd.DataFrame(records).sort_values("chunk").reset_index(drop=True)
    table.to_csv(out_dir / "chunks.csv", index=False)
    weights = (table["i1"] - table["i0"]).to_numpy(dtype=float)
    per_frequency: dict[str, Any] = {}
    for f in ds["frequency"].values:
        key = str(int(f))
        stats: dict[str, Any] = {}
        for name in list(MASK_BITS) + ["valid_fraction", "impulsive_pings_fraction"]:
            col = f"{name}_{key}"
            if col in table:
                stats[name] = float(np.average(table[col].to_numpy(dtype=float), weights=weights))
        nd = table.get(f"noise_db_median_{key}")
        if nd is not None and nd.notna().any():
            stats["noise_db_median_of_chunks"] = float(nd.median())
            stats["noise_db_chunk_range"] = [float(nd.min()), float(nd.max())]
        pulse = table.get(f"pulse_ms_{key}")
        if pulse is not None and nd is not None:
            by_pulse = table.groupby(pulse)[f"noise_db_median_{key}"].median()
            stats["noise_db_by_pulse_ms"] = {str(k): float(v) for k, v in by_pulse.items()}
        per_frequency[key] = stats
    summary = {
        "label": label,
        "store": str(out_path),
        "shape": [n_f, n_cells, n_d],
        "ping_factor": pf,
        "range_step_m": float(rf * np.median(np.diff(rng))),
        "n_chunks": n_chunks,
        "chunks_done": int(len(table)),
        "expected_bytes_uncompressed": int(expected_bytes),
        "bytes_on_disk": int(_dir_size(out_path)) if out_path.exists() else None,
        "seconds_this_run": round(time.time() - t_start, 1),
        "worker_seconds_total": float(table["seconds"].sum()) if "seconds" in table else None,
        "bottom_detected_fraction": float(np.average(table["bottom_detected_fraction"], weights=weights)),
        "per_frequency": per_frequency,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if len(table) == n_chunks:
        write_manifest(
            out_dir, "downsample-cruise", cfg, inputs, [out_dir / "summary.json", out_dir / "chunks.csv"]
        )
    else:
        log.warning("%d chunks still missing; rerun to complete", n_chunks - len(table))
    return out_dir
