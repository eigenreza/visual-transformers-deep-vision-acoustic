"""Stage 9 command: attribution maps and case studies for one trained acoustic run.

The trained network of an experiment, fold and seed is rebuilt from its
configuration and best checkpoint; for every class the windows with the
smallest and the largest absolute error on the test split are attributed
with the methods that suit the architecture (integrated gradients and
Grad-CAM for the convolutional network, attention rollout and
gradient-weighted attention for the transformer), the share of attribution
mass on the camera depth band is measured for every test window, and the
case studies are drawn as the 38 kHz patch with the map on top. A second
command summarises the band mass over all attributed runs against the
share that a map spread evenly over the window would put on the band.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import resolve_path
from .eval.attribution import (
    attention_rollout,
    class_mean_target,
    floor_baseline,
    grad_cam,
    gradient_weighted_attention,
    integrated_gradients,
    mass_on_band,
)
from .eval.metrics import fold_block_bootstrap
from .manifest import manifest_is_current, write_manifest
from .models.counts import n_params
from .models.data import InputSpec, WindowStore, build_inputs
from .models.nets import build_network
from .models_stage import _window_paths, experiment_config
from .viz.results import band_mass_plot

log = logging.getLogger(__name__)

# Longest names first, so that a file name is split at the right place.
METHODS = ("gradient_weighted_attention", "integrated_gradients", "attention_rollout", "grad_cam")


def _methods(network: str) -> list[str]:
    return (
        ["integrated_gradients", "grad_cam"]
        if network == "cnn"
        else ["attention_rollout", "gradient_weighted_attention"]
    )


def attribute(
    net: torch.nn.Module, network: str, method: str, x: torch.Tensor, target, n_db: int, floor_value: float
) -> np.ndarray:
    if method == "integrated_gradients":
        return integrated_gradients(net, x, target, floor_baseline(x, n_db, floor_value), steps=32)
    if method == "grad_cam":
        return grad_cam(net, x, target)
    vit = net.vit if network == "hybrid" else net
    if network == "hybrid":
        raise ValueError(
            "attention methods are run on the transformer alone; use integrated_gradients for the hybrid"
        )
    if method == "attention_rollout":
        return attention_rollout(vit, x)
    return gradient_weighted_attention(vit, x, target)


def _figure(
    sv38: np.ndarray,
    attribution: np.ndarray,
    depth_cells: np.ndarray,
    camera_depth: float,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    extent = (0, sv38.shape[0], depth_cells[-1], depth_cells[0])
    axes[0].imshow(sv38.T, aspect="auto", extent=extent, cmap="viridis", vmin=-90, vmax=-50)
    axes[0].axhline(camera_depth, color="white", lw=1)
    axes[0].set_title("38 kHz, dB")
    axes[0].set_ylabel("depth (m)")
    axes[0].set_xlabel("time cell")
    scale = np.abs(attribution).max() or 1.0
    axes[1].imshow((attribution / scale).T, aspect="auto", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].axhline(camera_depth, color="black", lw=1)
    axes[1].set_title("attribution")
    axes[1].set_xlabel("time cell")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_attribute_acoustic(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    fold: str = "",
    protocol: str = "",
) -> Path:
    """Attribution maps, band-mass statistics and case studies for one trained run."""
    exp = experiment_config(cfg, experiment)
    protocol = protocol or str(exp.get("protocol", "loto_unflagged"))
    network = str(exp.get("network", "cnn"))
    if network in ("mean", "gbm"):
        raise ValueError("attribution applies to the networks only")
    acfg = cfg.get("attribution", {})
    run_dir = resolve_path(cfg, "project_dir", "runs", "acoustic", experiment, protocol, fold, f"seed{seed}")
    checkpoint = resolve_path(
        cfg, "scratch_dir", "acoustic", experiment, protocol, fold, f"seed{seed}", "best.pt"
    )
    target = out_dir / experiment / protocol / fold / f"seed{seed}"
    inputs: dict[str, str | Path] = {
        "checkpoint": checkpoint,
        "predictions": run_dir / "predictions_test.parquet",
    }
    run_cfg = {
        **cfg,
        "experiment": experiment,
        "seed": seed,
        "fold": fold,
        "protocol": protocol,
        "stage": "attribution",
    }
    if not force and manifest_is_current(target, run_cfg, inputs):
        log.info("attribution of %s %s seed %d is up to date", experiment, fold, seed)
        return target
    target.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    classes, family = list(summary["classes"]), str(summary["family"])
    geometry = str(exp.get("geometry", "base"))
    if (exp.get("net") or {}).get("context"):
        log.warning("%s is a two-scale network; attribution is run on single-input networks only", experiment)
        return target
    zarr_path, meta_dir, splits_path = _window_paths(cfg, geometry, str(exp.get("windows_variant", "")))
    store = WindowStore(zarr_path, meta_dir)
    ispec = InputSpec.from_config(exp.get("inputs", {}))
    stats = dict(np.load(run_dir / "normalisation.npz"))
    net = build_network(network, ispec.n_channels, len(classes) * n_params(family), exp.get("net"))
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    net.eval()
    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    position = pd.Series(np.arange(len(store.meta)), index=store.meta["window_id"])
    rows_idx = position.loc[preds["window_id"].to_numpy()].to_numpy()
    n_db = len(ispec.frequencies_hz) + len(ispec.differences)
    floor_value = float(((ispec.fill_db - stats["mean"][:n_db]) / stats["std"][:n_db]).mean())
    methods = list(acfg.get("methods", _methods(network)))
    if network == "hybrid":
        methods = ["integrated_gradients"]
    n_cases = int(acfg.get("n_cases_per_class", 3))
    half_width = float(acfg.get("band_half_width_m", 5.0))
    batch = int(acfg.get("batch_size", 64))
    camera_depth = store.meta.loc[rows_idx, "camera_depth_m"].to_numpy(dtype=float)
    depth_cells = store.group["depth_cells"].oindex[rows_idx, :]
    pos38 = store.channel_positions([38000.0])[0] if 38000.0 in store.frequencies else 0
    mass_rows, case_rows = [], []
    figures: list[Path] = []
    for k, class_name in enumerate(classes):
        err = np.abs(preds[f"mean_{class_name}"].to_numpy() - preds[f"y_{class_name}_at_lag"].to_numpy())
        target_fn = class_mean_target(family, len(classes), k)
        for method in methods:
            maps = np.zeros((len(rows_idx), *store.group["sv_db"].shape[2:]), dtype=np.float32)
            for i in range(0, len(rows_idx), batch):
                sel = rows_idx[i : i + batch]
                x = torch.as_tensor(
                    build_inputs(store, sel, ispec, stats), dtype=torch.float32
                ).requires_grad_(method != "attention_rollout")
                maps[i : i + batch] = attribute(net, network, method, x, target_fn, n_db, floor_value)
            share = mass_on_band(maps, depth_cells, camera_depth, half_width)
            median_err = np.nanmedian(err)
            for group, sel in (
                ("all", np.ones(len(err), dtype=bool)),
                ("better_half", err <= median_err),
                ("worse_half", err > median_err),
            ):
                mass_rows.append(
                    {
                        "class_name": class_name,
                        "method": method,
                        "group": group,
                        "n": int(sel.sum()),
                        "band_mass_mean": float(np.nanmean(share[sel])) if sel.any() else np.nan,
                        "band_mass_median": float(np.nanmedian(share[sel])) if sel.any() else np.nan,
                    }
                )
            order = np.argsort(err)
            present = np.flatnonzero(preds[f"y_{class_name}_at_lag"].to_numpy() > 0)
            best = [i for i in order if i in set(present)][:n_cases]
            worst = list(order[::-1][:n_cases])
            for label, cases in (("best", best), ("worst", worst)):
                for i in cases:
                    wid = int(preds.loc[i, "window_id"])
                    sv38 = store.group["sv_db"].oindex[int(rows_idx[i]), pos38, :, :].astype(np.float32)
                    path = target / f"{class_name}_{method}_{label}_{wid}.png"
                    y_true = float(preds.loc[i, f"y_{class_name}_at_lag"])
                    y_hat = float(preds.loc[i, f"mean_{class_name}"])
                    title = (
                        f"{experiment} {fold}: {class_name}, {method}, {label}: "
                        f"y={y_true:.1f}, mean={y_hat:.1f}, band mass {share[i]:.2f}"
                    )
                    _figure(sv38, maps[i], depth_cells[i], camera_depth[i], title, path)
                    figures.append(path)
                    case_rows.append(
                        {
                            "class_name": class_name,
                            "method": method,
                            "case": label,
                            "window_id": wid,
                            "y": float(preds.loc[i, f"y_{class_name}_at_lag"]),
                            "mean": float(preds.loc[i, f"mean_{class_name}"]),
                            "band_mass": float(share[i]),
                            "figure": path.name,
                        }
                    )
            np.savez_compressed(
                target / f"maps_{class_name}_{method}.npz",
                maps=maps.astype(np.float16),
                window_id=preds["window_id"].to_numpy(),
                band_mass=share,
            )
    pd.DataFrame(mass_rows).to_csv(target / "band_mass.csv", index=False)
    pd.DataFrame(case_rows).to_csv(target / "cases.csv", index=False)
    write_manifest(
        target,
        "attribute-acoustic",
        run_cfg,
        inputs,
        [target / "band_mass.csv", target / "cases.csv", *figures],
    )
    return target


def half_difference(
    mass: np.ndarray,
    worse: np.ndarray,
    blocks: np.ndarray,
    folds: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Mean band mass of the worse-predicted windows minus that of the better ones, mean over folds.

    ``worse`` marks the windows whose absolute error lies above the median of
    their fold. The interval comes from time blocks resampled within folds.
    """
    rng = np.random.default_rng(seed)
    values, stats = [], []
    for name in np.unique(folds):
        sel = (folds == name) & np.isfinite(mass)
        m, w, b = mass[sel], worse[sel].astype(bool), blocks[sel]
        if w.sum() == 0 or (~w).sum() == 0:
            continue
        labels, inverse = np.unique(b, return_inverse=True)
        n = len(labels)
        sum_w = np.bincount(inverse, weights=m * w, minlength=n)
        n_w = np.bincount(inverse, weights=w.astype(float), minlength=n)
        sum_b = np.bincount(inverse, weights=m * ~w, minlength=n)
        n_b = np.bincount(inverse, weights=(~w).astype(float), minlength=n)
        draw = rng.integers(0, n, size=(n_boot, n))
        with np.errstate(invalid="ignore", divide="ignore"):
            stats.append(
                sum_w[draw].sum(axis=1) / n_w[draw].sum(axis=1)
                - sum_b[draw].sum(axis=1) / n_b[draw].sum(axis=1)
            )
        values.append(float(m[w].mean() - m[~w].mean()))
    if not values:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    stat = np.nanmean(np.stack(stats), axis=0)
    return {
        "value": float(np.mean(values)),
        "ci_low": float(np.nanpercentile(stat, 2.5)),
        "ci_high": float(np.nanpercentile(stat, 97.5)),
    }


def run_summarize_attribution(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Band mass of every attributed run against the share of an even map, with intervals.

    For every experiment, method and class the table gives the mean over
    folds of the share of attribution mass within the camera band, the share
    that a map spread evenly over the window puts there, their difference,
    and the difference between the worse and the better predicted half of
    the windows, each with an interval from time blocks resampled within
    folds. The row of the class ``all`` averages the classes per window.
    """
    acfg = cfg.get("attribution", {})
    evaluation = cfg.get("evaluation", {})
    half_width = float(acfg.get("band_half_width_m", 5.0))
    block_s = float(evaluation.get("block_s", 300))
    n_boot = int(evaluation.get("n_boot", 500))
    root = resolve_path(cfg, "project_dir", "runs", "attribution")
    runs_root = resolve_path(cfg, "project_dir", "runs", "acoustic")
    out_dir.mkdir(parents=True, exist_ok=True)
    stores: dict[tuple[str, str], WindowStore] = {}
    parts = []
    for run_dir in sorted(p.parent for p in root.glob("*/*/*/seed*/band_mass.csv")):
        experiment, protocol, fold, seed = run_dir.parts[-4:]
        exp = experiment_config(cfg, experiment)
        key = (str(exp.get("geometry", "base")), str(exp.get("windows_variant", "")))
        if key not in stores:
            zarr_path, meta_dir, _ = _window_paths(cfg, *key)
            stores[key] = WindowStore(zarr_path, meta_dir)
        store = stores[key]
        position = pd.Series(np.arange(len(store.meta)), index=store.meta["window_id"])
        preds = pd.read_parquet(
            Path(runs_root, experiment, protocol, fold, seed, "predictions_test.parquet")
        ).set_index("window_id")
        for path in sorted(run_dir.glob("maps_*.npz")):
            name = path.stem[len("maps_") :]
            method = next((m for m in METHODS if name.endswith(f"_{m}")), None)
            if method is None:
                continue
            class_name = name[: -len(method) - 1]
            data = np.load(path)
            ids = data["window_id"]
            idx = position.loc[ids].to_numpy()
            camera_depth = store.meta.loc[idx, "camera_depth_m"].to_numpy(dtype=float)
            depth_cells = np.asarray(store.group["depth_cells"].oindex[idx, :], dtype=float)
            uniform = (np.abs(depth_cells - camera_depth[:, None]) <= half_width).mean(axis=1)
            error = np.abs(
                preds.loc[ids, f"mean_{class_name}"].to_numpy(dtype=float)
                - preds.loc[ids, f"y_{class_name}_at_lag"].to_numpy(dtype=float)
            )
            if "time_camera" in store.meta.columns:
                stamp = pd.to_datetime(store.meta.loc[idx, "time_camera"]).astype("int64").to_numpy()
                transect = store.meta.loc[idx, "transect"].to_numpy()
                blocks = [
                    f"{fold}|{t}|{int(s) // int(block_s * 1e9)}" for t, s in zip(transect, stamp, strict=True)
                ]
            else:
                blocks = [f"{fold}|{i}" for i in ids]
            parts.append(
                pd.DataFrame(
                    {
                        "experiment": experiment,
                        "protocol": protocol,
                        "fold": fold,
                        "seed": seed,
                        "class_name": class_name,
                        "method": method,
                        "window_id": ids,
                        "band_mass": np.asarray(data["band_mass"], dtype=float),
                        "uniform_share": uniform,
                        "worse_half": error > np.nanmedian(error),
                        "block": blocks,
                    }
                )
            )
    if not parts:
        raise FileNotFoundError(f"no attributed run under {root}")
    windows = pd.concat(parts, ignore_index=True)
    rows = []
    for (experiment, protocol, method), g in windows.groupby(
        ["experiment", "protocol", "method"], sort=False
    ):
        pooled = g.groupby(["fold", "seed", "window_id"], as_index=False).agg(
            band_mass=("band_mass", "mean"),
            uniform_share=("uniform_share", "first"),
            block=("block", "first"),
        )
        for class_name, gc in [*g.groupby("class_name", sort=False), ("all", pooled)]:
            mass = gc["band_mass"].to_numpy(dtype=float)
            uniform = gc["uniform_share"].to_numpy(dtype=float)
            blocks, folds = gc["block"].to_numpy(), gc["fold"].to_numpy()
            share = fold_block_bootstrap(mass, blocks, folds, n_boot)
            excess = fold_block_bootstrap(mass - uniform, blocks, folds, n_boot)
            row = {
                "experiment": experiment,
                "protocol": protocol,
                "method": method,
                "class_name": class_name,
                "n_windows": share["n"],
                "n_folds": share["n_folds"],
                "band_mass": share["value"],
                "band_mass_ci_low": share["ci_low"],
                "band_mass_ci_high": share["ci_high"],
                "uniform_share": fold_block_bootstrap(uniform, blocks, folds, 1)["value"],
                "excess": excess["value"],
                "excess_ci_low": excess["ci_low"],
                "excess_ci_high": excess["ci_high"],
            }
            if "worse_half" in gc.columns:
                half = half_difference(mass, gc["worse_half"].to_numpy(), blocks, folds, n_boot)
                row.update(
                    {
                        "worse_minus_better": half["value"],
                        "worse_minus_better_ci_low": half["ci_low"],
                        "worse_minus_better_ci_high": half["ci_high"],
                    }
                )
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "band_mass_summary.csv", index=False)
    figures = [
        band_mass_plot(
            table,
            out_dir / f"band_mass.{fmt}",
            title=f"Attribution mass within {half_width:g} m of the camera depth",
        )
        for fmt in acfg.get("formats", ["pdf", "png"])
    ]
    write_manifest(
        out_dir,
        "summarize-attribution",
        cfg,
        {"attribution": root},
        [out_dir / "band_mass_summary.csv", *figures],
    )
    log.info("summarised %d attributed runs", windows.groupby(["experiment", "fold", "seed"]).ngroups)
    return out_dir
