"""Stage 8 commands: train one acoustic model configuration and pretrain the transformer.

``train-acoustic`` trains one experiment on one fold of one protocol with
one seed: it reads the window store and the splits of the configured
detector run, fits the input normalisation on the training windows of the
fold, trains the network or the baseline, and writes the predictive
distribution parameters, means and quantiles for the validation and test
windows next to the targets. ``pretrain-mae`` trains the masked autoencoder
on patches of the whole-cruise product and stores the encoder weights.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .manifest import manifest_is_current, write_manifest
from .models.baselines import build_baseline
from .models.counts import n_params, predictive_mean, quantiles, smearing_factor, tweedie_dispersion
from .models.data import (
    InputSpec,
    TargetSpec,
    WindowStore,
    build_inputs,
    build_targets,
    context_features,
    normalisation_stats,
    window_features,
)
from .models.nets import build_network
from .models.train import TrainConfig, load_pretrained_encoder, predict_params, train_model

log = logging.getLogger(__name__)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def experiment_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Experiment settings: the defaults overridden by the named entry."""
    if name not in cfg.get("experiments", {}):
        raise KeyError(f"unknown experiment {name}")
    exp = _merge(cfg.get("defaults", {}), cfg["experiments"][name])
    exp["name"] = name
    return exp


def _window_paths(cfg: dict[str, Any], geometry: str, variant: str = "") -> tuple[Path, Path, Path]:
    """Paths of the windows, metadata and splits of one geometry; ``variant`` names a lag-model variant."""
    run = cfg.get("windows_run", {})
    tag = f"{run.get('experiment', 'dfine_l_1024')}_seed{int(run.get('seed', 0))}"
    variant = variant or str(run.get("variant", "") or "")
    if variant:
        tag = f"{tag}_{variant}"
    project = resolve_path(cfg, "project_dir")
    zarr_path = resolve_path(cfg, "scratch_dir", "windows", tag, f"{geometry}.zarr")
    meta_dir = Path(project, "runs", "windows", tag, geometry)
    splits = Path(project, "runs", "windows", tag, "splits", f"{geometry}.parquet")
    return zarr_path, meta_dir, splits


def _prediction_table(
    store: WindowStore,
    index: np.ndarray,
    split: str,
    family: str,
    params: dict[str, np.ndarray],
    classes: list[str],
    tspec: TargetSpec,
    extra: dict[str, Any],
) -> pd.DataFrame:
    mean = predictive_mean(family, params, extra.get("smearing", 1.0))
    qs = quantiles(
        family,
        params,
        QUANTILES,
        **{k: v for k, v in extra.items() if k in ("tweedie_power", "tweedie_phi", "log1p_sigma")},
    )
    table = pd.DataFrame({"window_id": store.meta.loc[index, "window_id"].to_numpy(), "split": split})
    for c_idx, name in enumerate(classes):
        table[f"mean_{name}"] = mean[:, c_idx]
        for q_idx, q in enumerate(QUANTILES):
            table[f"q{int(q * 100):02d}_{name}"] = qs[q_idx, :, c_idx]
        for p_name, values in params.items():
            table[f"{p_name}_{name}"] = values[:, c_idx]
        for suffix in ("", "_at_lag"):
            col = f"{tspec.form}_{name}{suffix}"
            if col in store.targets.columns:
                table[f"y_{name}{suffix or '_expected'}"] = store.targets.loc[index, col].to_numpy()
    return table


def run_train_acoustic(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    fold: str = "",
    protocol: str = "",
) -> Path:
    """Train one experiment on one fold and write its predictions."""
    exp = experiment_config(cfg, experiment)
    protocol = protocol or str(exp.get("protocol", "loto_unflagged"))
    geometry = str(exp.get("geometry", "base"))
    zarr_path, meta_dir, splits_path = _window_paths(cfg, geometry, str(exp.get("windows_variant", "")))
    target = out_dir / experiment / protocol / fold / f"seed{seed}"
    inputs: dict[str, str | Path] = {
        "meta": meta_dir / "meta.parquet",
        "targets": meta_dir / "targets.parquet",
        "splits": splits_path,
    }
    run_cfg = {**cfg, "experiment": experiment, "seed": seed, "fold": fold, "protocol": protocol}
    if not force and manifest_is_current(target, run_cfg, inputs):
        log.info("%s %s %s seed %d is up to date", experiment, protocol, fold, seed)
        return target
    target.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    store = WindowStore(zarr_path, meta_dir)
    parts = store.splits(splits_path, protocol, fold)
    if "train" not in parts or "test" not in parts:
        raise RuntimeError(f"fold {fold} of {protocol} has no train or test windows")
    val_idx = parts.get("val", parts["train"][-max(1, len(parts["train"]) // 10) :])
    ispec = InputSpec.from_config(exp.get("inputs", {}))
    tspec = TargetSpec.from_config(exp.get("targets", {}))
    classes = tspec.classes or store.classes
    tspec.classes = [c for c in classes if f"{tspec.form}_{c}_at_lag" in store.targets.columns]
    family = str(exp.get("family", "nb"))
    network = str(exp.get("network", "cnn"))
    device = "cuda" if _cuda() else "cpu"
    y = {
        s: build_targets(
            store,
            idx,
            TargetSpec(
                tspec.form, tspec.classes, "expected" if tspec.lag_offsets == "sample" else tspec.lag_offsets
            ),
        )
        for s, idx in (("train", parts["train"]), ("val", val_idx), ("test", parts["test"]))
    }
    extra: dict[str, Any] = {"tweedie_power": float(exp.get("train", {}).get("tweedie_power", 1.5))}
    history: list[dict[str, float]] = []
    summary: dict[str, Any] = {
        "experiment": experiment,
        "protocol": protocol,
        "fold": fold,
        "seed": seed,
        "geometry": geometry,
        "family": family,
        "network": network,
        "classes": tspec.classes,
        "n_windows": {
            s: int(len(i)) for s, i in (("train", parts["train"]), ("val", val_idx), ("test", parts["test"]))
        },
        "device": device,
    }
    if network in ("mean", "gbm"):
        # A baseline may be given named properties of the window in place of the echogram statistics.
        feature_names = [str(f) for f in exp.get("features") or []]
        feats = {
            s: context_features(store, idx, feature_names)
            if feature_names
            else window_features(store, idx, ispec)
            for s, idx in (("train", parts["train"]), ("val", val_idx), ("test", parts["test"]))
        }
        model = build_baseline(network, seed, exp.get("net"))
        model.fit(feats["train"], y["train"])
        family = model.family
        params = {s: model.predict_params(feats[s]) for s in ("val", "test")}
        train_params = model.predict_params(feats["train"])
    else:
        stats = normalisation_stats(store, parts["train"], ispec)
        x = {
            s: build_inputs(store, idx, ispec, stats).astype(np.float16)
            for s, idx in (("train", parts["train"]), ("val", val_idx), ("test", parts["test"]))
        }
        # A two-scale network also reads the coarse full-column context stored with the geometry.
        use_context = bool((exp.get("net") or {}).get("context", False))
        c: dict[str, np.ndarray | None] = {"train": None, "val": None, "test": None}
        if use_context:
            if "context_sv_db" not in store.group:
                raise RuntimeError(f"geometry {geometry} has no context arrays for a two-scale network")
            cstats = normalisation_stats(store, parts["train"], ispec, context=True)
            stats = {**stats, "context_mean": cstats["mean"], "context_std": cstats["std"]}
            c = {
                s: build_inputs(store, idx, ispec, cstats, context=True).astype(np.float16)
                for s, idx in (("train", parts["train"]), ("val", val_idx), ("test", parts["test"]))
            }
        net = build_network(network, ispec.n_channels, len(tspec.classes) * n_params(family), exp.get("net"))
        if exp.get("pretrained"):
            weights = resolve_path(cfg, "project_dir", "weights", "acoustic", f"{exp['pretrained']}.pt")
            summary["pretrained_tensors"] = load_pretrained_encoder(net, weights)
        offsets, offset_weights = None, None
        if tspec.lag_offsets == "sample":
            columns = store.target_offset_columns(tspec)
            offsets = np.stack(
                [store.targets.loc[parts["train"], cols].to_numpy(dtype=np.float32) for cols in columns],
                axis=2,
            )
            from scipy.stats import norm

            z = np.linspace(-2, 2, len(columns))
            offset_weights = norm.pdf(z) / norm.pdf(z).sum()
        checkpoints = resolve_path(cfg, "scratch_dir", "acoustic", experiment, protocol, fold, f"seed{seed}")
        result = train_model(
            net,
            family,
            x["train"],
            y["train"],
            x["val"],
            y["val"],
            TrainConfig.from_config(exp.get("train", {})),
            device,
            seed,
            checkpoints,
            offsets,
            offset_weights,
            str(exp.get("sampler", "uniform")),
            resume=not force,
            c_train=c["train"],
            c_val=c["val"],
        )
        history = result["history"]
        summary.update(
            {
                "best_epoch": result["best_epoch"],
                "best_val_mae": result["best_val_mae"],
                "n_parameters": int(sum(p.numel() for p in net.parameters())),
            }
        )
        np.savez(target / "normalisation.npz", **stats)
        params = {
            s: predict_params(net, family, x[s], len(tspec.classes), device, context=c[s])
            for s in ("val", "test")
        }
        train_params = predict_params(net, family, x["train"], len(tspec.classes), device, context=c["train"])
    if family == "log1p":
        extra["smearing"] = smearing_factor(train_params["log1p_mu"], y["train"])
        residual = np.log1p(np.maximum(y["train"], 0)) - train_params["log1p_mu"]
        extra["log1p_sigma"] = float(residual.std())
    if family == "tweedie":
        extra["tweedie_phi"] = tweedie_dispersion(
            np.exp(train_params["log_mu"]), y["train"], extra["tweedie_power"]
        )
    summary["family"] = family
    summary["extra"] = {k: float(v) for k, v in extra.items()}
    outputs: list[str | Path] = []
    for split, idx in (("val", val_idx), ("test", parts["test"])):
        table = _prediction_table(store, idx, split, family, params[split], tspec.classes, tspec, extra)
        path = target / f"predictions_{split}.parquet"
        table.to_parquet(path, index=False)
        outputs.append(path)
        mean_cols = [f"mean_{c}" for c in tspec.classes]
        y_cols = [f"y_{c}_at_lag" for c in tspec.classes if f"y_{c}_at_lag" in table]
        if y_cols:
            summary[f"{split}_mae_macro"] = float(
                np.mean(np.abs(table[mean_cols].to_numpy() - table[y_cols].to_numpy()).mean(axis=0))
            )
    if history:
        pd.DataFrame(history).to_csv(target / "history.csv", index=False)
        outputs.append(target / "history.csv")
    summary["seconds"] = round(time.time() - t_start, 1)
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    outputs.append(target / "summary.json")
    write_manifest(target, "train-acoustic", run_cfg, inputs, outputs)
    log.info(
        "%s %s %s seed %d: val mae %.3f, test mae %.3f, %.0f s",
        experiment,
        protocol,
        fold,
        seed,
        summary.get("val_mae_macro", float("nan")),
        summary.get("test_mae_macro", float("nan")),
        summary["seconds"],
    )
    return target


def run_pretrain_mae(cfg: dict[str, Any], out_dir: Path, force: bool = False, resume: bool = False) -> Path:
    """Masked autoencoding on the whole-cruise product; writes the encoder weights to the project."""
    import torch

    from .models.nets import MaskedAutoencoder
    from .models.pretrain import CruisePatchSampler, pretrain

    pcfg = cfg.get("pretraining", {})
    name = str(pcfg.get("name", "mae_base"))
    store = resolve_path(cfg, "scratch_dir", pcfg.get("store", "acoustics/cruise_1ping_1m.zarr"))
    target = out_dir / name
    weights_dir = resolve_path(cfg, "project_dir", "weights", "acoustic")
    inputs: dict[str, str | Path] = {"store": store}
    if not force and manifest_is_current(target, {**cfg, "stage": "pretrain"}, inputs):
        log.info("pretraining %s is up to date", name)
        return target
    target.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    ispec = InputSpec.from_config(cfg.get("defaults", {}).get("inputs", {}))
    sampler = CruisePatchSampler(
        store,
        ispec,
        int(pcfg.get("n_time", 30)),
        int(pcfg.get("n_depth", 50)),
        float(pcfg.get("depth_min_m", 10)),
        float(pcfg.get("depth_max_m", 450)),
        seed=int(cfg.get("seed", 0)),
    )
    stats = sampler.fit_stats()
    model = MaskedAutoencoder(
        ispec.n_channels,
        tuple(pcfg.get("patch", [3, 5])),
        int(pcfg.get("dim", 128)),
        int(pcfg.get("depth", 4)),
        int(pcfg.get("heads", 4)),
        mask_ratio=float(pcfg.get("mask_ratio", 0.75)),
    )
    device = "cuda" if _cuda() else "cpu"
    checkpoints = resolve_path(cfg, "scratch_dir", "acoustic", "pretraining", name)
    history = pretrain(
        sampler,
        model,
        int(pcfg.get("steps", 20000)),
        int(pcfg.get("batch_size", 128)),
        float(pcfg.get("lr", 1.5e-4)),
        device,
        checkpoints,
        resume=resume,
    )
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "stats": stats,
            "config": pcfg,
            "inputs": cfg.get("defaults", {}).get("inputs", {}),
        },
        weights_dir / f"{name}.pt",
    )
    pd.DataFrame(history).to_csv(target / "history.csv", index=False)
    summary = {
        "name": name,
        "steps": int(pcfg.get("steps", 20000)),
        "final_loss": history[-1]["loss"] if history else None,
        "seconds": history[-1]["seconds"] if history else None,
        "device": device,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "weights": str(weights_dir / f"{name}.pt"),
    }
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_manifest(
        target,
        "pretrain-mae",
        {**cfg, "stage": "pretrain"},
        inputs,
        [target / "history.csv", target / "summary.json"],
    )
    return target


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False
