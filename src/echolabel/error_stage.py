"""Stage 9 command: where the errors of the acoustic models fall.

For each named configuration the test predictions of every fold are scored
per window against the common truth, averaged over seeds, and joined with
the window metadata. The absolute error, the signed error and the absolute
error in log(1 + count) are tabulated per class against the transect, the
camera depth, the hour of day, the mean confidence of the counted objects,
the width of the lag support and the size of the true count, with
block-bootstrap intervals, and drawn as one figure per configuration. A
second table gives the rank correlation between prediction and truth
within every test tow, which separates following the count along a tow
from matching its level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .eval.metrics import block_bootstrap_mean
from .eval_stage import TRUTH_FORM
from .manifest import write_manifest
from .models_stage import _window_paths, experiment_config
from .viz.results import error_factor_plot

log = logging.getLogger(__name__)

ABSENT = "absent"


def window_errors(preds_by_seed: list[pd.DataFrame], classes: list[str]) -> pd.DataFrame:
    """Truth, prediction and errors per window and class, averaged over the seeds of one fold.

    Every table holds ``window_id`` and, per class, ``y_<class>`` (count) and
    ``mean_<class>`` (predictive mean, count); the seeds share their windows.
    Returns one row per window with the truth, the mean prediction, the
    absolute error, the signed error (prediction minus truth) and the
    absolute error in log(1 + count) of every class.
    """
    first = preds_by_seed[0].set_index("window_id")
    out = pd.DataFrame(index=first.index)
    for c in classes:
        y = first[f"y_{c}"].to_numpy(dtype=float)
        mean = np.stack(
            [
                t.set_index("window_id").loc[first.index, f"mean_{c}"].to_numpy(dtype=float)
                for t in preds_by_seed
            ]
        )
        out[f"y_{c}"] = y
        out[f"mean_{c}"] = mean.mean(axis=0)
        out[f"abs_error_{c}"] = np.abs(mean - y).mean(axis=0)
        out[f"error_{c}"] = (mean - y).mean(axis=0)
        out[f"log_error_{c}"] = np.abs(np.log1p(np.maximum(mean, 0.0)) - np.log1p(np.maximum(y, 0.0))).mean(
            axis=0
        )
    return out.reset_index()


def edge_bins(values: np.ndarray, edges: list[float]) -> tuple[np.ndarray, list[str]]:
    """Label per value from fixed bin edges (left closed), and the labels in order.

    The last bin is open above when its edge is infinite; values outside the
    edges get an empty label.
    """
    v = np.asarray(values, dtype=float)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo:g}+" if not np.isfinite(hi) else f"{lo:g} to {hi:g}")
    idx = np.digitize(v, edges[1:-1])
    inside = np.isfinite(v) & (v >= edges[0]) & (v < edges[-1])
    out = np.where(inside, np.asarray(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)], "")
    return out, labels


def quantile_bins(values: np.ndarray, n_bins: int, digits: int = 2) -> tuple[np.ndarray, list[str]]:
    """Label per value from equal-count bins of the finite values, and the labels in order."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    out = np.full(v.shape, "", dtype=object)
    if ok.sum() < n_bins:
        return out, []
    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return out, []
    labels = [f"{lo:.{digits}f} to {hi:.{digits}f}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    idx = np.clip(np.digitize(v[ok], edges[1:-1]), 0, len(labels) - 1)
    out[ok] = np.asarray(labels, dtype=object)[idx]
    return out, labels


def factor_rows(
    windows: pd.DataFrame,
    class_name: str,
    factor: str,
    bins: np.ndarray,
    order: list[str],
    blocks: np.ndarray,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Error summary of one class in every bin of one factor, with block-bootstrap intervals."""
    rows = []
    for position, label in enumerate(order):
        sel = bins == label
        if not sel.any():
            continue
        mae = block_bootstrap_mean(windows.loc[sel, f"abs_error_{class_name}"], blocks[sel], n_boot)
        log_error = block_bootstrap_mean(windows.loc[sel, f"log_error_{class_name}"], blocks[sel], n_boot)
        rows.append(
            {
                "class_name": class_name,
                "factor": factor,
                "bin": label,
                "order": position,
                "n_windows": int(sel.sum()),
                "n_blocks": int(mae["n_blocks"]),
                "mean_count": float(windows.loc[sel, f"y_{class_name}"].mean()),
                "mean_prediction": float(windows.loc[sel, f"mean_{class_name}"].mean()),
                "bias": float(windows.loc[sel, f"error_{class_name}"].mean()),
                "mae": mae["value"],
                "mae_ci_low": mae["ci_low"],
                "mae_ci_high": mae["ci_high"],
                "log_error": log_error["value"],
                "log_error_ci_low": log_error["ci_low"],
                "log_error_ci_high": log_error["ci_high"],
            }
        )
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; NaN when either series is constant or shorter than three."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def tracking_rows(
    windows: pd.DataFrame, class_name: str, blocks: np.ndarray, n_boot: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Rank correlation between the predicted and the true count of one class within every fold.

    The correlation is taken over the test windows of a fold, so it measures
    whether the prediction follows the count along a tow whatever its level.
    Intervals come from time blocks resampled within the fold; the last row
    is the mean over the folds in which the correlation is defined.
    """
    rng = np.random.default_rng(seed)
    y_all = windows[f"y_{class_name}"].to_numpy(dtype=float)
    p_all = windows[f"mean_{class_name}"].to_numpy(dtype=float)
    folds = windows["fold"].to_numpy()
    rows, values, draws = [], [], []
    for fold in sorted(set(folds)):
        sel = np.flatnonzero(folds == fold)
        value = _spearman(p_all[sel], y_all[sel])
        labels, inverse = np.unique(blocks[sel], return_inverse=True)
        members = [sel[inverse == j] for j in range(len(labels))]
        stat = np.full(n_boot, np.nan)
        for i in range(n_boot):
            idx = np.concatenate([members[j] for j in rng.integers(0, len(labels), len(labels))])
            stat[i] = _spearman(p_all[idx], y_all[idx])
        finite = stat[np.isfinite(stat)]
        rows.append(
            {
                "class_name": class_name,
                "fold": fold,
                "n_windows": int(len(sel)),
                "n_present": int((y_all[sel] > 0).sum()),
                "spearman": value,
                "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
                "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
            }
        )
        if np.isfinite(value):
            values.append(value)
            draws.append(stat)
    if values:
        with np.errstate(invalid="ignore"):
            stat = np.nanmean(np.stack(draws), axis=0)
        finite = stat[np.isfinite(stat)]
        rows.append(
            {
                "class_name": class_name,
                "fold": "mean over folds",
                "n_windows": int(len(windows)),
                "n_present": int((y_all > 0).sum()),
                "spearman": float(np.mean(values)),
                "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
                "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
            }
        )
    return rows


def paired_errors(
    first: pd.DataFrame,
    second: pd.DataFrame,
    classes: list[str],
    block_s: float,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Errors of two configurations on the test windows that they share, and their paired difference.

    Both tables hold one row per test window with ``window_id``, ``transect``
    and the absolute error per class; they must come from the same window
    dataset. Two protocols that test a window after training on different
    data are compared this way on identical windows. The interval of the
    difference (first minus second) resamples time blocks.
    """
    both = first.merge(second, on="window_id", suffixes=("", "_other"))
    if both.empty:
        return []
    if "time_camera" in both.columns:
        block = pd.to_datetime(both["time_camera"]).astype("int64") // int(block_s * 1e9)
        blocks = (both["transect"].astype(str) + "|" + block.astype(str)).to_numpy()
    else:
        blocks = both["window_id"].astype(str).to_numpy()
    present = [
        c for c in classes if f"abs_error_{c}" in both.columns and f"abs_error_{c}_other" in both.columns
    ]
    series = {
        c: (both[f"abs_error_{c}"].to_numpy(), both[f"abs_error_{c}_other"].to_numpy()) for c in present
    }
    if len(present) > 1:
        series["mean over classes"] = (
            np.mean([series[c][0] for c in present], axis=0),
            np.mean([series[c][1] for c in present], axis=0),
        )
    rows = []
    for name, (a, b) in series.items():
        d = block_bootstrap_mean(a - b, blocks, n_boot)
        rows.append(
            {
                "class_name": name,
                "n_windows": int(len(both)),
                "n_blocks": int(d["n_blocks"]),
                "mae": float(a.mean()),
                "mae_other": float(b.mean()),
                "difference": d["value"],
                "ci_low": d["ci_low"],
                "ci_high": d["ci_high"],
            }
        )
    return rows


def _fold_windows(
    root: Path,
    experiment: str,
    protocol: str,
    classes: list[str],
    targets: pd.DataFrame,
    truth_form: str = TRUTH_FORM,
) -> pd.DataFrame:
    """Seed-averaged window errors of every fold of one experiment against its truth."""
    by_id = targets.set_index("window_id")
    parts = []
    for fold_dir in sorted(p for p in (root / experiment / protocol).glob("*") if p.is_dir()):
        tables = []
        for path in sorted(fold_dir.glob("seed*/predictions_test.parquet")):
            table = pd.read_parquet(path)
            for c in classes:
                table[f"y_{c}"] = by_id[f"{truth_form}_{c}_at_lag"].reindex(table["window_id"]).to_numpy()
            tables.append(table)
        if not tables:
            continue
        errors = window_errors(tables, classes)
        errors["fold"] = fold_dir.name
        errors["n_seeds"] = len(tables)
        parts.append(errors)
    if not parts:
        raise FileNotFoundError(f"no test predictions of {experiment} under {root}")
    return pd.concat(parts, ignore_index=True)


def run_analyse_errors(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Tabulate and draw the errors of the named configurations against the window properties."""
    ecfg = cfg.get("errors", {})
    evaluation = cfg.get("evaluation", {})
    experiments = list(ecfg.get("experiments", [evaluation.get("reference", "cnn")]))
    block_s = float(evaluation.get("block_s", 300))
    n_boot = int(evaluation.get("n_boot", 500))
    depth_edges = [float(e) for e in ecfg.get("depth_edges_m", [0, 50, 100, 200, 300, 500])]
    hour_edges = [float(e) for e in ecfg.get("hour_edges", [0, 6, 12, 18, 24])]
    count_edges = [float(e) for e in ecfg.get("count_edges", [0, 1, 10, 100, 1000, float("inf")])]
    n_quantiles = int(ecfg.get("n_quantile_bins", 3))
    formats = list(ecfg.get("formats", ["pdf", "png"]))
    root = resolve_path(cfg, "project_dir", "runs", "acoustic")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tracking: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for experiment in experiments:
        exp = experiment_config(cfg, experiment)
        protocol = str(exp.get("protocol", "loto_unflagged"))
        label_classes = list(exp.get("targets", {}).get("classes", []))
        classes = [c for c in ecfg.get("classes", label_classes) if c in label_classes]
        _, meta_dir, _ = _window_paths(
            cfg, str(exp.get("geometry", "base")), str(exp.get("windows_variant", ""))
        )
        meta = pd.read_parquet(Path(meta_dir, "meta.parquet"))
        targets = pd.read_parquet(Path(meta_dir, "targets.parquet"))
        truth_form = str(exp.get("targets", {}).get("truth_form", TRUTH_FORM))
        windows = _fold_windows(root, experiment, protocol, classes, targets, truth_form).merge(
            meta, on="window_id", how="left"
        )
        by_id = targets.set_index("window_id")
        if "time_camera" in windows.columns:
            stamp = pd.to_datetime(windows["time_camera"])
            windows["hour"] = stamp.dt.hour + stamp.dt.minute / 60.0
            block = stamp.astype("int64") // int(block_s * 1e9)
            blocks = (windows["transect"].astype(str) + "|" + block.astype(str)).to_numpy()
        else:
            blocks = windows["window_id"].astype(str).to_numpy()
        if {"lag_lo_s", "lag_hi_s"} <= set(windows.columns):
            windows["lag_support_s"] = windows["lag_hi_s"] - windows["lag_lo_s"]
        for c in classes:
            weighted = f"weighted_{c}_at_lag"
            if weighted in by_id.columns:
                hard = by_id[f"{TRUTH_FORM}_{c}_at_lag"].reindex(windows["window_id"]).to_numpy(dtype=float)
                soft = by_id[weighted].reindex(windows["window_id"]).to_numpy(dtype=float)
                windows[f"confidence_{c}"] = np.where(hard > 0, soft / np.maximum(hard, 1e-9), np.nan)
        shared: list[tuple[str, np.ndarray, list[str]]] = []
        transects = sorted(windows["transect"].astype(str).unique())
        shared.append(("transect", windows["transect"].astype(str).to_numpy(), transects))
        if "camera_depth_m" in windows.columns:
            shared.append(("camera_depth_m", *edge_bins(windows["camera_depth_m"].to_numpy(), depth_edges)))
        if "hour" in windows.columns:
            shared.append(("hour_utc", *edge_bins(windows["hour"].to_numpy(), hour_edges)))
        if "lag_support_s" in windows.columns:
            shared.append(
                ("lag_support_s", *quantile_bins(windows["lag_support_s"].to_numpy(), n_quantiles, digits=0))
            )
        table: list[dict[str, Any]] = []
        for c in classes:
            for factor, bins, order in shared:
                table += factor_rows(windows, c, factor, bins, order, blocks, n_boot)
            bins, order = edge_bins(windows[f"y_{c}"].to_numpy(), count_edges)
            table += factor_rows(windows, c, "true_count", bins, order, blocks, n_boot)
            if f"confidence_{c}" in windows.columns:
                confidence = windows[f"confidence_{c}"].to_numpy(dtype=float)
                bins, order = quantile_bins(confidence, n_quantiles)
                bins = np.where(np.isfinite(confidence), bins, ABSENT)
                table += factor_rows(windows, c, "label_confidence", bins, [ABSENT, *order], blocks, n_boot)
        rows += [{"experiment": experiment, "protocol": protocol, **r} for r in table]
        for c in classes:
            tracking += [
                {"experiment": experiment, "protocol": protocol, **r}
                for r in tracking_rows(windows, c, blocks, n_boot)
            ]
        windows.to_parquet(out_dir / f"windows_{experiment}.parquet", index=False)
        outputs.append(out_dir / f"windows_{experiment}.parquet")
        frame = pd.DataFrame(table)
        for fmt in formats:
            outputs.append(
                error_factor_plot(
                    frame,
                    classes,
                    out_dir / f"errors_{experiment}.{fmt}",
                    title=f"{experiment}: absolute error in log(1 + count) against the window properties",
                )
            )
        log.info("%s: %d windows, %d rows", experiment, len(windows), len(table))
    pd.DataFrame(rows).to_csv(out_dir / "errors_by_factor.csv", index=False)
    pd.DataFrame(tracking).to_csv(out_dir / "tracking.csv", index=False)
    # Pairs of configurations compared on the windows that both were tested on.
    paired: list[dict[str, Any]] = []
    for a, b in ecfg.get("paired", []):
        pa, pb = out_dir / f"windows_{a}.parquet", out_dir / f"windows_{b}.parquet"
        if pa.is_file() and pb.is_file():
            shared = [c for c in ecfg.get("classes", []) or []]
            paired += [
                {"experiment": a, "other": b, **r}
                for r in paired_errors(pd.read_parquet(pa), pd.read_parquet(pb), shared, block_s, n_boot)
            ]
    pd.DataFrame(paired).to_csv(out_dir / "paired_errors.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"experiments": experiments, "block_s": block_s, "n_boot": n_boot}, fh, indent=2)
    write_manifest(
        out_dir,
        "analyse-errors",
        cfg,
        {"runs": root},
        [
            out_dir / "errors_by_factor.csv",
            out_dir / "tracking.csv",
            out_dir / "paired_errors.csv",
            out_dir / "summary.json",
            *outputs,
        ],
    )
    return out_dir
