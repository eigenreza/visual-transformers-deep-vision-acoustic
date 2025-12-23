"""Metrics for predicted count distributions and presence.

Count metrics are computed in count space from the predictive mean (mean
absolute error, root mean squared error, bias) and from the predictive
distribution (continuous ranked probability score from samples, log score).
Presence is the probability of a positive count under the distribution,
scored threshold-free by average precision and at a threshold chosen on
validation by F1, with a reliability table. Predictive intervals are scored
by coverage and width, and split conformal prediction adjusts them on
validation. Uncertainty comes from a paired bootstrap over windows, which
keeps model comparisons paired on the same windows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..models.counts import crps_from_samples, log_score, sample


def count_errors(y: np.ndarray, mean: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window absolute error, squared error and signed error, shape (N, K) each."""
    diff = mean - y
    return {"abs_error": np.abs(diff), "sq_error": diff**2, "error": diff}


def presence_probability(
    family: str, params: dict[str, np.ndarray], n_samples: int = 400, seed: int = 0, **extra: Any
) -> np.ndarray:
    """Probability of a positive count per window and class."""
    if family == "poisson":
        return 1.0 - np.exp(-np.exp(params["log_mu"]))
    if family == "nb":
        mu, alpha = np.exp(params["log_mu"]), np.exp(params["log_alpha"])
        r = 1.0 / alpha
        return 1.0 - (r / (r + mu)) ** r
    draws = sample(family, params, n_samples, np.random.default_rng(seed), **extra)
    return (draws > 0.5).mean(axis=0)


def average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    """Area under the precision-recall curve (all-point interpolation) of one class; NaN without positives."""
    truth = truth.astype(bool)
    if truth.sum() == 0 or len(score) == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(truth[order])
    fp = np.cumsum(~truth[order])
    recall = tp / truth.sum()
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def f1_at(score: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "threshold": float(threshold)}


def best_f1_threshold(score: np.ndarray, truth: np.ndarray) -> float:
    """Threshold on the presence probability that maximises F1; 0.5 without positives."""
    truth = truth.astype(bool)
    if truth.sum() == 0:
        return 0.5
    candidates = np.unique(np.round(score, 3))
    best, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_at(score, truth, t)["f1"]
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def reliability(prob: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Reliability table of presence probabilities and the expected calibration error."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        conf, acc = float(prob[sel].mean()), float(truth[sel].mean())
        rows.append(
            {
                "bin": b,
                "lower": edges[b],
                "upper": edges[b + 1],
                "n": int(sel.sum()),
                "confidence": conf,
                "frequency": acc,
            }
        )
        ece += sel.mean() * abs(conf - acc)
    return pd.DataFrame(rows), float(ece)


def interval_scores(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window coverage indicator and width of central predictive intervals."""
    return {"covered": ((y >= lower) & (y <= upper)).astype(float), "width": upper - lower}


def conformal_adjustment(
    lower_val: np.ndarray, upper_val: np.ndarray, y_val: np.ndarray, alpha: float = 0.1
) -> np.ndarray:
    """Per-class split conformal correction for quantile intervals (conformalised quantile regression)."""
    scores = np.maximum(lower_val - y_val, y_val - upper_val)
    n = scores.shape[0]
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores, axis=0)[k - 1]


def bootstrap_mean(values: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Mean of per-window values with a percentile interval over resampled windows."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {
        "value": float(v.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n": int(v.size),
    }


def block_bootstrap_mean(
    values: np.ndarray, blocks: np.ndarray, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """Mean of per-window values with a percentile interval from resampling whole blocks of windows.

    ``blocks`` gives a label per window; windows that share a label (a time
    block of one fold) are drawn together, which keeps the dependence
    between neighbouring, overlapping windows inside every resample.
    """
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks)
    ok = np.isfinite(v)
    v, b = v[ok], b[ok]
    if v.size == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_blocks": 0}
    labels, inverse = np.unique(b, return_inverse=True)
    sums = np.bincount(inverse, weights=v, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    stat = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {
        "value": float(v.mean()),
        "ci_low": float(np.percentile(stat, 2.5)),
        "ci_high": float(np.percentile(stat, 97.5)),
        "n": int(v.size),
        "n_blocks": int(len(labels)),
    }


def fold_block_bootstrap(
    values: np.ndarray, blocks: np.ndarray, folds: np.ndarray, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """Mean over folds of the fold means of per-window values, with a percentile interval.

    Whole blocks of windows are resampled within every fold, the fold means
    of a resample are averaged with equal weight, and the interval is read
    from the resampled means, so the value and its interval describe the
    same quantity. ``blocks`` and ``folds`` give a label per window.
    """
    v = np.asarray(values, dtype=float)
    b, f = np.asarray(blocks), np.asarray(folds)
    empty = {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if v.size == 0 or v.size != b.size or v.size != f.size:
        return {**empty, "n": 0, "n_blocks": 0, "n_folds": 0}
    ok = np.isfinite(v)
    v, b, f = v[ok], b[ok], f[ok]
    if v.size == 0:
        return {**empty, "n": 0, "n_blocks": 0, "n_folds": 0}
    rng = np.random.default_rng(seed)
    fold_values, fold_stats, n_blocks = [], [], 0
    for name in np.unique(f):
        sel = f == name
        labels, inverse = np.unique(b[sel], return_inverse=True)
        sums = np.bincount(inverse, weights=v[sel], minlength=len(labels))
        counts = np.bincount(inverse, minlength=len(labels)).astype(float)
        draw = rng.integers(0, len(labels), size=(n_boot, len(labels)))
        fold_stats.append(sums[draw].sum(axis=1) / counts[draw].sum(axis=1))
        fold_values.append(float(v[sel].mean()))
        n_blocks += len(labels)
    stat = np.mean(fold_stats, axis=0)
    return {
        "value": float(np.mean(fold_values)),
        "ci_low": float(np.percentile(stat, 2.5)),
        "ci_high": float(np.percentile(stat, 97.5)),
        "n": int(v.size),
        "n_blocks": int(n_blocks),
        "n_folds": len(fold_values),
    }


def bootstrap_statistic(
    fn: Callable[[np.ndarray], float], n: int, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """Statistic of a window sample with a percentile interval; ``fn`` receives resampled window indices."""
    rng = np.random.default_rng(seed)
    value = fn(np.arange(n))
    draws = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    return {
        "value": float(value),
        "ci_low": float(np.percentile(draws, 2.5)) if draws.size else float("nan"),
        "ci_high": float(np.percentile(draws, 97.5)) if draws.size else float("nan"),
        "n": int(n),
    }


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """Difference of the means of two per-window series (b minus a) over the same resampled windows."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = (b[idx] - a[idx]).mean(axis=1)
    return {
        "difference": float((b - a).mean()),
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "n": int(a.size),
    }


PARAMETER_PREFIXES = ("log_mu", "log_alpha", "logit_pi", "log1p_mu")


def parameter_names(columns: Any, classes: list[str]) -> list[str]:
    """Distribution parameters present in a prediction table, read against the known classes.

    Class names may contain underscores (a merged class does), so a parameter
    is recognised by its prefix followed by a known class, never by splitting
    the column name.
    """
    present = set(columns)
    return [p for p in PARAMETER_PREFIXES if classes and f"{p}_{classes[0]}" in present]


def evaluate_predictions(
    preds: pd.DataFrame,
    classes: list[str],
    family: str,
    thresholds: dict[str, float] | None = None,
    conformal: dict[str, float] | None = None,
    n_samples: int = 400,
    seed: int = 0,
    extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-class metrics of one predictions table and the per-window contributions behind them.

    ``thresholds`` (presence probability per class) and ``conformal``
    (interval correction per class) come from validation; without them the
    thresholds are chosen on the table itself and the intervals are the raw
    quantiles. The returned contributions hold, per class, the absolute
    error, squared error, CRPS, log score, coverage and width per window, for
    bootstrapping and paired comparisons.
    """
    extra = extra or {}
    y = np.stack([preds[f"y_{c}_at_lag"].to_numpy(dtype=float) for c in classes], axis=1)
    mean = np.stack([preds[f"mean_{c}"].to_numpy(dtype=float) for c in classes], axis=1)
    param_names = parameter_names(preds.columns, classes)
    params = {
        p: np.stack([preds[f"{p}_{c}"].to_numpy(dtype=float) for c in classes], axis=1) for p in param_names
    }
    rng = np.random.default_rng(seed)
    draws = sample(
        family,
        params,
        n_samples,
        rng,
        **{k: v for k, v in extra.items() if k in ("tweedie_power", "tweedie_phi", "log1p_sigma")},
    )
    crps = crps_from_samples(draws, y)
    ls = log_score(family, params, y, tweedie_power=extra.get("tweedie_power", 1.5))
    prob = presence_probability(
        family,
        params,
        n_samples,
        seed,
        **{k: v for k, v in extra.items() if k in ("tweedie_power", "tweedie_phi", "log1p_sigma")},
    )
    lower = np.stack([preds[f"q05_{c}"].to_numpy(dtype=float) for c in classes], axis=1)
    upper = np.stack([preds[f"q95_{c}"].to_numpy(dtype=float) for c in classes], axis=1)
    if conformal:
        adj = np.array([conformal.get(c, 0.0) for c in classes])
        lower, upper = np.maximum(lower - adj, 0.0), upper + adj
    errors = count_errors(y, mean)
    intervals = interval_scores(lower, upper, y)
    rows = []
    contributions: dict[str, np.ndarray] = {
        "abs_error": errors["abs_error"],
        "sq_error": errors["sq_error"],
        "crps": crps,
        "log_score": ls,
        "covered": intervals["covered"],
        "width": intervals["width"],
        "presence_prob": prob,
        "presence_truth": (y > 0).astype(float),
    }
    for k, c in enumerate(classes):
        truth = y[:, k] > 0
        thr = (
            thresholds.get(c, best_f1_threshold(prob[:, k], truth))
            if thresholds
            else best_f1_threshold(prob[:, k], truth)
        )
        f1 = f1_at(prob[:, k], truth, thr)
        _, ece = reliability(prob[:, k], truth.astype(float))
        rows.append(
            {
                "class_name": c,
                "n_windows": int(len(y)),
                "n_present": int(truth.sum()),
                "mean_count": float(y[:, k].mean()),
                "mae": float(errors["abs_error"][:, k].mean()),
                "rmse": float(np.sqrt(errors["sq_error"][:, k].mean())),
                "bias": float(errors["error"][:, k].mean()),
                "crps": float(crps[:, k].mean()),
                "log_score": float(ls[:, k].mean()),
                "presence_ap": average_precision(prob[:, k], truth),
                "presence_f1": f1["f1"],
                "presence_precision": f1["precision"],
                "presence_recall": f1["recall"],
                "presence_threshold": f1["threshold"],
                "presence_ece": ece,
                "coverage90": float(intervals["covered"][:, k].mean()),
                "width90": float(intervals["width"][:, k].mean()),
                "mae_mean_predictor": float(np.abs(y[:, k] - y[:, k].mean()).mean()),
            }
        )
    return pd.DataFrame(rows), contributions


def macro_over_classes(table: pd.DataFrame, classes: list[str], columns: tuple[str, ...]) -> dict[str, float]:
    """Mean of the given metric columns over the named classes with at least one present window."""
    sub = table[table["class_name"].isin(classes) & (table["n_present"] > 0)]
    return {col: float(sub[col].mean()) if len(sub) else float("nan") for col in columns}
