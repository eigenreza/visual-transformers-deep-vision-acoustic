"""Stage 9 command: evaluate every trained acoustic configuration and compare them.

For every run (experiment, protocol, fold, seed) with predictions, the
presence thresholds and the conformal corrections are chosen on the
validation windows and applied to the test windows; per-class metrics are
written per run. The runs are then aggregated per experiment and protocol:
means over folds, spread across seeds, and bootstrap intervals from time
blocks resampled within folds. Experiments are compared with the reference,
and named pairs of experiments with each other, by paired differences on the
same windows, and macro means are reported with and without krill and with
the separate against the merged herring and blue whiting targets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .eval.metrics import (
    average_precision,
    best_f1_threshold,
    conformal_adjustment,
    evaluate_predictions,
    fold_block_bootstrap,
    paired_bootstrap_difference,
    parameter_names,
    presence_probability,
)
from .manifest import write_manifest
from .models_stage import _window_paths, experiment_config

log = logging.getLogger(__name__)

DECOMPOSABLE = ("abs_error", "crps", "log_score", "covered", "width")
METRIC_COLUMNS = (
    "mae",
    "rmse",
    "bias",
    "crps",
    "log_score",
    "presence_ap",
    "presence_f1",
    "presence_ece",
    "coverage90",
    "width90",
)


TRUTH_FORM = "hard"
# Families whose score is the negative log probability of a count; the log score of the
# others (a deviance, a squared error in log space) is not comparable and is left empty.
LIKELIHOOD_FAMILIES = ("poisson", "nb", "zinb")


def window_truth(
    cfg: dict[str, Any], experiment: str, cache: dict[tuple[str, str], Any]
) -> pd.DataFrame | None:
    """Truth and pairing key per window of the dataset an experiment was trained on.

    Every configuration is scored against the same quantity, the resolved
    object counts in the window's camera span at the centre of the lag
    distribution, whatever target form it was trained on, unless its target
    configuration names another ``truth_form`` (a configuration trained to
    predict another quantity is scored on that quantity); windows of
    different datasets are paired by transect and camera time, since a
    window identifier is only an index within its own dataset. Returns a
    table indexed by ``window_id`` with ``window_key`` and one
    ``y_<class>_at_lag`` column per class, or None when the dataset is not
    on disk.
    """
    try:
        exp = experiment_config(cfg, experiment)
    except KeyError:
        return None
    form = str(exp.get("targets", {}).get("truth_form", TRUTH_FORM))
    key = (str(exp.get("geometry", "base")), str(exp.get("windows_variant", "")), form)
    if key in cache:
        return cache[key]
    try:
        _, meta_dir, _ = _window_paths(cfg, key[0], key[1])
    except KeyError:
        cache[key] = None
        return None
    meta_path, targets_path = Path(meta_dir, "meta.parquet"), Path(meta_dir, "targets.parquet")
    if not (meta_path.is_file() and targets_path.is_file()):
        cache[key] = None
        return None
    meta = pd.read_parquet(meta_path)
    targets = pd.read_parquet(targets_path)
    table = pd.DataFrame(index=pd.Index(meta["window_id"].to_numpy(), name="window_id"))
    stamp = (
        pd.to_datetime(meta["time_camera"]).astype("int64").to_numpy()
        if "time_camera" in meta.columns
        else meta["window_id"].to_numpy()
    )
    table["window_key"] = [f"{t}|{s}" for t, s in zip(meta["transect"].to_numpy(), stamp, strict=True)]
    if "target_span_s" in meta.columns:
        table["target_span_s"] = meta["target_span_s"].to_numpy(dtype=float)
    by_id = targets.set_index("window_id")
    for col in by_id.columns:
        if col.startswith(f"{form}_") and col.endswith("_at_lag"):
            name = col[len(form) + 1 : -len("_at_lag")]
            table[f"y_{name}_at_lag"] = by_id[col].reindex(table.index).to_numpy()
    table["truth_form"] = form
    cache[key] = table
    return table


def apply_truth(preds: pd.DataFrame, truth: pd.DataFrame | None, classes: list[str]) -> pd.DataFrame:
    """Prediction table with its truth columns replaced by the common truth, and the pairing key added."""
    out = preds.copy()
    if truth is None:
        out["window_key"] = out["window_id"].astype(str)
        return out
    rows = truth.reindex(out["window_id"].to_numpy())
    out["window_key"] = rows["window_key"].to_numpy()
    for c in classes:
        col = f"y_{c}_at_lag"
        if col in rows.columns:
            out[col] = rows[col].to_numpy()
    return out


def _runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/*/*/seed*/predictions_test.parquet"))


def _validation_choices(
    preds_val: pd.DataFrame, classes: list[str], family: str, extra: dict[str, Any], alpha: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Presence thresholds and conformal corrections per class from the validation predictions."""
    y = np.stack([preds_val[f"y_{c}_at_lag"].to_numpy(dtype=float) for c in classes], axis=1)
    names = parameter_names(preds_val.columns, classes)
    params = {
        p: np.stack([preds_val[f"{p}_{c}"].to_numpy(dtype=float) for c in classes], axis=1) for p in names
    }
    prob = presence_probability(
        family,
        params,
        **{k: v for k, v in extra.items() if k in ("tweedie_power", "tweedie_phi", "log1p_sigma")},
    )
    thresholds = {c: best_f1_threshold(prob[:, k], y[:, k] > 0) for k, c in enumerate(classes)}
    lower = np.stack([preds_val[f"q05_{c}"].to_numpy(dtype=float) for c in classes], axis=1)
    upper = np.stack([preds_val[f"q95_{c}"].to_numpy(dtype=float) for c in classes], axis=1)
    adj = conformal_adjustment(lower, upper, y, alpha)
    return thresholds, {c: float(max(adj[k], 0.0)) for k, c in enumerate(classes)}


def evaluate_run(
    run_dir: Path,
    n_samples: int,
    alpha: float,
    force: bool,
    truth: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Metrics table, per-window contributions and the run summary for one run directory."""
    with open(run_dir / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    classes, family, extra = list(summary["classes"]), str(summary["family"]), dict(summary.get("extra", {}))
    eval_dir = run_dir / "eval"
    contrib_path = eval_dir / "contributions_test.npz"
    if not force and (eval_dir / "metrics_test.csv").is_file() and contrib_path.is_file():
        table = pd.read_csv(eval_dir / "metrics_test.csv")
        loaded = np.load(contrib_path, allow_pickle=False)
        return table, {k: loaded[k] for k in loaded.files}, summary
    eval_dir.mkdir(exist_ok=True)
    preds_val = apply_truth(pd.read_parquet(run_dir / "predictions_val.parquet"), truth, classes)
    preds_test = apply_truth(pd.read_parquet(run_dir / "predictions_test.parquet"), truth, classes)
    thresholds, conformal = _validation_choices(preds_val, classes, family, extra, alpha)
    table, contributions = evaluate_predictions(
        preds_test, classes, family, thresholds, conformal, n_samples, 0, extra
    )
    table_val, _ = evaluate_predictions(preds_val, classes, family, None, None, n_samples, 0, extra)
    table.to_csv(eval_dir / "metrics_test.csv", index=False)
    table_val.to_csv(eval_dir / "metrics_val.csv", index=False)
    contributions = {
        **contributions,
        "window_id": preds_test["window_id"].to_numpy(),
        "window_key": preds_test["window_key"].to_numpy().astype(str),
    }
    np.savez_compressed(contrib_path, **contributions)
    with open(eval_dir / "choices.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"presence_thresholds": thresholds, "conformal_adjustment": conformal, "alpha": alpha},
            fh,
            indent=2,
        )
    return table, contributions, summary


def run_evaluate_acoustic(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Evaluate all acoustic runs and write the aggregated tables and comparisons."""
    ecfg = cfg.get("evaluation", {})
    n_samples = int(ecfg.get("n_samples", 400))
    alpha = float(ecfg.get("interval_alpha", 0.1))
    n_boot = int(ecfg.get("n_boot", 500))
    block_s = float(ecfg.get("block_s", 300))
    reference = str(ecfg.get("reference", "cnn"))
    label_classes = list(cfg.get("defaults", {}).get("targets", {}).get("classes", []))
    merged = [c for c in label_classes if "_" in c and all(part in label_classes for part in c.split("_"))]
    root = resolve_path(cfg, "project_dir", "runs", "acoustic")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, contrib_store = [], {}
    truth_cache: dict[tuple[str, str, str], Any] = {}
    spans: dict[str, float] = {}
    truth_forms: dict[str, str] = {}
    for run_dir in _runs(root):
        try:
            truth = window_truth(cfg, run_dir.parents[2].name, truth_cache)
            table, contributions, summary = evaluate_run(run_dir, n_samples, alpha, force, truth)
        except Exception as exc:  # pragma: no cover
            log.warning("evaluation of %s failed: %s", run_dir, exc)
            continue
        key = (summary["experiment"], summary["protocol"], summary["fold"], int(summary["seed"]))
        if str(summary["family"]) not in LIKELIHOOD_FAMILIES:
            table = table.assign(log_score=np.nan)
            contributions = {
                **contributions,
                "log_score": np.full(np.shape(contributions["log_score"]), np.nan),
            }
        if truth is not None and "target_span_s" in truth.columns and len(truth):
            spans[key[0]] = float(truth["target_span_s"].iloc[0])
        if truth is not None and len(truth):
            truth_forms[key[0]] = str(truth["truth_form"].iloc[0])
        contrib_store[key] = (list(summary["classes"]), contributions)
        for r in table.to_dict("records"):
            rows.append({"experiment": key[0], "protocol": key[1], "fold": key[2], "seed": key[3], **r})
    by_run = pd.DataFrame(rows)
    by_run.to_csv(out_dir / "metrics_by_run.csv", index=False)
    agg_rows, macro_rows, diff_rows, fold_rows = [], [], [], []
    subsets = {
        "macro_label_classes": [c for c in label_classes if c not in merged],
        "macro_without_krill": [c for c in label_classes if c not in merged and c != "krill"],
    }
    seed_means: dict[tuple[str, str], dict[str, Any]] = {}
    for (experiment, protocol), g in by_run.groupby(["experiment", "protocol"]):
        seed_means[(experiment, protocol)] = _seed_means(contrib_store, experiment, protocol)
        fold_rows += _fold_rows(
            experiment, protocol, seed_means[(experiment, protocol)], subsets, block_s, n_boot
        )
        # Counts per minute of camera time put targets of different spans on one scale.
        per_minute = 60.0 / spans[experiment] if spans.get(experiment) else float("nan")
        for class_name, gc in g.groupby("class_name"):
            fold_means = gc.groupby("seed")[list(METRIC_COLUMNS)].mean()  # mean over folds per seed
            row = {
                "experiment": experiment,
                "protocol": protocol,
                "class_name": class_name,
                "n_folds": int(gc["fold"].nunique()),
                "n_seeds": int(gc["seed"].nunique()),
                "n_windows": int(gc.groupby("fold")["n_windows"].first().sum()),
                "target_span_s": spans.get(experiment, float("nan")),
                "truth_form": truth_forms.get(experiment, TRUTH_FORM),
            }
            for m in METRIC_COLUMNS:
                row[m] = float(fold_means[m].mean())
                row[f"{m}_sd_seeds"] = float(fold_means[m].std(ddof=0)) if len(fold_means) > 1 else 0.0
            pooled, blocks, fold_labels = _pooled(seed_means[(experiment, protocol)], class_name, block_s)
            for name, col in (
                ("mae", "abs_error"),
                ("crps", "crps"),
                ("log_score", "log_score"),
                ("coverage90", "covered"),
            ):
                b = fold_block_bootstrap(pooled.get(col, np.array([])), blocks, fold_labels, n_boot)
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = b["ci_low"], b["ci_high"]
            row["presence_ap_ci_low"], row["presence_ap_ci_high"] = _presence_ap_interval(
                contrib_store, experiment, protocol, class_name, block_s, n_boot
            )
            row["n_blocks"] = int(len(np.unique(blocks))) if len(blocks) else 0
            for m in ("mae", "crps"):
                for suffix in ("", "_ci_low", "_ci_high"):
                    row[f"{m}_per_minute{suffix}"] = row[f"{m}{suffix}"] * per_minute
            agg_rows.append(row)
        for subset_name, subset in (
            ("label_classes", [c for c in label_classes if c not in merged]),
            ("without_krill", [c for c in label_classes if c not in merged and c != "krill"]),
            ("merged_herring_bluewhiting", [c for c in label_classes if c not in ("herring", "bluewhiting")]),
        ):
            present = [c for c in subset if c in set(g["class_name"])]
            if not present:
                continue
            sub = g[g["class_name"].isin(present)]
            per_seed = sub.groupby("seed")[list(METRIC_COLUMNS)].mean()
            parts = [_pooled(seed_means[(experiment, protocol)], c, block_s) for c in present]
            macro_ci: dict[str, float] = {}
            if all(p[0] for p in parts) and len({len(p[1]) for p in parts}) == 1:
                for name, col in (("mae", "abs_error"), ("crps", "crps"), ("coverage90", "covered")):
                    b = fold_block_bootstrap(
                        np.mean([p[0][col] for p in parts], axis=0), parts[0][1], parts[0][2], n_boot
                    )
                    macro_ci[f"{name}_ci_low"], macro_ci[f"{name}_ci_high"] = b["ci_low"], b["ci_high"]
            macro_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "subset": subset_name,
                    "classes": " ".join(present),
                    "target_span_s": spans.get(experiment, float("nan")),
                    "truth_form": truth_forms.get(experiment, TRUTH_FORM),
                    **{m: float(per_seed[m].mean()) for m in METRIC_COLUMNS},
                    **{
                        f"{m}_sd_seeds": float(per_seed[m].std(ddof=0)) if len(per_seed) > 1 else 0.0
                        for m in ("mae", "crps", "presence_ap")
                    },
                    **macro_ci,
                    **{
                        f"{m}_per_minute{suffix}": value * per_minute
                        for m in ("mae", "crps")
                        for suffix, value in (
                            ("", float(per_seed[m].mean())),
                            ("_ci_low", macro_ci.get(f"{m}_ci_low", float("nan"))),
                            ("_ci_high", macro_ci.get(f"{m}_ci_high", float("nan"))),
                        )
                    },
                }
            )
    aggregated = pd.DataFrame(agg_rows)
    aggregated.to_csv(out_dir / "metrics_aggregated.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(out_dir / "macro.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "metrics_by_fold.csv", index=False)
    for key, (classes, contributions) in contrib_store.items():
        experiment, protocol, fold, seed = key
        ref_key = (reference, protocol, fold, seed)
        if experiment == reference or ref_key not in contrib_store:
            continue
        if not _comparable(experiment, reference, spans, truth_forms):
            continue
        ref_classes, ref = contrib_store[ref_key]
        pair_col = "window_key" if "window_key" in ref and "window_key" in contributions else "window_id"
        shared, a_idx, b_idx = np.intersect1d(ref[pair_col], contributions[pair_col], return_indices=True)
        if shared.size == 0:
            continue
        for c in classes:
            if c not in ref_classes:
                continue
            ka, kb = ref_classes.index(c), classes.index(c)
            for name, col in (("mae", "abs_error"), ("crps", "crps"), ("log_score", "log_score")):
                d = paired_bootstrap_difference(ref[col][a_idx, ka], contributions[col][b_idx, kb], n_boot)
                diff_rows.append(
                    {
                        "experiment": experiment,
                        "reference": reference,
                        "protocol": protocol,
                        "fold": fold,
                        "seed": seed,
                        "class_name": c,
                        "metric": name,
                        **d,
                    }
                )
    differences = pd.DataFrame(diff_rows)
    if not differences.empty:
        differences.to_csv(out_dir / "differences_by_run.csv", index=False)
    # Paired differences: per-window values averaged over seeds, paired by window key within each
    # fold, and resampled in time blocks within folds. A pair whose targets cover different camera
    # spans has no common truth and is left out.
    separate = [c for c in label_classes if c not in merged]
    pairs = [(e, reference) for (e, _p) in seed_means if e != reference]
    named = [(str(a), str(b)) for a, b in ecfg.get("contrasts", [])]
    tables: dict[str, list[dict[str, Any]]] = {"differences": [], "contrasts": []}
    for name, wanted in (("differences", pairs), ("contrasts", named)):
        for experiment, other in dict.fromkeys(wanted):
            if not _comparable(experiment, other, spans, truth_forms):
                log.info("no paired difference for %s against %s: the targets differ", experiment, other)
                continue
            for (exp, protocol), means in seed_means.items():
                ref_means = seed_means.get((other, protocol))
                if exp != experiment or ref_means is None:
                    continue
                tables[name] += _pooled_differences(
                    experiment, other, protocol, means, ref_means, separate, block_s, n_boot
                )
    pd.DataFrame(tables["differences"]).to_csv(out_dir / "differences.csv", index=False)
    pd.DataFrame(tables["contrasts"]).to_csv(out_dir / "contrasts.csv", index=False)
    summary = {
        "n_runs": int(len(contrib_store)),
        "experiments": sorted({k[0] for k in contrib_store}),
        "reference": reference,
        "n_samples": n_samples,
        "interval_alpha": alpha,
        "n_boot": n_boot,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_manifest(
        out_dir,
        "evaluate-acoustic",
        cfg,
        {"runs": root},
        [
            out_dir / "metrics_by_run.csv",
            out_dir / "metrics_aggregated.csv",
            out_dir / "macro.csv",
            out_dir / "metrics_by_fold.csv",
            out_dir / "differences.csv",
            out_dir / "contrasts.csv",
            out_dir / "summary.json",
        ],
    )
    log.info("evaluated %d runs", len(contrib_store))
    return out_dir


def _comparable(a: str, b: str, spans: dict[str, float], truth_forms: dict[str, str]) -> bool:
    """Whether two experiments are scored on the same quantity over the same camera span."""
    span_a, span_b = spans.get(a), spans.get(b)
    if span_a is not None and span_b is not None and span_a != span_b:
        return False
    return truth_forms.get(a, TRUTH_FORM) == truth_forms.get(b, TRUTH_FORM)


def _fold_rows(
    experiment: str,
    protocol: str,
    means: dict[str, Any],
    subsets: dict[str, list[str]],
    block_s: float,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Seed-averaged MAE and CRPS of every fold on its own, per class and per class subset.

    The interval of a fold resamples its time blocks; a subset is the mean
    over its classes per window.
    """
    rows: list[dict[str, Any]] = []
    for fold, entry in means.items():
        if not entry["values"]:
            continue
        blocks = _block_labels(entry["keys"], fold, block_s)
        folds = np.full(len(blocks), fold)
        groups = {c: [c] for c in entry["classes"]}
        groups.update(
            {name: [c for c in members if c in entry["classes"]] for name, members in subsets.items()}
        )
        for name, members in groups.items():
            if not members:
                continue
            row: dict[str, Any] = {
                "experiment": experiment,
                "protocol": protocol,
                "fold": fold,
                "class_name": name,
                "n_windows": int(len(blocks)),
            }
            for metric, col in (("mae", "abs_error"), ("crps", "crps")):
                values = np.mean(
                    [entry["values"][col][:, entry["classes"].index(c)] for c in members], axis=0
                )
                b = fold_block_bootstrap(values, blocks, folds, n_boot)
                row.update(
                    {metric: b["value"], f"{metric}_ci_low": b["ci_low"], f"{metric}_ci_high": b["ci_high"]}
                )
            rows.append(row)
    return rows


def _block_labels(keys: np.ndarray, fold: str, block_s: float) -> np.ndarray:
    """Block label per window from its pairing key (transect and camera time in nanoseconds) and its fold."""
    out = []
    for n, key in enumerate(keys):
        transect, _, stamp = str(key).partition("|")
        block = int(int(stamp) // int(block_s * 1e9)) if stamp.lstrip("-").isdigit() else n
        out.append(f"{fold}|{transect}|{block}")
    return np.asarray(out)


def _seed_means(store: dict, experiment: str, protocol: str) -> dict[str, Any]:
    """Per fold: classes, window keys and the per-window contributions averaged over the seeds.

    The seeds of a fold share its test windows, so their errors are averaged
    per window before any resampling; a seed whose windows or classes differ
    from the first one of the fold is left out.
    """
    by_fold: dict[str, list] = {}
    for (exp, prot, fold, _seed), (classes, contributions) in sorted(store.items()):
        if exp == experiment and prot == protocol:
            by_fold.setdefault(fold, []).append((classes, contributions))
    out: dict[str, Any] = {}
    for fold, runs in by_fold.items():
        classes = runs[0][0]
        first = runs[0][1]
        keys = np.asarray(first["window_key"] if "window_key" in first else first["window_id"]).astype(str)
        stacks: dict[str, list[np.ndarray]] = {col: [] for col in DECOMPOSABLE}
        for cls, contributions in runs:
            k = np.asarray(
                contributions["window_key"] if "window_key" in contributions else contributions["window_id"]
            ).astype(str)
            order = pd.Series(np.arange(len(k)), index=k).reindex(keys).to_numpy()
            if cls != classes or np.isnan(order).any():
                continue
            order = order.astype(int)
            for col in DECOMPOSABLE:
                stacks[col].append(np.asarray(contributions[col], dtype=float)[order])
        out[fold] = {
            "classes": classes,
            "keys": keys,
            "values": {col: np.mean(v, axis=0) for col, v in stacks.items() if v},
            "n_seeds": len(stacks[DECOMPOSABLE[0]]),
        }
    return out


def _presence_ap_interval(
    store: dict, experiment: str, protocol: str, class_name: str, block_s: float, n_boot: int, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of the presence average precision, mean over seeds and folds.

    Time blocks are resampled within every fold; on each resample the average
    precision is computed per seed and averaged, and the fold values are
    averaged over the folds that hold a present window in the resample.
    """
    by_fold: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for (exp, prot, fold, _seed), (classes, contributions) in sorted(store.items()):
        if exp != experiment or prot != protocol or class_name not in classes:
            continue
        if "presence_prob" not in contributions:
            continue
        k = classes.index(class_name)
        keys = np.asarray(
            contributions["window_key"] if "window_key" in contributions else contributions["window_id"]
        ).astype(str)
        by_fold.setdefault(fold, []).append(
            (
                keys,
                np.asarray(contributions["presence_prob"], dtype=float)[:, k],
                np.asarray(contributions["presence_truth"], dtype=float)[:, k] > 0,
            )
        )
    rng = np.random.default_rng(seed)
    fold_draws = []
    for fold, runs in by_fold.items():
        keys = runs[0][0]
        runs = [r for r in runs if len(r[0]) == len(keys) and (r[0] == keys).all()]
        labels, inverse = np.unique(_block_labels(keys, fold, block_s), return_inverse=True)
        members = [np.flatnonzero(inverse == j) for j in range(len(labels))]
        draws = np.full(n_boot, np.nan)
        for i in range(n_boot):
            idx = np.concatenate([members[j] for j in rng.integers(0, len(labels), len(labels))])
            values = np.array([average_precision(prob[idx], truth[idx]) for _, prob, truth in runs])
            if np.isfinite(values).any():
                draws[i] = float(np.nanmean(values))
        fold_draws.append(draws)
    if not fold_draws:
        return float("nan"), float("nan")
    stack = np.stack(fold_draws)
    keep = np.isfinite(stack).any(axis=0)
    if not keep.any():
        return float("nan"), float("nan")
    stat = np.nanmean(stack[:, keep], axis=0)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def _pooled(
    means: dict[str, Any], class_name: str, block_s: float
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Seed-averaged contributions of one class over all folds, with the block and fold label per window."""
    out: dict[str, list[np.ndarray]] = {}
    blocks: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    for fold, entry in means.items():
        if class_name not in entry["classes"] or not entry["values"]:
            continue
        k = entry["classes"].index(class_name)
        for col, values in entry["values"].items():
            out.setdefault(col, []).append(values[:, k])
        blocks.append(_block_labels(entry["keys"], fold, block_s))
        folds.append(np.full(len(entry["keys"]), fold))
    if not blocks:
        return {}, np.array([]), np.array([])
    return (
        {col: np.concatenate(v) for col, v in out.items()},
        np.concatenate(blocks),
        np.concatenate(folds),
    )


def _pooled_differences(
    experiment: str,
    reference: str,
    protocol: str,
    means: dict[str, Any],
    ref_means: dict[str, Any],
    separate_classes: list[str],
    block_s: float,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Paired differences (experiment minus reference) per class and as macro means, with block intervals."""
    per_metric: dict[str, dict[str, list[np.ndarray]]] = {}
    blocks: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    for fold, entry in means.items():
        ref = ref_means.get(fold)
        if ref is None or not entry["values"] or not ref["values"]:
            continue
        shared, a_idx, b_idx = np.intersect1d(ref["keys"], entry["keys"], return_indices=True)
        if shared.size == 0:
            continue
        blocks.append(_block_labels(shared, fold, block_s))
        folds.append(np.full(len(shared), fold))
        for name, col in (("mae", "abs_error"), ("crps", "crps"), ("log_score", "log_score")):
            store = per_metric.setdefault(name, {})
            for c in entry["classes"]:
                if c not in ref["classes"]:
                    continue
                d = (
                    entry["values"][col][b_idx, entry["classes"].index(c)]
                    - ref["values"][col][a_idx, ref["classes"].index(c)]
                )
                store.setdefault(c, []).append(d)
    rows: list[dict[str, Any]] = []
    if not blocks:
        return rows
    block_labels, fold_labels = np.concatenate(blocks), np.concatenate(folds)
    for name, by_class in per_metric.items():
        full = {
            c: np.concatenate(v) for c, v in by_class.items() if sum(len(x) for x in v) == len(block_labels)
        }
        macros = {
            "macro_label_classes": [c for c in separate_classes if c in full],
            "macro_without_krill": [c for c in separate_classes if c in full and c != "krill"],
        }
        series = dict(full)
        for label, members in macros.items():
            if members:
                series[label] = np.mean([full[c] for c in members], axis=0)
        for c, d in series.items():
            b = fold_block_bootstrap(d, block_labels, fold_labels, n_boot)
            if not np.isfinite(b["value"]):
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "reference": reference,
                    "protocol": protocol,
                    "class_name": c,
                    "metric": name,
                    "difference": b["value"],
                    "ci_low": b["ci_low"],
                    "ci_high": b["ci_high"],
                    "n_windows": b["n"],
                    "n_blocks": b["n_blocks"],
                    "n_folds": b["n_folds"],
                }
            )
    return rows
