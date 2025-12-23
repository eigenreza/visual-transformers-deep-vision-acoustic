"""Confidence calibration of detections.

A detection's calibrated confidence is an estimate of the probability that
it is a true positive at the matching threshold. Three monotone maps of the
raw score are fitted on validation detections labelled by matching: a single
temperature on the logit, Platt scaling (slope and intercept on the logit) and
isotonic regression. Reliability tables and the expected calibration error
are reported per slice of images, so that a difference between years is
visible, and the maps can be refitted on one slice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from ..manifest import write_manifest
from .evaluate import (
    best_thresholds,
    gt_table,
    image_table,
    load_coco,
    match_detections,
    precision_recall_curves,
)

log = logging.getLogger(__name__)

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on the confidence."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "count": int(m.sum()),
                "mean_confidence": float(p[m].mean()) if m.any() else np.nan,
                "fraction_positive": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


class Calibrator:
    """Monotone map from raw score to calibrated confidence."""

    def __init__(self, method: str, params: dict[str, Any]) -> None:
        self.method = method
        self.params = params

    @classmethod
    def fit(cls, method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        z = logit(scores)
        y = np.asarray(y, dtype=float)
        if method == "identity":
            return cls(method, {})
        if method == "temperature":
            res = minimize(
                lambda t: nll(sigmoid(z / t[0]), y), x0=[1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B"
            )
            return cls(method, {"temperature": float(res.x[0])})
        if method == "platt":
            res = minimize(lambda ab: nll(sigmoid(ab[0] * z + ab[1]), y), x0=[1.0, 0.0], method="L-BFGS-B")
            return cls(method, {"slope": float(res.x[0]), "intercept": float(res.x[1])})
        if method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(scores, y)
            return cls(method, {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        raise ValueError(f"unknown calibration method {method!r}")

    def apply(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.method == "identity":
            return s
        if self.method == "temperature":
            return sigmoid(logit(s) / self.params["temperature"])
        if self.method == "platt":
            return sigmoid(self.params["slope"] * logit(s) + self.params["intercept"])
        if self.method == "isotonic":
            return np.interp(s, self.params["x"], self.params["y"])
        raise ValueError(f"unknown calibration method {self.method!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibrator:
        return cls(d["method"], d["params"])


def labelled_detections(preds: pd.DataFrame, coco: dict[str, Any], iou_threshold: float) -> pd.DataFrame:
    """Predictions with a true-positive flag and the year of their image."""
    gts = gt_table(coco)
    images = image_table(coco)
    matched = match_detections(preds, gts[gts["iscrowd"] == 0], iou_threshold)
    year = dict(zip(images["image_id"], images["year"], strict=True))
    tow = dict(zip(images["image_id"], images["tow"], strict=True))
    matched["year"] = matched["image_id"].map(year)
    matched["tow"] = matched["image_id"].map(tow)
    return matched


def cross_fitted_confidences(
    matched: pd.DataFrame, methods: list[str], n_folds: int = 2, seed: int = 0
) -> dict[str, np.ndarray]:
    """Calibrated confidences where every detection's map was fitted on the other tows.

    Tows are assigned to folds at random; for each fold the maps are fitted on
    the remaining folds and applied to it, so that flexible maps such as
    isotonic regression are not judged on the detections they were fitted to.
    """
    tows = matched["tow"].astype(str).to_numpy()
    unique = np.array(sorted(set(tows)))
    rng = np.random.default_rng(seed)
    fold_of_tow = {t: k % n_folds for k, t in enumerate(unique[rng.permutation(len(unique))])}
    folds = np.array([fold_of_tow[t] for t in tows])
    scores = matched["score"].to_numpy()
    y = matched["tp"].to_numpy(dtype=float)
    out = {m: np.empty_like(scores) for m in methods}
    for k in range(n_folds):
        test = folds == k
        if not test.any() or test.all():
            continue
        for m in methods:
            out[m][test] = Calibrator.fit(m, scores[~test], y[~test]).apply(scores[test])
    return out


def metrics_by_slice(
    confidences: dict[str, np.ndarray], y: np.ndarray, slices: dict[str, pd.Series], n_bins: int = 15
) -> pd.DataFrame:
    """Negative log likelihood and calibration error per named confidence vector and slice."""
    rows = []
    for name, p in confidences.items():
        for slice_name, mask in slices.items():
            m = mask.to_numpy()
            if m.sum() == 0:
                continue
            rows.append(
                {
                    "calibrator": name,
                    "slice": slice_name,
                    "n_detections": int(m.sum()),
                    "n_true_positives": int(y[m].sum()),
                    "nll": nll(p[m], y[m]),
                    "ece": expected_calibration_error(p[m], y[m], n_bins),
                    "mean_confidence": float(p[m].mean()),
                    "fraction_positive": float(y[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def calibration_by_slice(
    matched: pd.DataFrame,
    calibrators: dict[str, Calibrator],
    slices: dict[str, pd.Series],
    n_bins: int = 15,
) -> pd.DataFrame:
    """Negative log likelihood and calibration error per calibrator and slice of detections."""
    scores = matched["score"].to_numpy()
    confidences = {name: cal.apply(scores) for name, cal in calibrators.items()}
    return metrics_by_slice(confidences, matched["tp"].to_numpy(dtype=float), slices, n_bins)


def run_calibrate_detector(
    cfg: dict[str, Any], out_dir: Path, force: bool = False, experiment: str = "", seed: int = 0
) -> Path:
    """Fit calibration maps on validation detections, compare slices, store the chosen map and thresholds."""
    from ..detector_stage import _experiment, _paths

    exp = _experiment(cfg, experiment)
    paths = _paths(cfg, exp, seed)
    home_run = paths["home_run"]
    ccfg = cfg.get("calibration", {})
    iou_thr = float(cfg.get("evaluation", {}).get("iou_threshold", 0.5))
    coco_path = paths["coco_dir"] / "val.json"
    preds = pd.read_parquet(home_run / "predictions_val.parquet")
    coco = load_coco(coco_path)
    matched = labelled_detections(preds, coco, iou_thr)
    cal_dir = home_run / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    focus_year = str(ccfg.get("focus_year", "2018"))
    methods = list(ccfg.get("methods", ["temperature", "platt", "isotonic"]))
    scores = matched["score"].to_numpy()
    y = matched["tp"].to_numpy(dtype=float)
    in_focus = matched["year"].astype(str) == focus_year
    calibrators: dict[str, Calibrator] = {"identity": Calibrator("identity", {})}
    for m in methods:
        calibrators[f"{m}_all"] = Calibrator.fit(m, scores, y)
        if in_focus.sum() >= int(ccfg.get("min_focus_detections", 200)) and m != "isotonic":
            calibrators[f"{m}_{focus_year}"] = Calibrator.fit(m, scores[in_focus], y[in_focus])
    slices = {"all": pd.Series(True, index=matched.index), focus_year: in_focus}
    for year in sorted(matched["year"].astype(str).unique()):
        if year and year != focus_year:
            slices[year] = matched["year"].astype(str) == year
    pooled = [yr for yr in matched["year"].astype(str).unique() if yr not in (focus_year, "2017", "2024")]
    if pooled:
        slices["pooled_other_years"] = matched["year"].astype(str).isin(pooled)
    # Most detections at the inference threshold carry near-zero scores; the
    # operating regime is better described by detections above a small floor.
    floor = float(ccfg.get("score_floor", 0.05))
    above = matched["score"] >= floor
    for name in list(slices):
        slices[f"{name}_above_{floor:g}"] = slices[name] & above
    n_bins = int(ccfg.get("n_bins", 15))
    table = calibration_by_slice(matched, calibrators, slices, n_bins=n_bins)
    table.to_csv(cal_dir / "calibration_by_slice.csv", index=False)
    crossed = cross_fitted_confidences(
        matched, methods, n_folds=int(ccfg.get("n_folds", 2)), seed=int(cfg.get("seed", 0))
    )
    crossed["identity"] = scores
    cross_table = metrics_by_slice(crossed, y, slices, n_bins=n_bins)
    cross_table.to_csv(cal_dir / "calibration_cross_fitted.csv", index=False)

    chosen_method = str(ccfg.get("method", "platt"))
    fit_on = str(ccfg.get("fit_on", "all"))
    chosen_name = f"{chosen_method}_{focus_year if fit_on == 'focus' else 'all'}"
    chosen = calibrators.get(chosen_name, calibrators[f"{chosen_method}_all"])
    rel_rows = []
    for name in ("identity", chosen_name):
        cal = calibrators.get(name)
        if cal is None:
            continue
        p = cal.apply(scores)
        for slice_name, mask in slices.items():
            t = reliability_table(p[mask.to_numpy()], y[mask.to_numpy()], n_bins=int(ccfg.get("n_bins", 15)))
            t.insert(0, "slice", slice_name)
            t.insert(0, "calibrator", name)
            rel_rows.append(t)
    pd.concat(rel_rows, ignore_index=True).to_csv(cal_dir / "reliability.csv", index=False)

    matched["score_calibrated"] = chosen.apply(scores)
    cal_preds = matched.rename(columns={"score": "score_raw"}).rename(columns={"score_calibrated": "score"})
    n_gt = gt_table(coco)
    n_gt = n_gt[n_gt["iscrowd"] == 0]["category_id"].value_counts().to_dict()
    curves = precision_recall_curves(
        cal_preds, {int(c["id"]): int(n_gt.get(int(c["id"]), 0)) for c in coco["categories"]}
    )
    names = {int(c["id"]): c["name"] for c in coco["categories"]}
    thresholds = best_thresholds(
        curves, min_precision=ccfg.get("min_precision"), n_gt={int(k): int(v) for k, v in n_gt.items()}
    )
    thresholds["class_name"] = thresholds["category_id"].map(names)
    thresholds.to_csv(cal_dir / "thresholds_calibrated.csv", index=False)

    # Object rule: per-class thresholds against one pooled threshold for the label classes (validation only).
    from .objects import choose_object_rule
    from .resolve import assign_queries

    label_classes = list(cfg.get("evaluation", {}).get("class_sets", {}).get("label_classes", []))
    val_preds = preds.copy()
    val_preds["class_name"] = val_preds["category_id"].map(names)
    val_preds["score_calibrated"] = chosen.apply(val_preds["score"].to_numpy())
    val_preds = assign_queries(val_preds, "image_id")
    per_class = dict(zip(thresholds["class_name"], thresholds["threshold"].astype(float), strict=True))
    gts_all = gt_table(coco)
    rule = choose_object_rule(
        val_preds,
        gts_all,
        names,
        per_class,
        label_classes,
        image_table(coco)["image_id"].to_numpy(),
        iou_suppress=float(cfg.get("inference", {}).get("resolution_iou", 0.7)),
        iou_match=iou_thr,
    )
    rule["tables"].to_csv(cal_dir / "object_rules_validation.csv", index=False)
    rule["curve"].to_csv(cal_dir / "pooled_f1_curve.csv", index=False)
    with open(cal_dir / "object_rule.json", "w", encoding="utf-8") as fh:
        json.dump(
            {k: v for k, v in rule.items() if k not in ("tables", "curve")}, fh, indent=2, default=float
        )
    log.info(
        "object rule: per-class macro F1 %.3f, pooled macro F1 %.3f at threshold %.2f, chosen %s",
        rule["per_class"]["macro_f1_label_classes"],
        rule["pooled"]["macro_f1_label_classes"],
        rule["pooled_threshold"],
        rule["chosen"],
    )

    focus_rows = table[table["slice"] == focus_year].set_index("calibrator")
    all_rows = table[table["slice"] == "all"].set_index("calibrator")
    cross_all = cross_table[cross_table["slice"] == f"all_above_{floor:g}"].set_index("calibrator")
    cross_focus = cross_table[cross_table["slice"] == f"{focus_year}_above_{floor:g}"].set_index("calibrator")
    summary = {
        "experiment": experiment,
        "seed": seed,
        "n_detections": int(len(matched)),
        "n_true_positives": int(y.sum()),
        "focus_year": focus_year,
        "n_focus_detections": int(in_focus.sum()),
        "chosen": chosen_name,
        "calibrator": chosen.to_dict(),
        "ece_all": {k: float(v) for k, v in all_rows["ece"].items()},
        f"ece_{focus_year}": {k: float(v) for k, v in focus_rows["ece"].items()},
        "nll_all": {k: float(v) for k, v in all_rows["nll"].items()},
        f"nll_{focus_year}": {k: float(v) for k, v in focus_rows["nll"].items()},
        f"cross_fitted_ece_all_above_{floor:g}": {k: float(v) for k, v in cross_all["ece"].items()},
        f"cross_fitted_ece_{focus_year}_above_{floor:g}": {
            k: float(v) for k, v in cross_focus["ece"].items()
        },
        f"cross_fitted_nll_all_above_{floor:g}": {k: float(v) for k, v in cross_all["nll"].items()},
    }
    with open(cal_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    outputs: list[str | Path] = [
        cal_dir / "calibration_by_slice.csv",
        cal_dir / "calibration_cross_fitted.csv",
        cal_dir / "reliability.csv",
        cal_dir / "thresholds_calibrated.csv",
        cal_dir / "calibration.json",
    ]
    write_manifest(
        cal_dir,
        "calibrate-detector",
        {**cfg, "experiment": experiment, "seed": seed},
        {"predictions": home_run / "predictions_val.parquet", "ground_truth": coco_path},
        outputs,
    )
    log.info(
        "calibration %s: ece all %.4f -> %.4f",
        chosen_name,
        all_rows.loc["identity", "ece"],
        all_rows.loc[chosen_name, "ece"] if chosen_name in all_rows.index else float("nan"),
    )
    return cal_dir
