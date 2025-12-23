"""Object rules: which detections count as objects, and how the rule is chosen on validation.

Two rules are compared. Under the per-class rule an object exists when the
highest calibrated score of its query clears the operating threshold of that
class. Under the pooled rule the five label classes share one threshold,
chosen on validation to maximise the F1 of "any label-class object"; other
classes keep their own thresholds. Both rules assign the class by the top
hypothesis and apply the same cross-class suppression and expected-count
split. The rule with the higher macro F1 over the label classes on validation
is used for inference; both are reported on the test slices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluate import match_detections
from .resolve import BOX, expected_counts, resolve_cross_class

RULES = ("per_class", "pooled")


def effective_thresholds(
    rule: str, per_class: dict[str, float], pooled_threshold: float | None, label_classes: list[str]
) -> dict[str, float]:
    """Thresholds that decide whether an object exists under a rule."""
    if rule == "per_class" or pooled_threshold is None:
        return dict(per_class)
    return {c: (pooled_threshold if c in label_classes else per_class.get(c, 1.0)) for c in per_class} | {
        c: pooled_threshold for c in label_classes if c not in per_class
    }


def objects_under_rule(
    preds: pd.DataFrame,
    thresholds: dict[str, float],
    image_col: str = "image_id",
    iou_suppress: float = 0.7,
) -> pd.DataFrame:
    """Rows that are objects under the given existence thresholds: rank 1, above threshold, not suppressed."""
    kept = resolve_cross_class(preds, thresholds, image_col, iou_threshold=iou_suppress)
    out = preds[kept].copy()
    out["kept"] = True
    return out


def per_class_pr(
    objects: pd.DataFrame,
    gts: pd.DataFrame,
    names: dict[int, str],
    image_ids: np.ndarray,
    iou_match: float = 0.5,
) -> pd.DataFrame:
    """Boxes, counted objects, true and false positives, misses, precision and recall per class in a slice."""
    ids = set(int(i) for i in np.asarray(image_ids))
    g = gts[(gts["iscrowd"] == 0) & gts["image_id"].isin(ids)]
    o = objects[objects["image_id"].isin(ids)]
    if o.empty:
        matched = o.assign(tp=pd.Series(dtype=bool))
    else:
        table = o[["image_id", "category_id", *BOX]].copy()
        table["score"] = o["score_calibrated"].to_numpy(dtype=float)
        matched = match_detections(table, g, iou_match)
    rows = []
    for cid, name in names.items():
        n_boxes = int((g["category_id"] == cid).sum())
        m = matched[matched["category_id"] == cid]
        counted, tp = int(len(m)), int(m["tp"].sum()) if len(m) else 0
        fp, fn = counted - tp, n_boxes - tp
        precision = tp / counted if counted else np.nan
        recall = tp / n_boxes if n_boxes else np.nan
        if counted and n_boxes and tp > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (counted or n_boxes) else np.nan
        rows.append(
            {
                "class_name": name,
                "n_boxes": n_boxes,
                "counted": counted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def pooled_matches(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    label_ids: list[int],
    image_ids: np.ndarray,
    iou_match: float = 0.5,
) -> tuple[pd.DataFrame, int]:
    """Rank-1 label-class rows matched to label-class boxes of any label class, and the number of boxes."""
    ids = set(int(i) for i in np.asarray(image_ids))
    cand = preds[(preds["rank"] == 1) & preds["category_id"].isin(label_ids) & preds["image_id"].isin(ids)]
    g = gts[(gts["iscrowd"] == 0) & gts["category_id"].isin(label_ids) & gts["image_id"].isin(ids)].copy()
    if cand.empty or g.empty:
        return cand.assign(tp=False, score=cand.get("score_calibrated", pd.Series(dtype=float))), int(len(g))
    table = cand[["image_id", *BOX]].copy()
    table["category_id"] = -1
    table["score"] = cand["score_calibrated"].to_numpy(dtype=float)
    g["category_id"] = -1
    return match_detections(table, g, iou_match), int(len(g))


def pooled_f1_curve(matched: pd.DataFrame, n_boxes: int, taus: np.ndarray) -> pd.DataFrame:
    """F1 of "any label-class object" against the shared threshold, from one class-agnostic matching."""
    rows = []
    scores = matched["score"].to_numpy(dtype=float) if len(matched) else np.zeros(0)
    tp_flags = matched["tp"].to_numpy(dtype=bool) if len(matched) else np.zeros(0, dtype=bool)
    for tau in taus:
        above = scores >= tau
        tp = int((tp_flags & above).sum())
        fp = int((~tp_flags & above).sum())
        fn = n_boxes - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_boxes if n_boxes else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(tau),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def pooled_f1_of_objects(
    objects: pd.DataFrame,
    gts: pd.DataFrame,
    label_ids: list[int],
    image_ids: np.ndarray,
    iou_match: float = 0.5,
) -> dict[str, float]:
    """F1 of "any label-class object" for a given object set."""
    ids = set(int(i) for i in np.asarray(image_ids))
    o = objects[objects["category_id"].isin(label_ids) & objects["image_id"].isin(ids)]
    g = gts[(gts["iscrowd"] == 0) & gts["category_id"].isin(label_ids) & gts["image_id"].isin(ids)].copy()
    n_boxes = int(len(g))
    if o.empty or g.empty:
        return {"tp": 0, "fp": int(len(o)), "fn": n_boxes, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    table = o[["image_id", *BOX]].copy()
    table["category_id"] = -1
    table["score"] = o["score_calibrated"].to_numpy(dtype=float)
    g["category_id"] = -1
    m = match_detections(table, g, iou_match)
    tp = int(m["tp"].sum())
    fp = int(len(m) - tp)
    precision = tp / len(m)
    recall = tp / n_boxes
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": n_boxes - tp, "precision": precision, "recall": recall, "f1": f1}


def rule_table(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    names: dict[int, str],
    image_ids: np.ndarray,
    thresholds: dict[str, float],
    iou_suppress: float = 0.7,
    iou_match: float = 0.5,
) -> pd.DataFrame:
    """Per-class boxes, counted, expected, precision and recall for one rule's thresholds in one slice."""
    ids = set(int(i) for i in np.asarray(image_ids))
    sub = preds[preds["image_id"].isin(ids)]
    kept = resolve_cross_class(sub, thresholds, "image_id", iou_threshold=iou_suppress)
    objects = sub[kept]
    table = per_class_pr(objects, gts, names, image_ids, iou_match)
    classes = [names[c] for c in sorted(names)]
    index = pd.Index(sorted(ids), name="image_id")
    expected = expected_counts(sub, index, classes, kept, "image_id").sum(axis=0)
    table["expected"] = table["class_name"].map(expected).astype(float)
    table["threshold"] = table["class_name"].map(thresholds).astype(float)
    return table


def choose_object_rule(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    names: dict[int, str],
    per_class_thresholds: dict[str, float],
    label_classes: list[str],
    image_ids: np.ndarray,
    taus: np.ndarray | None = None,
    iou_suppress: float = 0.7,
    iou_match: float = 0.5,
) -> dict[str, Any]:
    """Choose between the per-class and the pooled rule on validation.

    The pooled threshold maximises the F1 of "any label-class object" over
    ``taus``. Both rules are then scored by the macro F1 over the label
    classes with boxes in the validation images; the higher macro F1 wins and
    the per-class rule keeps ties.
    """
    taus = np.round(np.arange(0.05, 0.96, 0.01), 2) if taus is None else taus
    ids_by_name = {v: k for k, v in names.items()}
    label_ids = [ids_by_name[c] for c in label_classes if c in ids_by_name]
    matched, n_label_boxes = pooled_matches(preds, gts, label_ids, image_ids, iou_match)
    curve = pooled_f1_curve(matched, n_label_boxes, taus)
    best = curve.loc[curve["f1"].idxmax()]
    pooled_tau = float(best["threshold"])
    tables = {}
    summary: dict[str, Any] = {
        "pooled_threshold": pooled_tau,
        "pooled_curve_best": best.to_dict(),
        "label_classes": label_classes,
    }
    for rule in RULES:
        thr = effective_thresholds(rule, per_class_thresholds, pooled_tau, label_classes)
        table = rule_table(preds, gts, names, image_ids, thr, iou_suppress, iou_match)
        table.insert(0, "rule", rule)
        tables[rule] = table
        label_rows = table[table["class_name"].isin(label_classes) & (table["n_boxes"] > 0)]
        kept = resolve_cross_class(
            preds[preds["image_id"].isin(set(int(i) for i in image_ids))],
            thr,
            "image_id",
            iou_threshold=iou_suppress,
        )
        objects = preds[preds["image_id"].isin(set(int(i) for i in image_ids))][kept]
        pooled = pooled_f1_of_objects(objects, gts, label_ids, image_ids, iou_match)
        summary[rule] = {
            "macro_f1_label_classes": float(label_rows["f1"].mean()) if len(label_rows) else float("nan"),
            "n_label_classes_scored": int(len(label_rows)),
            "pooled_f1": pooled["f1"],
            "pooled_precision": pooled["precision"],
            "pooled_recall": pooled["recall"],
        }
    f_per, f_pool = (
        summary["per_class"]["macro_f1_label_classes"],
        summary["pooled"]["macro_f1_label_classes"],
    )
    summary["chosen"] = "pooled" if np.isfinite(f_pool) and f_pool > f_per else "per_class"
    summary["tables"] = pd.concat(tables.values(), ignore_index=True)
    summary["curve"] = curve
    return summary
