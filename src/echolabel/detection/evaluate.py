"""Detector evaluation from a predictions table and COCO ground truth.

The evaluation is independent of the training framework. Every framework
writes its detections as a table with one row per box (image identifier,
class identifier, score and corner coordinates in pixels of the original
image); this module matches them to the ground truth and computes average
precision, precision and recall against the confidence threshold, confusion
with background, slices by year and bootstrap intervals over images.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRED_COLUMNS = ["image_id", "category_id", "score", "x0", "y0", "x1", "y1"]


def load_coco(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gt_table(coco: dict[str, Any]) -> pd.DataFrame:
    """Ground truth boxes as corner coordinates in pixels."""
    rows = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        rows.append(
            {
                "image_id": int(a["image_id"]),
                "category_id": int(a["category_id"]),
                "x0": float(x),
                "y0": float(y),
                "x1": float(x + w),
                "y1": float(y + h),
                "iscrowd": int(a.get("iscrowd", 0)),
            }
        )
    return pd.DataFrame(rows, columns=["image_id", "category_id", "x0", "y0", "x1", "y1", "iscrowd"])


def image_table(coco: dict[str, Any]) -> pd.DataFrame:
    """Images with year and tow parsed from the relative file name."""
    rows = []
    for im in coco["images"]:
        parts = str(im["file_name"]).split("/")
        rows.append(
            {
                "image_id": int(im["id"]),
                "file_name": im["file_name"],
                "width": int(im["width"]),
                "height": int(im["height"]),
                "year": parts[0] if len(parts) > 1 else "",
                "tow": "/".join(parts[:2]) if len(parts) > 2 else "",
            }
        )
    return pd.DataFrame(rows)


def year_slices(images: pd.DataFrame, pooled: dict[str, list[str]] | None = None) -> dict[str, np.ndarray]:
    """Image identifiers per year, plus pooled groups such as 2020 to 2023."""
    pooled = pooled or {"2020-2023": ["2020", "2021", "2022", "2023"]}
    slices: dict[str, np.ndarray] = {"all": images["image_id"].to_numpy()}
    for year, sub in images.groupby("year"):
        if year:
            slices[str(year)] = sub["image_id"].to_numpy()
    for name, years in pooled.items():
        ids = images.loc[images["year"].isin(years), "image_id"].to_numpy()
        if ids.size:
            slices[name] = ids
    return slices


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over union of two arrays of corner boxes, shapes (n, 4) and (m, 4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(preds: pd.DataFrame, gts: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    """Greedy matching of detections to ground truth of the same class within each image.

    Detections are processed in order of decreasing score; each is a true
    positive when its best unmatched ground truth box of the same class has an
    intersection over union of at least ``iou_threshold``. Returns the
    predictions with the columns ``tp`` (bool), ``matched_gt`` (index into
    ``gts`` or -1) and ``iou``.
    """
    preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
    tp = np.zeros(len(preds), dtype=bool)
    matched = np.full(len(preds), -1, dtype=int)
    ious = np.zeros(len(preds))
    gt_groups = {k: v for k, v in gts.groupby(["image_id", "category_id"])}
    for (image_id, cat), group in preds.groupby(["image_id", "category_id"], sort=False):
        g = gt_groups.get((image_id, cat))
        if g is None or g.empty:
            continue
        gt_boxes = g[["x0", "y0", "x1", "y1"]].to_numpy()
        gt_index = g.index.to_numpy()
        used = np.zeros(len(g), dtype=bool)
        iou = box_iou(group[["x0", "y0", "x1", "y1"]].to_numpy(), gt_boxes)
        for row_pos, pred_idx in enumerate(group.index):
            candidates = np.where(used, -1.0, iou[row_pos])
            j = int(candidates.argmax())
            if candidates[j] >= iou_threshold:
                used[j] = True
                tp[pred_idx] = True
                matched[pred_idx] = gt_index[j]
                ious[pred_idx] = candidates[j]
    out = preds.copy()
    out["tp"] = tp
    out["matched_gt"] = matched
    out["iou"] = ious
    return out


def precision_recall_curves(
    matched: pd.DataFrame,
    n_gt: dict[int, int],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision, recall and F1 per class at a grid of confidence thresholds."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 3)
    rows = []
    for cat, total in n_gt.items():
        m = matched[matched["category_id"] == cat]
        scores = m["score"].to_numpy()
        tps = m["tp"].to_numpy()
        for thr in thresholds:
            keep = scores >= thr
            tp = int(tps[keep].sum())
            fp = int(keep.sum() - tp)
            fn = int(total - tp)
            precision = tp / (tp + fp) if tp + fp > 0 else 1.0
            recall = tp / total if total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rows.append(
                {
                    "category_id": cat,
                    "threshold": float(thr),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def average_precision(matched: pd.DataFrame, n_gt: int) -> float:
    """All-point interpolated average precision for one class from matched detections."""
    if n_gt == 0:
        return float("nan")
    if matched.empty:
        return 0.0
    m = matched.sort_values("score", ascending=False)
    tp = np.cumsum(m["tp"].to_numpy())
    fp = np.cumsum(~m["tp"].to_numpy())
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], recall, [recall[-1] if recall.size else 0.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def per_class_ap(matched: pd.DataFrame, n_gt: dict[int, int]) -> dict[int, float]:
    return {
        cat: average_precision(matched[matched["category_id"] == cat], total) for cat, total in n_gt.items()
    }


def best_thresholds(
    curves: pd.DataFrame,
    criterion: str = "f1",
    min_precision: float | None = None,
    n_gt: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Operating threshold per class maximising ``criterion``, optionally under a precision floor.

    A class without ground truth in the data behind ``curves`` gets a threshold
    of 1.0, so that it is never predicted, and ``has_support`` False.
    """
    rows = []
    for cat, sub in curves.groupby("category_id"):
        if n_gt is not None and int(n_gt.get(int(cat), 0)) == 0:
            rows.append(
                {
                    "category_id": cat,
                    "threshold": 1.0,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "has_support": False,
                }
            )
            continue
        cand = sub if min_precision is None else sub[sub["precision"] >= min_precision]
        if cand.empty:
            cand = sub
        best = cand.sort_values([criterion, "threshold"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "category_id": cat,
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "has_support": True,
            }
        )
    return pd.DataFrame(rows)


def confusion_with_background(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    class_ids: list[int],
    thresholds: dict[int, float],
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Confusion matrix including background, after per-class thresholds.

    Detections above their class threshold are matched to ground truth boxes
    of any class by intersection over union within each image, greedily by
    score. Rows are true classes plus ``background`` (false positives), columns
    are predicted classes plus ``background`` (missed ground truth).
    """
    names = [str(c) for c in class_ids] + ["background"]
    mat = pd.DataFrame(0, index=names, columns=names, dtype=int)
    thr = preds["category_id"].astype(int).map(thresholds).fillna(1.0).to_numpy(dtype=float)
    keep = preds[preds["score"].to_numpy(dtype=float) >= thr]
    gt_by_image = {k: v for k, v in gts.groupby("image_id")}
    for image_id, group in keep.groupby("image_id"):
        g = gt_by_image.get(image_id)
        group = group.sort_values("score", ascending=False)
        if g is None or g.empty:
            for cat in group["category_id"]:
                mat.loc["background", str(int(cat))] += 1
            continue
        iou = box_iou(group[["x0", "y0", "x1", "y1"]].to_numpy(), g[["x0", "y0", "x1", "y1"]].to_numpy())
        used = np.zeros(len(g), dtype=bool)
        gt_cats = g["category_id"].to_numpy()
        for row_pos, cat in enumerate(group["category_id"].to_numpy()):
            candidates = np.where(used, -1.0, iou[row_pos])
            j = int(candidates.argmax())
            if candidates[j] >= iou_threshold:
                used[j] = True
                mat.loc[str(int(gt_cats[j])), str(int(cat))] += 1
            else:
                mat.loc["background", str(int(cat))] += 1
        for j in np.flatnonzero(~used):
            mat.loc[str(int(gt_cats[j])), "background"] += 1
    for image_id, g in gt_by_image.items():
        if image_id not in set(keep["image_id"]):
            for cat in g["category_id"]:
                mat.loc[str(int(cat)), "background"] += 1
    return mat


def _ap_from_cumulative(tp_cum: np.ndarray, fp_cum: np.ndarray, n_gt: float) -> float:
    """All-point interpolated average precision from cumulative weighted true and false positives."""
    if n_gt <= 0:
        return float("nan")
    if tp_cum.size == 0:
        return 0.0
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * envelope))


def bootstrap_ap_by_slice(
    matched: pd.DataFrame,
    gts: pd.DataFrame,
    class_ids: list[int],
    names: dict[int, str],
    slices: dict[str, np.ndarray],
    n_boot: int = 200,
    seed: int = 0,
    min_gt: int = 10,
    class_sets: dict[str, list[int]] | None = None,
    return_draws: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Per-class AP50 and macro mean per slice of images, with a paired percentile bootstrap over images.

    The macro mean ``map50`` averages the classes with at least ``min_gt``
    ground truth boxes in the slice (or resample), so that classes with a
    handful of boxes do not dominate it. ``class_sets`` adds one macro mean
    per named subset of classes (``map50_<name>``) under the same rule.

    A resample draws images with replacement; each detection and each ground
    truth box then carries the multiplicity of its image as a weight, so that
    the average precision of a resample is computed from weighted cumulative
    sums without rebuilding tables. The same draws are used for every class,
    and for the same images and seed they are the same across models, so two
    models can be compared resample by resample (``return_draws`` also returns
    the per-resample values per slice and metric; the first entry is the
    unweighted estimate).
    """
    rng = np.random.default_rng(seed)
    gts = gts[gts["iscrowd"] == 0]
    rows = []
    all_draws: dict[str, dict[str, np.ndarray]] = {}
    for slice_name, ids in slices.items():
        ids = np.unique(np.asarray(ids))
        n_images = int(ids.size)
        if n_images == 0:
            continue
        position = pd.Series(np.arange(n_images), index=ids)
        m = matched[matched["image_id"].isin(ids)].sort_values("score", ascending=False)
        m_pos = position.loc[m["image_id"].to_numpy()].to_numpy()
        m_cat = m["category_id"].to_numpy()
        m_tp = m["tp"].to_numpy(dtype=float)
        g = gts[gts["image_id"].isin(ids)]
        g_pos = position.loc[g["image_id"].to_numpy()].to_numpy()
        g_cat = g["category_id"].to_numpy()
        per_class = {}
        for c in class_ids:
            sel = m_cat == c
            gt_counts = np.bincount(g_pos[g_cat == c], minlength=n_images).astype(float)
            per_class[c] = (m_pos[sel], m_tp[sel], gt_counts)
        class_sets = class_sets or {}
        draws: dict[Any, list[float]] = {c: [] for c in class_ids}
        draws["map50"] = []
        for set_name in class_sets:
            draws[f"map50_{set_name}"] = []
        n_classes: dict[str, int] = {}
        weights = [np.ones(n_images)]
        weights += [
            np.bincount(rng.integers(0, n_images, n_images), minlength=n_images).astype(float)
            for _ in range(n_boot)
        ]
        for k, counts in enumerate(weights):
            usable: dict[int, float] = {}
            for c in class_ids:
                mp, mt, gc = per_class[c]
                w = counts[mp]
                n_gt_c = float((gc * counts).sum())
                ap = _ap_from_cumulative(np.cumsum(mt * w), np.cumsum((1.0 - mt) * w), n_gt_c)
                draws[c].append(ap)
                if not np.isnan(ap) and n_gt_c >= min_gt:
                    usable[c] = ap
            draws["map50"].append(float(np.mean(list(usable.values()))) if usable else float("nan"))
            if k == 0:
                n_classes["map50"] = len(usable)
            for set_name, members in class_sets.items():
                vals = [usable[c] for c in members if c in usable]
                draws[f"map50_{set_name}"].append(float(np.mean(vals)) if vals else float("nan"))
                if k == 0:
                    n_classes[f"map50_{set_name}"] = len(vals)
        all_draws[slice_name] = {}
        for key, values in draws.items():
            arr = np.asarray(values, dtype=float)
            metric = key if isinstance(key, str) else f"ap50_{names[key]}"
            all_draws[slice_name][metric] = arr
            boot = arr[1:]
            boot = boot[np.isfinite(boot)]
            rows.append(
                {
                    "slice": slice_name,
                    "metric": metric,
                    "value": float(arr[0]),
                    "ci_low": float(np.percentile(boot, 2.5)) if boot.size else np.nan,
                    "ci_high": float(np.percentile(boot, 97.5)) if boot.size else np.nan,
                    "n_boot": int(boot.size),
                    "n_images": n_images,
                    "n_classes": n_classes.get(metric, 1 if not isinstance(key, str) else 0),
                }
            )
    columns = ["slice", "metric", "value", "ci_low", "ci_high", "n_boot", "n_images", "n_classes"]
    table = pd.DataFrame(rows, columns=columns)
    return (table, all_draws) if return_draws else table


def paired_difference(
    draws_a: dict[str, dict[str, np.ndarray]], draws_b: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Difference B minus A per slice and metric with a percentile interval over the shared resamples."""
    rows = []
    for slice_name, metrics in draws_a.items():
        for metric, a in metrics.items():
            b = draws_b.get(slice_name, {}).get(metric)
            if b is None or len(a) != len(b):
                continue
            d = b[1:] - a[1:]
            d = d[np.isfinite(d)]
            rows.append(
                {
                    "slice": slice_name,
                    "metric": metric,
                    "value_a": float(a[0]),
                    "value_b": float(b[0]),
                    "difference": float(b[0] - a[0]),
                    "ci_low": float(np.percentile(d, 2.5)) if d.size else np.nan,
                    "ci_high": float(np.percentile(d, 97.5)) if d.size else np.nan,
                    "n_boot": int(d.size),
                }
            )
    return pd.DataFrame(rows)


def operating_point_by_slice(
    matched: pd.DataFrame,
    gts: pd.DataFrame,
    class_ids: list[int],
    names: dict[int, str],
    slices: dict[str, np.ndarray],
    thresholds: dict[int, float],
    n_boot: int = 200,
    seed: int = 0,
    score_col: str = "score",
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Recall and precision per class at the class thresholds per slice, resampled like the AP bootstrap.

    The draws follow the same sequence as ``bootstrap_ap_by_slice`` for the
    same slices and seed, so the per-resample values pair with it and with
    another model evaluated on the same images.
    """
    rng = np.random.default_rng(seed)
    gts = gts[gts["iscrowd"] == 0]
    rows = []
    all_draws: dict[str, dict[str, np.ndarray]] = {}
    for slice_name, ids in slices.items():
        ids = np.unique(np.asarray(ids))
        n_images = int(ids.size)
        if n_images == 0:
            continue
        position = pd.Series(np.arange(n_images), index=ids)
        m = matched[matched["image_id"].isin(ids)]
        thr = m["category_id"].astype(int).map(thresholds).fillna(1.0).to_numpy(dtype=float)
        m = m[m[score_col].to_numpy(dtype=float) >= thr]
        m_pos = position.loc[m["image_id"].to_numpy()].to_numpy()
        m_cat = m["category_id"].to_numpy()
        m_tp = m["tp"].to_numpy(dtype=float)
        g = gts[gts["image_id"].isin(ids)]
        g_pos = position.loc[g["image_id"].to_numpy()].to_numpy()
        g_cat = g["category_id"].to_numpy()
        weights = [np.ones(n_images)]
        weights += [
            np.bincount(rng.integers(0, n_images, n_images), minlength=n_images).astype(float)
            for _ in range(n_boot)
        ]
        all_draws[slice_name] = {}
        for c in class_ids:
            sel = m_cat == c
            tp_counts = np.bincount(m_pos[sel & (m_tp > 0)], minlength=n_images).astype(float)
            fp_counts = np.bincount(m_pos[sel & (m_tp == 0)], minlength=n_images).astype(float)
            gt_counts = np.bincount(g_pos[g_cat == c], minlength=n_images).astype(float)
            recalls, precisions = [], []
            for w in weights:
                tp, fp, n_gt = (
                    float((tp_counts * w).sum()),
                    float((fp_counts * w).sum()),
                    float((gt_counts * w).sum()),
                )
                recalls.append(tp / n_gt if n_gt > 0 else np.nan)
                precisions.append(tp / (tp + fp) if tp + fp > 0 else np.nan)
            for metric, values in ((f"recall_{names[c]}", recalls), (f"precision_{names[c]}", precisions)):
                arr = np.asarray(values, dtype=float)
                all_draws[slice_name][metric] = arr
                boot = arr[1:]
                boot = boot[np.isfinite(boot)]
                rows.append(
                    {
                        "slice": slice_name,
                        "metric": metric,
                        "value": float(arr[0]),
                        "ci_low": float(np.percentile(boot, 2.5)) if boot.size else np.nan,
                        "ci_high": float(np.percentile(boot, 97.5)) if boot.size else np.nan,
                        "n_boot": int(boot.size),
                        "n_images": n_images,
                        "threshold": float(thresholds.get(c, 1.0)),
                        "n_gt": int(gt_counts.sum()),
                    }
                )
    return pd.DataFrame(rows), all_draws


def predicted_as(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    class_id: int,
    image_ids: np.ndarray,
    names: dict[int, str],
    thresholds: dict[int, float],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """What the ground truth boxes of one class in the given images are predicted as.

    Each box takes the class of the highest-scoring detection above its class
    threshold that overlaps it at the IoU threshold, or ``background`` when
    none does. Detections are not shared between boxes.
    """
    ids = set(int(i) for i in np.asarray(image_ids))
    g_all = gts[(gts["iscrowd"] == 0) & gts["image_id"].isin(ids)]
    thr = preds["category_id"].astype(int).map(thresholds).fillna(1.0).to_numpy(dtype=float)
    kept = preds[(preds["score"].to_numpy(dtype=float) >= thr) & preds["image_id"].isin(ids)]
    counts: dict[str, int] = {names[c]: 0 for c in names}
    counts["background"] = 0
    for image_id, g in g_all.groupby("image_id"):
        target = g[g["category_id"] == class_id]
        if target.empty:
            continue
        p = kept[kept["image_id"] == image_id].sort_values("score", ascending=False)
        if p.empty:
            counts["background"] += len(target)
            continue
        iou = box_iou(p[["x0", "y0", "x1", "y1"]].to_numpy(), target[["x0", "y0", "x1", "y1"]].to_numpy())
        used = np.zeros(len(p), dtype=bool)
        for j in range(len(target)):
            col = np.where(used, -1.0, iou[:, j])
            i = int(col.argmax())
            if col[i] >= iou_threshold:
                used[i] = True
                counts[names[int(p["category_id"].iloc[i])]] += 1
            else:
                counts["background"] += 1
    return counts


def matched_predictions(
    gt_path: str | Path, preds: pd.DataFrame, iou_threshold: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], dict[int, str]]:
    """Match one predictions table to a COCO file: matched detections, ground truth, images, ids and names."""
    coco = load_coco(gt_path)
    gts = gt_table(coco)
    images = image_table(coco)
    class_ids = [int(c["id"]) for c in coco["categories"]]
    names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    preds = preds[PRED_COLUMNS].copy()
    preds = preds[preds["image_id"].isin(images["image_id"])]
    matched = match_detections(preds, gts[gts["iscrowd"] == 0], iou_threshold)
    return matched, gts, images, class_ids, names


def coco_metrics(
    gt_path: str | Path, preds: pd.DataFrame, image_ids: np.ndarray | None = None
) -> dict[str, float]:
    """COCO-style mAP50:95 and mAP50 through faster-coco-eval or pycocotools, on an optional image subset."""
    try:
        from faster_coco_eval import COCO  # type: ignore
        from faster_coco_eval import COCOeval_faster as COCOeval
    except ImportError:  # pragma: no cover
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    gt = COCO(str(gt_path))
    if preds.empty:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0}
    results = [
        {
            "image_id": int(r.image_id),
            "category_id": int(r.category_id),
            "score": float(r.score),
            "bbox": [float(r.x0), float(r.y0), float(r.x1 - r.x0), float(r.y1 - r.y0)],
        }
        for r in preds.itertuples(index=False)
    ]
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, "bbox")
    if image_ids is not None:
        ev.params.imgIds = [int(i) for i in image_ids]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {
        "map": float(s[0]),
        "map50": float(s[1]),
        "map75": float(s[2]),
        "map_small": float(s[3]),
        "map_medium": float(s[4]),
        "map_large": float(s[5]),
    }


def evaluate_predictions(
    gt_path: str | Path,
    preds: pd.DataFrame,
    iou_threshold: float = 0.5,
    n_boot: int = 200,
    seed: int = 0,
    slices: dict[str, np.ndarray] | None = None,
    thresholds: dict[int, float] | None = None,
    min_gt_for_macro: int = 10,
    class_sets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Full evaluation of one predictions table against one COCO file.

    Returns a dictionary with ``coco`` (framework-style mAP for the whole set),
    ``curves`` (precision and recall per class and threshold), ``thresholds``
    (operating points chosen by F1 unless given), ``confusion`` (with
    background at those thresholds), and ``slices`` (per-class AP50 with
    bootstrap intervals for every slice of images).
    """
    coco = load_coco(gt_path)
    gts = gt_table(coco)
    images = image_table(coco)
    class_ids = [int(c["id"]) for c in coco["categories"]]
    names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    preds = preds[PRED_COLUMNS].copy()
    preds = preds[preds["image_id"].isin(images["image_id"])]
    matched = match_detections(preds, gts[gts["iscrowd"] == 0], iou_threshold)
    n_gt = {c: int((gts["category_id"] == c).sum()) for c in class_ids}
    curves = precision_recall_curves(matched, n_gt)
    curves["class_name"] = curves["category_id"].map(names)
    chosen = (
        best_thresholds(curves, n_gt=n_gt)
        if thresholds is None
        else pd.DataFrame([{"category_id": c, "threshold": t} for c, t in thresholds.items()])
    )
    chosen["class_name"] = chosen["category_id"].map(names)
    thr_map = dict(zip(chosen["category_id"].astype(int), chosen["threshold"], strict=True))
    confusion = confusion_with_background(preds, gts, class_ids, thr_map, iou_threshold)
    confusion.index = [names.get(int(i), i) if i != "background" else i for i in confusion.index]
    confusion.columns = [names.get(int(c), c) if c != "background" else c for c in confusion.columns]
    slices = slices or year_slices(images)
    ids_by_name = {v: k for k, v in names.items()}
    sets_by_id = {
        set_name: [ids_by_name[n] for n in members if n in ids_by_name]
        for set_name, members in (class_sets or {}).items()
    }
    slice_table = bootstrap_ap_by_slice(
        matched,
        gts,
        class_ids,
        names,
        slices,
        n_boot=n_boot,
        seed=seed,
        min_gt=min_gt_for_macro,
        class_sets=sets_by_id,
    )
    try:
        coco_stats = coco_metrics(gt_path, preds)
    except Exception as exc:  # pragma: no cover
        log.warning("COCO evaluation failed: %s", exc)
        coco_stats = {}
    return {
        "coco": coco_stats,
        "curves": curves,
        "thresholds": chosen,
        "confusion": confusion,
        "slices": slice_table,
        "n_gt": {names[c]: n for c, n in n_gt.items()},
        "n_images": int(len(images)),
        "n_predictions": int(len(preds)),
    }
