"""Queries, ranks and cross-class resolution of detections.

A detection transformer scores every class for every query, and the stored
detections are the highest (query, class) pairs, so one object can appear
with two classes and the same box. Rows that share a box within an image are
the same query; their rank orders the classes of that query by score.

Counts are formed three ways. Unresolved: every row at or above its class
threshold. Resolved: an object exists when the highest calibrated score of
its query clears that class's threshold, and overlapping objects of different
classes (intersection over union above a limit) keep the highest score; each
surviving object counts once, in its highest class. Expected: each surviving
object distributes one unit of count over its two highest classes in
proportion to their calibrated scores, so the expected counts sum to the
resolved object count and differ from it only in how objects are shared
between confusable classes. The sum of calibrated scores of all stored
hypotheses is kept as a diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import box_iou

BOX = ["x0", "y0", "x1", "y1"]


def assign_queries(
    detections: pd.DataFrame, image_col: str = "image_id", score_col: str = "score"
) -> pd.DataFrame:
    """Add ``query`` (identical box within an image) and ``rank`` (1 for the highest score of the query)."""
    out = detections.copy()
    if out.empty:
        out["query"] = pd.Series(dtype="int64")
        out["rank"] = pd.Series(dtype="int64")
        return out
    key = out[[image_col, *BOX]].round(3)
    out["query"] = key.groupby([image_col, *BOX], sort=False).ngroup().astype("int64")
    out["rank"] = out.groupby("query")[score_col].rank(method="first", ascending=False).astype("int64")
    return out


def resolve_cross_class(
    detections: pd.DataFrame,
    thresholds: dict[str, float],
    image_col: str = "image_id",
    class_col: str = "class_name",
    score_col: str = "score_calibrated",
    iou_threshold: float = 0.7,
) -> np.ndarray:
    """Boolean per row: the row is an object's highest class, clears its threshold, and survives suppression.

    Candidates are the rank-1 rows whose calibrated score is at or above the
    threshold of their class. Within each image the candidates are visited by
    decreasing score; one is kept unless an already kept candidate of a
    different class overlaps it at more than ``iou_threshold``. Rows of rank
    2 and beyond are never kept.
    """
    kept = np.zeros(len(detections), dtype=bool)
    if detections.empty:
        return kept
    thr = detections[class_col].map(thresholds).fillna(1.0).to_numpy(dtype=float)
    candidate = (detections["rank"].to_numpy() == 1) & (detections[score_col].to_numpy(dtype=float) >= thr)
    sub = detections[candidate]
    for _, group in sub.groupby(image_col, sort=False):
        order = group[score_col].to_numpy(dtype=float).argsort()[::-1]
        g = group.iloc[order]
        boxes = g[BOX].to_numpy(dtype=float)
        classes = g[class_col].to_numpy()
        idx = g.index.to_numpy()
        iou = box_iou(boxes, boxes) if len(g) > 1 else np.ones((1, 1))
        keep_local = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            earlier = np.flatnonzero(keep_local[:i])
            if earlier.size and np.any((iou[i, earlier] > iou_threshold) & (classes[earlier] != classes[i])):
                continue
            keep_local[i] = True
        kept[detections.index.get_indexer(idx[keep_local])] = True
    return kept


def expected_counts(
    detections: pd.DataFrame,
    index: pd.Index,
    classes: list[str],
    kept: np.ndarray,
    image_col: str = "frame_index",
    class_col: str = "class_name",
    score_col: str = "score_calibrated",
) -> pd.DataFrame:
    """One unit per kept object, split over its two highest classes in proportion to their calibrated scores.

    Objects are the rows flagged in ``kept``; the second class is the rank-2
    row of the same query when one is stored, otherwise the whole unit goes to
    the first class. The result sums, per image, to the resolved count.
    """
    table = pd.DataFrame(0.0, index=index, columns=classes)
    if detections.empty or not kept.any():
        return table.astype("float32")
    first = detections[kept]
    second = detections[detections["rank"] == 2].set_index("query")
    if second.index.has_duplicates:
        raise ValueError("query identifiers are not unique; assign queries over the whole table first")
    s1 = first[score_col].to_numpy(dtype=float)
    has_second = first["query"].isin(second.index).to_numpy()
    s2 = np.zeros(len(first))
    c2 = np.array([None] * len(first), dtype=object)
    if has_second.any():
        match = second.loc[first.loc[has_second, "query"].to_numpy()]
        s2[has_second] = match[score_col].to_numpy(dtype=float)
        c2[has_second] = match[class_col].to_numpy()
    total = s1 + s2
    share1 = np.where(total > 0, s1 / np.where(total > 0, total, 1.0), 1.0)
    rows = pd.DataFrame(
        {
            image_col: np.concatenate([first[image_col].to_numpy(), first[image_col].to_numpy()[has_second]]),
            class_col: np.concatenate([first[class_col].to_numpy(), c2[has_second]]),
            "mass": np.concatenate([share1, (1.0 - share1)[has_second]]),
        }
    )
    summed = rows.groupby([image_col, class_col])["mass"].sum().unstack(fill_value=0.0)
    return summed.reindex(index=index, columns=classes, fill_value=0.0).astype("float32")


def hypothesis_sums(
    detections: pd.DataFrame,
    index: pd.Index,
    classes: list[str],
    image_col: str = "frame_index",
    class_col: str = "class_name",
    score_col: str = "score_calibrated",
    max_rank: int = 2,
    floor: float = 0.05,
) -> pd.DataFrame:
    """Diagnostic sum of calibrated scores per image and class over rows up to ``max_rank`` above floor."""
    sel = detections[(detections["rank"] <= max_rank) & (detections[score_col] >= floor)]
    if sel.empty:
        return pd.DataFrame(0.0, index=index, columns=classes).astype("float32")
    table = sel.groupby([image_col, class_col])[score_col].sum().unstack(fill_value=0.0)
    return table.reindex(index=index, columns=classes, fill_value=0.0).astype("float32")


def pair_counts(
    detections: pd.DataFrame,
    thresholds: dict[str, float],
    kept: np.ndarray,
    image_col: str = "image_id",
    class_col: str = "class_name",
    score_col: str = "score_calibrated",
    iou_threshold: float = 0.7,
) -> pd.DataFrame:
    """Which classes suppress which: counts of (kept class, suppressed class) among the candidate objects."""
    thr = detections[class_col].map(thresholds).fillna(1.0).to_numpy(dtype=float)
    candidate = (detections["rank"].to_numpy() == 1) & (detections[score_col].to_numpy(dtype=float) >= thr)
    rows = []
    for _, group in detections[candidate].groupby(image_col, sort=False):
        k = kept[detections.index.get_indexer(group.index)]
        winners, losers = group[k], group[~k]
        if winners.empty or losers.empty:
            continue
        iou = box_iou(losers[BOX].to_numpy(dtype=float), winners[BOX].to_numpy(dtype=float))
        for i in range(len(losers)):
            j = int(iou[i].argmax())
            if iou[i, j] > iou_threshold:
                rows.append(
                    {"kept_class": winners[class_col].iloc[j], "suppressed_class": losers[class_col].iloc[i]}
                )
    if not rows:
        return pd.DataFrame(columns=["kept_class", "suppressed_class", "count"])
    return pd.DataFrame(rows).value_counts().rename("count").reset_index()
