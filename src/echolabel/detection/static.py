"""Detections of fixed structure in the camera's field of view.

A towed camera sees parts of its own gear. Near the surface in daylight the
housing and the warp appear in every frame, and the detector labels them as
organisms with a stable box. Such a box recurs at the same position and size
in a large share of the frames of a folder, which no animal does. Templates
are therefore the boxes whose position and size cell recurs in at least a
minimum share of a folder's frames; a detection of any class, in any folder
of the transect, that overlaps a template is static and is not counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import box_iou

BOX = ["x0", "y0", "x1", "y1"]
TEMPLATE_COLUMNS = [
    "folder",
    "class_name",
    *BOX,
    "frames_with_box",
    "folder_frames",
    "share",
]


def static_templates(
    detections: pd.DataFrame,
    frames_per_folder: dict[str, int],
    cell_px: float = 16.0,
    min_share: float = 0.2,
    merge_iou: float = 0.5,
) -> pd.DataFrame:
    """Boxes that recur in at least ``min_share`` of a folder's frames.

    ``detections`` needs ``folder``, ``frame_index``, ``class_name`` and the
    box columns in pixels; pass the detections that count as objects so that
    the templates come from confident boxes. Position and size are quantised
    to ``cell_px``. Cells that hold at least half of ``min_share`` are
    candidates; candidates of one folder whose median boxes overlap at
    ``merge_iou`` or more are merged, because a box that jitters across a
    cell boundary is split between cells. One template per merged group,
    with the median box, the most frequent class and the number of distinct
    frames in which it appears.
    """
    if detections.empty:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    d = detections[["folder", "frame_index", "class_name", *BOX]].copy()
    d["cx"] = np.floor((d["x0"] + d["x1"]) / 2.0 / cell_px).astype(int)
    d["cy"] = np.floor((d["y0"] + d["y1"]) / 2.0 / cell_px).astype(int)
    d["w"] = np.floor((d["x1"] - d["x0"]) / cell_px).astype(int)
    d["h"] = np.floor((d["y1"] - d["y0"]) / cell_px).astype(int)
    key = ["folder", "cx", "cy", "w", "h"]
    cells = (
        d.groupby(key, sort=False)
        .agg(
            frames_with_box=("frame_index", "nunique"),
            x0=("x0", "median"),
            y0=("y0", "median"),
            x1=("x1", "median"),
            y1=("y1", "median"),
        )
        .reset_index()
    )
    cells["folder_frames"] = cells["folder"].map(frames_per_folder).astype(float)
    cells = cells[cells["frames_with_box"] / cells["folder_frames"] >= min_share / 2.0]
    if cells.empty:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    # Merge overlapping candidate cells of one folder (union-find on the IoU of their median boxes).
    cells = cells.reset_index(drop=True)
    cluster = np.arange(len(cells))

    def find(i: int) -> int:
        while cluster[i] != i:
            cluster[i] = cluster[cluster[i]]
            i = cluster[i]
        return i

    for _folder, group in cells.groupby("folder", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < 2:
            continue
        iou = box_iou(group[BOX].to_numpy(dtype=float), group[BOX].to_numpy(dtype=float))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if iou[a, b] >= merge_iou:
                    ra, rb = find(idx[a]), find(idx[b])
                    if ra != rb:
                        cluster[rb] = ra
    cells["cluster"] = [find(i) for i in range(len(cells))]
    merged = d.merge(cells[[*key, "cluster"]], on=key, how="inner")
    table = (
        merged.groupby(["folder", "cluster"], sort=False)
        .agg(
            frames_with_box=("frame_index", "nunique"),
            class_name=("class_name", lambda s: s.value_counts().idxmax()),
            x0=("x0", "median"),
            y0=("y0", "median"),
            x1=("x1", "median"),
            y1=("y1", "median"),
        )
        .reset_index()
    )
    table["folder_frames"] = table["folder"].map(frames_per_folder).astype(float)
    table["share"] = table["frames_with_box"] / table["folder_frames"]
    table = table[table["share"] >= min_share].sort_values("share", ascending=False)
    return table[TEMPLATE_COLUMNS].reset_index(drop=True)


def flag_static(detections: pd.DataFrame, templates: pd.DataFrame, iou_threshold: float = 0.5) -> np.ndarray:
    """Boolean per detection: its box overlaps a template box at or above ``iou_threshold``."""
    if detections.empty or templates.empty:
        return np.zeros(len(detections), dtype=bool)
    boxes = detections[BOX].to_numpy(dtype=float)
    temps = templates[BOX].to_numpy(dtype=float)
    out = np.zeros(len(detections), dtype=bool)
    step = 200_000
    for start in range(0, len(boxes), step):
        iou = box_iou(boxes[start : start + step], temps)
        out[start : start + step] = iou.max(axis=1) >= iou_threshold
    return out


def static_summary(detections: pd.DataFrame, static: np.ndarray, class_col: str = "class_name") -> dict:
    """Per class: detections flagged as static and the total, over the rows given."""
    if detections.empty:
        return {}
    counts = pd.DataFrame({"class": detections[class_col].to_numpy(), "static": static}).groupby("class")[
        "static"
    ]
    return {str(c): {"static": int(s.sum()), "total": int(len(s))} for c, s in counts if int(s.sum()) > 0}
