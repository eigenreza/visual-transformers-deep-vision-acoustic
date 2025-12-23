"""Export of annotation tables in COCO object detection format.

Image paths are written relative to the image root of the annotation
collection, so the files hold no machine-specific paths. Boxes are clipped to
the image and boxes that collapse below a minimum side length are dropped and
counted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .taxonomy import Taxonomy


def to_coco(
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    taxonomy: Taxonomy,
    clip_boxes: bool = True,
    min_box_side_px: float = 2.0,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build a COCO dictionary from image and box tables.

    ``images`` needs ``key``, ``image_rel``, ``width_used`` and ``height_used``;
    ``boxes`` needs ``key``, ``class_name``, ``xmin``, ``ymin``, ``xmax`` and
    ``ymax`` in pixels. Images without boxes are kept as negatives. Returns the
    dictionary and counts of clipped and dropped boxes.
    """
    categories = [{"id": taxonomy.class_id(c), "name": c} for c in taxonomy.classes]
    image_records = []
    key_to_id: dict[str, int] = {}
    for i, row in enumerate(images.itertuples(index=False), start=1):
        key_to_id[row.key] = i
        image_records.append(
            {
                "id": i,
                "file_name": row.image_rel,
                "width": int(row.width_used),
                "height": int(row.height_used),
            }
        )
    sizes = {
        row.key: (float(row.width_used), float(row.height_used)) for row in images.itertuples(index=False)
    }
    stats = {"clipped": 0, "dropped": 0}
    annotations = []
    ann_id = 1
    for row in boxes.itertuples(index=False):
        if row.key not in key_to_id:
            continue
        w_img, h_img = sizes[row.key]
        x0, y0, x1, y1 = float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)
        if clip_boxes:
            cx0, cy0 = min(max(x0, 0.0), w_img), min(max(y0, 0.0), h_img)
            cx1, cy1 = min(max(x1, 0.0), w_img), min(max(y1, 0.0), h_img)
            if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                stats["clipped"] += 1
            x0, y0, x1, y1 = cx0, cy0, cx1, cy1
        w, h = x1 - x0, y1 - y0
        if w < min_box_side_px or h < min_box_side_px:
            stats["dropped"] += 1
            continue
        annotations.append(
            {
                "id": ann_id,
                "image_id": key_to_id[row.key],
                "category_id": taxonomy.class_id(row.class_name),
                "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    coco = {
        "info": {"description": "Deep Vision annotation collection, detector classes"},
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, stats


def write_coco(coco: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh)
    return path
