from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.detection.objects import (
    choose_object_rule,
    effective_thresholds,
    pooled_f1_curve,
    pooled_matches,
    rule_table,
)
from echolabel.detection.resolve import assign_queries

NAMES = {1: "krill", 2: "herring", 3: "bluewhiting", 4: "maurolicus"}
LABELS = ["krill", "herring", "bluewhiting"]


def _box(i: int) -> dict:
    return {"x0": 100.0 * i, "y0": 0.0, "x1": 100.0 * i + 10, "y1": 10.0}


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Image 1: three herring boxes; detections: two herring (0.9, 0.6) and one
    # bluewhiting-top (0.7) on the third.
    # Image 2: one krill box detected at 0.55; one maurolicus false positive at 0.9.
    gts = pd.DataFrame(
        [
            {"image_id": 1, "category_id": 2, "iscrowd": 0, **_box(0)},
            {"image_id": 1, "category_id": 2, "iscrowd": 0, **_box(1)},
            {"image_id": 1, "category_id": 2, "iscrowd": 0, **_box(2)},
            {"image_id": 2, "category_id": 1, "iscrowd": 0, **_box(0)},
        ]
    )
    preds = pd.DataFrame(
        [
            {"image_id": 1, "category_id": 2, "score": 0.9, **_box(0)},
            {"image_id": 1, "category_id": 2, "score": 0.6, **_box(1)},
            {"image_id": 1, "category_id": 3, "score": 0.7, **_box(2)},
            {"image_id": 1, "category_id": 2, "score": 0.4, **_box(2)},
            {"image_id": 2, "category_id": 1, "score": 0.55, **_box(0)},
            {"image_id": 2, "category_id": 4, "score": 0.9, **_box(5)},
        ]
    )
    preds["class_name"] = preds["category_id"].map(NAMES)
    preds["score_calibrated"] = preds["score"]
    return assign_queries(preds), gts


def test_effective_thresholds() -> None:
    per = {"krill": 0.6, "herring": 0.8, "maurolicus": 0.5}
    pooled = effective_thresholds("pooled", per, 0.5, LABELS)
    assert pooled == {"krill": 0.5, "herring": 0.5, "maurolicus": 0.5, "bluewhiting": 0.5}
    assert effective_thresholds("per_class", per, 0.5, LABELS) == per


def test_rules_and_choice() -> None:
    preds, gts = _data()
    ids = np.array([1, 2])
    per_class = {"krill": 0.6, "herring": 0.8, "bluewhiting": 0.8, "maurolicus": 0.5}
    strict = rule_table(preds, gts, NAMES, ids, per_class).set_index("class_name")
    assert strict.loc["herring", "counted"] == 1 and strict.loc["herring", "recall"] == 1 / 3
    assert strict.loc["krill", "counted"] == 0 and strict.loc["maurolicus", "fp"] == 1
    matched, n_boxes = pooled_matches(preds, gts, [1, 2, 3], ids)
    assert (
        n_boxes == 4 and int(matched["tp"].sum()) == 4
    )  # bluewhiting-top detection counts for the pooled F1
    curve = pooled_f1_curve(matched, n_boxes, np.array([0.5, 0.65, 0.95]))
    assert curve.loc[0, "f1"] == 1.0 and curve.loc[2, "f1"] == 0.0
    choice = choose_object_rule(preds, gts, NAMES, per_class, LABELS, ids, taus=np.array([0.5, 0.65, 0.95]))
    assert choice["pooled_threshold"] == 0.5 and choice["chosen"] == "pooled"
    pooled = choice["tables"]
    pooled = pooled[pooled["rule"] == "pooled"].set_index("class_name")
    assert pooled.loc["herring", "counted"] == 2 and pooled.loc["bluewhiting", "counted"] == 1
    assert pooled.loc["krill", "counted"] == 1 and pooled.loc["maurolicus", "counted"] == 1
    assert abs(pooled["expected"].sum() - pooled["counted"].sum()) < 1e-6
