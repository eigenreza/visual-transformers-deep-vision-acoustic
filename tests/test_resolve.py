from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.detection.resolve import (
    assign_queries,
    expected_counts,
    hypothesis_sums,
    pair_counts,
    resolve_cross_class,
)


def _detections() -> pd.DataFrame:
    # Image 1: one query seen as herring (0.8) and bluewhiting (0.6); a separate krill box.
    # Image 2: two different herring boxes overlapping each other, same class (left alone).
    return pd.DataFrame(
        {
            "image_id": [1, 1, 1, 2, 2],
            "class_name": ["herring", "bluewhiting", "krill", "herring", "herring"],
            "score": [0.8, 0.6, 0.7, 0.9, 0.85],
            "score_calibrated": [0.8, 0.6, 0.7, 0.9, 0.85],
            "x0": [0.0, 0.0, 50.0, 0.0, 2.0],
            "y0": [0.0, 0.0, 50.0, 0.0, 2.0],
            "x1": [10.0, 10.0, 60.0, 10.0, 12.0],
            "y1": [10.0, 10.0, 60.0, 10.0, 12.0],
        }
    )


def test_queries_ranks_and_resolution() -> None:
    d = assign_queries(_detections())
    assert d["query"].nunique() == 4
    assert d.loc[0, "rank"] == 1 and d.loc[1, "rank"] == 2 and d.loc[2, "rank"] == 1
    thresholds = {"herring": 0.5, "bluewhiting": 0.5, "krill": 0.5}
    kept = resolve_cross_class(d, thresholds)
    assert kept.tolist() == [True, False, True, True, True]
    # An object whose highest class fails its threshold does not exist, even if its second class would pass.
    kept_high = resolve_cross_class(d, {"herring": 0.95, "bluewhiting": 0.5, "krill": 0.5})
    assert kept_high.tolist() == [False, False, True, False, False]
    # Two candidate objects of different classes on the same box: the stronger one suppresses the other.
    two = pd.DataFrame(
        {
            "image_id": [1, 1],
            "class_name": ["herring", "bluewhiting"],
            "score": [0.8, 0.7],
            "score_calibrated": [0.8, 0.7],
            "x0": [0.0, 1.0],
            "y0": [0.0, 1.0],
            "x1": [10.0, 10.5],
            "y1": [10.0, 10.5],
        }
    )
    two = assign_queries(two)
    assert two["query"].nunique() == 2
    kept_two = resolve_cross_class(two, thresholds)
    assert kept_two.tolist() == [True, False]
    pairs = pair_counts(two, thresholds, kept_two)
    assert pairs.iloc[0].to_dict() == {"kept_class": "herring", "suppressed_class": "bluewhiting", "count": 1}


def test_expected_counts_sum_to_resolved() -> None:
    d = assign_queries(_detections()).rename(columns={"image_id": "frame_index"})
    thresholds = {"herring": 0.5, "bluewhiting": 0.5, "krill": 0.5}
    kept = resolve_cross_class(d, thresholds, "frame_index")
    classes = ["herring", "bluewhiting", "krill"]
    index = pd.RangeIndex(3, name="frame_index")
    table = expected_counts(d, index, classes, kept)
    assert np.isclose(table.loc[1, "herring"], 0.8 / 1.4) and np.isclose(
        table.loc[1, "bluewhiting"], 0.6 / 1.4
    )
    assert np.isclose(table.loc[1, "krill"], 1.0) and np.isclose(table.loc[2, "herring"], 2.0)
    assert np.isclose(table.sum().sum(), kept.sum())
    hyp = hypothesis_sums(d, index, classes, floor=0.05)
    assert np.isclose(hyp.loc[1, "herring"], 0.8) and np.isclose(hyp.loc[1, "bluewhiting"], 0.6)
    assert np.isclose(hyp.sum().sum(), 0.8 + 0.6 + 0.7 + 0.9 + 0.85)  # calibrated scores, not objects


def test_expected_counts_requires_unique_queries() -> None:
    import pytest

    # Two parts numbered their queries from zero, so one query id names two different objects.
    part = assign_queries(_detections()).rename(columns={"image_id": "frame_index"})
    other = part.copy()
    other["frame_index"] += 10
    both = pd.concat([part, other], ignore_index=True)
    thresholds = {"herring": 0.5, "bluewhiting": 0.5, "krill": 0.5}
    classes = ["herring", "bluewhiting", "krill"]
    index = pd.RangeIndex(13, name="frame_index")
    with pytest.raises(ValueError):
        expected_counts(both, index, classes, resolve_cross_class(both, thresholds, "frame_index"))
    fixed = assign_queries(both, "frame_index")
    assert fixed["query"].nunique() == 8
    kept = resolve_cross_class(fixed, thresholds, "frame_index")
    table = expected_counts(fixed, index, classes, kept)
    assert kept.sum() == 8 and np.isclose(table.sum().sum(), 8)
