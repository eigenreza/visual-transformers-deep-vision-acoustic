from __future__ import annotations

import pandas as pd

from echolabel.inference import frame_summary


def test_frame_summary_counts() -> None:
    dets = pd.DataFrame(
        {
            "frame_index": [0, 0, 0, 2, 2],
            "class_name": ["krill", "krill", "herring", "krill", "unidentified"],
            "score": [0.9, 0.15, 0.6, 0.07, 0.3],
            "score_calibrated": [0.95, 0.1, 0.5, 0.02, 0.2],
        }
    )
    classes = ["krill", "herring", "unidentified"]
    table = frame_summary(dets, 3, classes, [0.05, 0.5], {"krill": 0.3, "herring": 0.4, "unidentified": 0.1})
    assert len(table) == 3
    assert table.loc[0, "krill_ge0p05"] == 2 and table.loc[0, "krill_ge0p5"] == 1
    assert table.loc[0, "n_ge0p05"] == 3 and table.loc[0, "n_ge0p5"] == 2
    assert table.loc[1, "n_ge0p05"] == 0
    assert table.loc[2, "krill_ge0p05"] == 1 and table.loc[2, "unidentified_ge0p05"] == 1
    assert table.loc[0, "krill_op"] == 1 and table.loc[0, "herring_op"] == 1
    assert table.loc[2, "krill_op"] == 0 and table.loc[2, "unidentified_op"] == 1
    assert table.loc[2, "n_op"] == 1
    empty = frame_summary(dets.iloc[0:0], 2, classes, [0.1])
    assert empty["n_ge0p1"].tolist() == [0, 0]


def test_frame_summary_resolved_and_expected() -> None:
    import pandas as pd

    from echolabel.detection.resolve import assign_queries
    from echolabel.inference import frame_summary

    dets = pd.DataFrame(
        {
            "frame_index": [0, 0, 1],
            "class_name": ["herring", "bluewhiting", "krill"],
            "score": [0.8, 0.6, 0.7],
            "score_calibrated": [0.8, 0.6, 0.7],
            "x0": [0.0, 0.0, 5.0],
            "y0": [0.0, 0.0, 5.0],
            "x1": [10.0, 10.0, 15.0],
            "y1": [10.0, 10.0, 15.0],
        }
    )
    dets = assign_queries(dets, "frame_index")
    classes = ["herring", "bluewhiting", "krill"]
    table = frame_summary(
        dets, 3, classes, [0.05], {"herring": 0.5, "bluewhiting": 0.5, "krill": 0.5}, floor=0.05
    )
    row = table.set_index("frame_index").loc[0]
    assert row["herring_op"] == 1 and row["bluewhiting_op"] == 1  # unresolved counts both classes
    assert row["herring_res"] == 1 and row["bluewhiting_res"] == 0  # resolved keeps the stronger class
    assert abs(row["herring_exp"] - 0.8 / 1.4) < 1e-6 and abs(row["bluewhiting_exp"] - 0.6 / 1.4) < 1e-6
    assert abs(row["n_exp"] - 1.0) < 1e-6 and abs(row["herring_hyp"] - 0.8) < 1e-6
    assert table.set_index("frame_index").loc[2, "n_exp"] == 0
