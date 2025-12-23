from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.detection.static import flag_static, static_summary, static_templates


def _detections() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # Folder A: the housing appears in 60 of 100 frames with a jitter of a few pixels; krill boxes move.
    for frame in range(100):
        if frame < 60:
            j = rng.integers(-2, 3, size=4)
            rows.append(("A", frame, "benthosema", 700 + j[0], 10 + j[1], 810 + j[2], 70 + j[3]))
        for _ in range(3):
            x, y = rng.uniform(0, 1100), rng.uniform(100, 900)
            rows.append(("A", frame, "krill", x, y, x + rng.uniform(10, 40), y + rng.uniform(10, 40)))
    # Folder B: the same housing in 5 of 100 frames only, labelled gelatinous.
    for frame in range(100, 200):
        if frame < 105:
            rows.append(("B", frame, "gelatinous", 701, 11, 809, 69))
        x, y = rng.uniform(0, 1100), rng.uniform(100, 900)
        rows.append(("B", frame, "krill", x, y, x + 20, y + 20))
    return pd.DataFrame(rows, columns=["folder", "frame_index", "class_name", "x0", "y0", "x1", "y1"])


def test_templates_and_flags() -> None:
    d = _detections()
    frames = {"A": 100, "B": 100}
    templates = static_templates(d, frames, cell_px=16, min_share=0.2)
    assert len(templates) == 1
    t = templates.iloc[0]
    assert t["folder"] == "A" and t["class_name"] == "benthosema" and 0.55 <= t["share"] <= 0.6
    assert abs(t["x0"] - 700) <= 2 and abs(t["y1"] - 70) <= 2
    static = flag_static(d, templates, iou_threshold=0.5)
    # Every housing box is static, in both folders and under both class names; no krill box is.
    housing = d["class_name"].isin(["benthosema", "gelatinous"]).to_numpy()
    assert static[housing].all() and not static[~housing].any()
    summary = static_summary(d, static)
    assert summary == {"benthosema": {"static": 60, "total": 60}, "gelatinous": {"static": 5, "total": 5}}


def test_no_templates_below_share() -> None:
    d = _detections()
    templates = static_templates(d, {"A": 100, "B": 100}, min_share=0.7)
    assert templates.empty
    assert not flag_static(d, templates).any()
    assert not flag_static(d.iloc[0:0], templates).any()
