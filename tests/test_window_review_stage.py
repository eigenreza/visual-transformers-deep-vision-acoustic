from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.viz.results import echogram_panels
from echolabel.window_review_stage import choose_windows, frame_summary, overlap_share


def test_overlap_share_separates_fixed_from_moving_boxes() -> None:
    fixed = [np.array([[10.0, 10.0, 30.0, 30.0]])] * 4
    moving = [np.array([[10.0 + 50 * k, 10.0, 30.0 + 50 * k, 30.0]]) for k in range(4)]
    assert overlap_share(fixed) == 1.0 and overlap_share(moving) == 0.0
    assert np.isnan(overlap_share([np.zeros((0, 4))] * 3))


def test_frame_summary() -> None:
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 30], dtype=float)
    s = frame_summary(counts, top_share=0.1)
    assert s["n_frames"] == 10 and s["sum"] == 40 and s["median_per_frame"] == 0 and s["max_per_frame"] == 30
    assert np.isclose(s["occupancy"], 0.2) and np.isclose(s["share_in_top_frames"], 0.75)
    assert frame_summary(np.array([]))["n_frames"] == 0


def test_choose_windows_takes_overall_and_fold_worst() -> None:
    errors = pd.DataFrame(
        {
            "window_id": np.arange(8),
            "fold": ["a"] * 4 + ["b"] * 4,
            "abs_error_krill": [9.0, 8.0, 7.0, 6.0, 1.0, 0.5, 0.2, 0.1],
        }
    )
    chosen = choose_windows(errors, "krill", n_worst=2, n_per_fold=1)
    assert chosen["window_id"].tolist() == [0, 1, 4] and chosen["selection"].tolist() == [
        "overall",
        "overall",
        "fold",
    ]


def test_echogram_panels_writes_figure(tmp_path) -> None:
    sv = np.full((3, 30, 50), -80.0, dtype=np.float32)
    sv[1, :, :5] = np.nan
    path = echogram_panels(
        sv, np.arange(50) + 20.5, 45.0, ["38 kHz", "120 kHz", "200 kHz"], tmp_path / "e.png", "t"
    )
    assert path.is_file()
