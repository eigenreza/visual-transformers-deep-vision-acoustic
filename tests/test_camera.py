from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.camera import aggregate_frames, clean_depth


def _series(n: int = 1500, seed: int = 0) -> tuple[pd.Series, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-05-09T20:00:00", periods=n, freq="200ms")
    true = 100 + 40 * np.sin(np.linspace(0, 2, n))  # slow descent, metres, under 0.2 m/s
    depth = true + rng.normal(0, 0.3, n)
    depth[300:303] = 2.4  # a burst of surface readings in deep water
    depth[600:602] = 0.0  # short dropout
    depth[800:802] = true[800:802] + 40  # jump
    depth[1000:1200] = 0.0  # long dropout, 40 s
    return pd.Series(t), pd.Series(depth), true


def test_clean_depth_flags_rejects_and_gaps() -> None:
    t, d, true = _series()
    out, summary = clean_depth(t, d, trend_window_s=60.0, smooth_window_s=30.0, max_gap_s=10.0)
    flags = out["reading_flag"]
    assert set(flags[300:303]) == {"rejected"}
    assert set(flags[800:802]) == {"rejected"}
    assert set(flags[600:602]) == {"dropout"}
    assert flags[0] == "consistent" and flags[1499] == "consistent"
    assert summary["reading_flags"]["rejected"] == 5
    assert summary["speed_bound_m_per_s"] < 0.5
    source = out["depth_source"]
    assert set(source[1080:1120]) == {"none"}
    assert set(source[600:602]) == {"median"}
    good = out["depth_clean"].notna().to_numpy()
    assert np.max(np.abs(out["depth_clean"].to_numpy()[good] - true[good])) < 2.5


def test_aggregate_frames() -> None:
    t, d, _ = _series(n=100)
    out, _ = clean_depth(t, d, trend_window_s=10.0, smooth_window_s=5.0)
    frames = out.copy()
    frames["krill_op"] = 1
    frames["n_op"] = 1
    dets = pd.DataFrame({"timestamp": t[:10], "class_name": ["krill"] * 10, "score_calibrated": [0.5] * 10})
    series = aggregate_frames(frames, dets, ["krill", "herring"], 10.0, ["krill_op", "n_op"])
    assert len(series) == 2
    assert series.loc[0, "n_frames"] == 50 and series.loc[0, "krill_op"] == 50
    assert abs(series.loc[0, "krill_weighted"] - 5.0) < 1e-9 and series.loc[0, "herring_weighted"] == 0.0
    assert series.loc[1, "krill_weighted"] == 0.0


def test_aggregate_frames_static_fraction() -> None:
    frames = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2018-05-05 06:00:00", "2018-05-05 06:00:05", "2018-05-05 06:00:12"]
            ),
            "depth_clean": [5.0, 5.0, 6.0],
            "n_static": [1, 0, 0],
        }
    )
    frames["static_frame"] = (frames["n_static"] > 0).astype(int)
    series = aggregate_frames(frames, None, ["krill"], 10.0, ["n_static", "static_frame"])
    assert series.loc[0, "frames_static_fraction"] == 0.5 and series.loc[1, "frames_static_fraction"] == 0.0
