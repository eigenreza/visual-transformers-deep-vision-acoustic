from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.camera import clean_depth
from echolabel.camera_audit import (
    choose_excerpt,
    classify_long_runs,
    cleaning_diagnostics,
    depth_figure,
    update_sequence,
)


def _frames(n: int = 6000, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """A slow descent read every three frames, with one-sided spikes to deeper values and dropouts."""
    rng = np.random.default_rng(seed)
    t = pd.Series(pd.date_range("2018-05-09T20:00:00", periods=n, freq="204ms"))
    true = 20 + 0.03 * np.arange(n)
    updates = np.round(true / 0.8) * 0.8
    readings = np.repeat(updates[::3], 3)[:n]
    spikes = rng.random(n // 3) < 0.3
    readings = readings + np.repeat(np.where(spikes, 15.0, 0.0), 3)[:n]
    readings[rng.random(n) < 0.05] = 0.0
    return t, pd.Series(readings)


def test_update_sequence_and_diagnostics() -> None:
    t, d = _frames()
    cleaned, _ = clean_depth(t, d)
    values, times = update_sequence(d.to_numpy(dtype=float), t.to_numpy())
    assert values.size < len(d) / 2 and np.all(values > 0)
    diag = cleaning_diagnostics(cleaned, t)
    assert diag["n_frames"] == len(d)
    rmc = diag["raw_minus_cleaned"]
    assert rmc["fraction_deeper_gt_10m"] > 0.15 and rmc["fraction_shallower_gt_10m"] < 0.02
    assert diag["rejected"]["deeper_than_trend"] >= diag["rejected"]["shallower_than_trend"]
    assert (
        diag["sensor_updates"]["single_update_spikes_deeper"]
        > diag["sensor_updates"]["single_update_spikes_shallower"]
    )
    assert diag["far_runs"]["n_runs"] > 0 and diag["excursions"]["deeper_median_m"] > 10
    assert len(diag["residual_by_trend_depth"]) >= 1


def test_excerpt_and_figure(tmp_path) -> None:
    t, d = _frames(3000)
    cleaned, _ = clean_depth(t, d)
    start, end = choose_excerpt(cleaned, t, minutes=2)
    assert end > start
    path = tmp_path / "fig.png"
    depth_figure(cleaned, t, start, end, "test", path)
    assert path.is_file()


def test_long_runs_are_classified() -> None:
    n = 300
    trend = np.linspace(100.0, 160.0, n)  # a descent at 1 m/s
    speed = np.full(n, 1.0)
    raw = trend.copy()
    raw[50:100] = trend[50]  # the sensor holds one value while the trend keeps moving
    raw[150:170] = (
        trend[150:170] + 8.0
    )  # readings deeper than the trend during a descent: consistent with lag
    raw[220:240] = trend[220:240] + np.where(np.arange(20) % 2 == 0, 9.0, -9.0)  # a two-sided burst
    far = np.abs(raw - trend) > 5
    out = classify_long_runs(raw, trend, speed, far, min_frames=10)
    assert out["n_runs"] == 3
    assert out["stale_hold"] > 0 and out["ramp_lag"] == 20 and out["noise_burst"] == 20
    t, d = _frames()
    cleaned, _ = clean_depth(t, d)
    diag = cleaning_diagnostics(cleaned, t)
    assert "long_far_runs" in diag and len(diag["trend_window_sensitivity"]) == 15
