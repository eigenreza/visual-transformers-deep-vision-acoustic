from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.windows.build import Geometry, TransectPatches
from echolabel.windows_stage import build_geometry


def test_build_geometry_end_to_end() -> None:
    n_p, n_r = 1200, 600
    ping_time = np.datetime64("2018-05-09T20:00:00") + (np.arange(n_p) * 1100).astype("timedelta64[ms]")
    depth = 8.5 + np.arange(n_r) * 0.2
    sv = np.full((2, n_p, n_r), -80.0, dtype=np.float16)
    mask = np.zeros((2, n_p, n_r), dtype=np.uint8)
    mask[:, :, depth < 13.5] = 2
    patches = TransectPatches(sv, mask, ping_time, depth, 2.0, 0.5)
    t_cam = pd.date_range("2018-05-09T20:05:00", periods=60, freq="10s")
    mapping = pd.DataFrame(
        {
            "time_camera": t_cam,
            "time_vessel": t_cam - pd.Timedelta(seconds=240),
            "camera_depth_m": np.linspace(8, 90, 60),
            "lag_s": 240.0,
            "lag_sd_s": 15.0,
            "camera_in_exclusion_38000": np.linspace(8, 90, 60) < 13.5,
            "band_usable_fraction_38000": 1.0,
        }
    )
    series = pd.DataFrame(
        {
            "time": t_cam,
            "n_frames": 49,
            "krill_res": 2.0,
            "herring_res": 0.0,
            "krill_weighted": 1.5,
            "herring_weighted": 0.0,
            "krill_exp": 2.0,
            "herring_exp": 0.0,
            "unidentified_res": 1.0,
            "unidentified_weighted": 0.5,
            "unidentified_exp": 1.0,
        }
    )
    geometry = Geometry("surface", 60.0, 50.0, 20.0, "surface")
    built = build_geometry(
        geometry,
        patches,
        mapping,
        series,
        13.5,
        {"min_ping_fraction": 0.8},
        {
            "forms": {"hard": "_res", "weighted": "_weighted", "expected": "_exp"},
            "exclude": ["unidentified"],
            "merged": {},
        },
    )
    meta, targets, arrays = built["meta"], built["targets"], built["arrays"]
    assert len(meta) > 10 and arrays["sv_db"].shape[1:] == (2, 30, 100)
    assert built["classes"] == ["herring", "krill"]
    assert (meta["band_top_m"] >= 13.5).all()  # surface anchor never rises above the exclusion depth
    assert np.allclose(targets["hard_krill_at_lag"], 12.0) and "weighted_krill" in targets.columns
    assert "hard_unidentified_at_lag" not in targets.columns


def test_short_target_span_keeps_lag_distribution() -> None:
    n_p, n_r = 1200, 600
    ping_time = np.datetime64("2018-05-09T20:00:00") + (np.arange(n_p) * 1100).astype("timedelta64[ms]")
    depth = 8.5 + np.arange(n_r) * 0.2
    sv = np.full((1, n_p, n_r), -80.0, dtype=np.float16)
    patches = TransectPatches(sv, np.zeros((1, n_p, n_r), dtype=np.uint8), ping_time, depth, 2.0, 1.0)
    t_cam = pd.date_range("2018-05-09T20:06:00", periods=60, freq="10s")
    mapping = pd.DataFrame(
        {
            "time_camera": t_cam,
            "time_vessel": t_cam - pd.Timedelta(seconds=240),
            "camera_depth_m": 60.0,
            "lag_s": 240.0,
            "lag_sd_s": 20.0,
            "lag_distribution": "normal",
        }
    )
    series = pd.DataFrame({"time": t_cam, "n_frames": 49, "krill_res": 2.0})
    tcfg = {"forms": {"hard": "_res"}, "exclude": [], "merged": {}}
    central = Geometry("context3_minute", 180.0, 50.0, 30.0, "camera", None, 60.0, "distribution")
    whole = Geometry("context3", 180.0, 50.0, 30.0, "camera")
    span = Geometry("span", 180.0, 50.0, 30.0, "camera", None, 60.0)
    built = {
        g.name: build_geometry(g, patches, mapping, series, 13.5, {}, tcfg) for g in (central, whole, span)
    }
    # Same input patches, one camera minute of counts in the central form, three in the whole-window form.
    assert built["context3_minute"]["arrays"]["sv_db"].shape == built["context3"]["arrays"]["sv_db"].shape
    assert np.allclose(built["context3_minute"]["targets"]["hard_krill_at_lag"], 12.0)
    assert np.median(built["context3"]["targets"]["hard_krill_at_lag"]) == 36.0
    assert built["context3_minute"]["lag_offsets"]["distribution"] == "normal"
    assert built["span"]["lag_offsets"]["distribution"] == "fixed"
    assert "hard_krill_k4" in built["context3_minute"]["targets"].columns
