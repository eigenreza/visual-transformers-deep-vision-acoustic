from __future__ import annotations

import numpy as np

from echolabel.acoustics.explore import (
    bottom_candidates,
    coarse_echogram,
    impulsive_pings,
    range_profile,
    to_db,
)


def test_to_db_and_profile() -> None:
    sv = np.full((2, 5, 4), 1e-8)
    sv[0, 0, 0] = np.nan
    sv[1, :, 3] = 1e-20
    db = to_db(sv, floor_db=-130)
    assert np.isnan(db[0, 0, 0]) and abs(db[0, 1, 1] + 80) < 1e-9 and abs(db[1, 0, 3] + 130) < 1e-9
    prof = range_profile(db, np.array([0.0, 0.2, 0.4, 0.6]), np.array([38000.0, 18000.0]))
    assert len(prof) == 8
    assert abs(prof.loc[0, "nan_fraction"] - 0.2) < 1e-9
    assert abs(prof.loc[(prof.frequency_hz == 18000.0) & (prof.range_m == 0.6), "db_q50"].item() + 130) < 1e-9


def test_bottom_and_impulsive() -> None:
    rng = np.arange(0, 100, 1.0)
    db = np.full((20, 100), -90.0)
    db[:, 80] = -20.0  # a strong return at 80 m in every ping
    bottom = bottom_candidates(db, rng, min_range_m=50, strong_db=-35)
    assert bottom["strong"].all() and (bottom["peak_range_m"] == 80).all()
    db2 = np.full((100, 100), -90.0)
    db2[40, :] = -40.0  # one impulsive ping
    mask = impulsive_pings(db2, rng, lo_m=10, hi_m=90, z=5.0, window=21)
    assert mask[40] and mask.sum() == 1


def test_coarse_echogram() -> None:
    sv = np.ones((1, 6, 4)) * 1e-8
    sv[0, :3, :] = 1e-6
    times = np.array(
        [
            "2018-05-09T20:00:00",
            "2018-05-09T20:00:01",
            "2018-05-09T20:00:02",
            "2018-05-09T20:00:10",
            "2018-05-09T20:00:11",
            "2018-05-09T20:00:12",
        ],
        dtype="datetime64[ns]",
    )
    depth = np.array([8.5, 9.5, 10.5, 11.5])
    coarse, t_edges, d_edges = coarse_echogram(sv, times, depth, time_step_s=10, depth_step_m=2)
    assert coarse.shape == (1, 2, 2)
    assert abs(coarse[0, 0, 0] + 60) < 1e-6 and abs(coarse[0, 1, 1] + 80) < 1e-6
    assert len(t_edges) == 3 and len(d_edges) == 3
