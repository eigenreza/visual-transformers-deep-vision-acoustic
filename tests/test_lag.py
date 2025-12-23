from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.alignment.lag import (
    band_series,
    band_usable_fraction,
    best_lag,
    block_bootstrap_lag,
    echogram_at_intervals,
    held_out_lag_check,
    lag_bounds,
    lag_bounds_per_interval,
    lag_correlation,
    lag_mapping,
    paired_series,
    piecewise_lags,
    setback_from_warp,
    vessel_speed,
)


def test_vessel_speed_and_bounds() -> None:
    n = 600
    t = np.datetime64("2018-05-09T20:00:00") + np.arange(n) * np.timedelta64(1, "s")
    lat = np.full(n, 70.0)
    lon = np.linspace(10.0, 10.0 + 2.0 / (111320 * np.cos(np.deg2rad(70.0))) * 600, n)  # 2 m/s eastward
    speed = vessel_speed(t, lat, lon, window_s=60)
    assert abs(np.nanmedian(speed["speed_m_s"]) - 2.0) < 0.05
    bounds = lag_bounds(speed["speed_m_s"].to_numpy(), (400.0, 1200.0))
    assert (
        abs(bounds["lag_s_at_median_speed"][0] - 200) < 10
        and abs(bounds["lag_s_at_median_speed"][1] - 600) < 30
    )
    assert abs(setback_from_warp(500.0, 300.0) - 400.0) < 1e-9


def _synthetic(lag_true: int = 6, n_t: int = 400, seed: int = 0):
    """An echogram with a time-varying layer and a camera that sees it ``lag_true`` intervals later."""
    rng = np.random.default_rng(seed)
    centres = np.arange(0.5, 300.5, 1.0)
    strength = np.convolve(rng.normal(size=n_t + 20), np.ones(10) / 10, mode="same")[:n_t]
    layer_depth = 150 + 30 * np.sin(np.arange(n_t) / 40)
    echogram = np.full((n_t, centres.size), 1e-9)
    for t in range(n_t):
        band = np.abs(centres - layer_depth[t]) <= 10
        echogram[t, band] = 1e-7 * np.exp(strength[t])
    camera_depth = np.roll(layer_depth, lag_true)
    camera_signal = np.full(n_t, np.nan)
    camera_signal[lag_true:] = np.exp(strength[:-lag_true]) * 5 + rng.normal(scale=0.3, size=n_t - lag_true)
    return echogram, centres, camera_depth, camera_signal


def test_lag_is_recovered_with_interval() -> None:
    echogram, centres, camera_depth, camera_signal = _synthetic(lag_true=6)
    lags = np.arange(0, 15)
    cam, acoustic = paired_series(camera_signal, echogram, centres, camera_depth, 5.0, lags, smooth=3)
    curve = lag_correlation(cam, acoustic, lags)
    lag, corr = best_lag(curve)
    assert lag == 6 and corr > 0.8
    boot = block_bootstrap_lag(cam, acoustic, lags, block=30, n_boot=40, seed=1)
    assert boot["lag_q025"] <= 6 <= boot["lag_q975"]
    assert boot["curves"].shape == (40, len(lags))


def test_piecewise_and_held_out() -> None:
    echogram, centres, camera_depth, camera_signal = _synthetic(lag_true=6, n_t=480)
    lags = np.arange(0, 15)
    cam, acoustic = paired_series(camera_signal, echogram, centres, camera_depth, 5.0, lags, smooth=3)
    segments = piecewise_lags(cam, acoustic, lags, segment=120, min_pairs=40)
    assert len(segments) == 4
    assert (segments["lag_intervals"].dropna() == 6).all()
    check = held_out_lag_check(segments, lags, global_lag=6)
    assert not check.empty and np.all(np.abs(check["gain_varying"]) < 1e-9)


def test_echogram_at_intervals_and_band() -> None:
    n_p, n_r = 30, 50
    ping_time = np.datetime64("2018-05-09T20:00:00") + np.arange(n_p) * np.timedelta64(1, "s")
    depth = 8.5 + np.arange(n_r) * 0.2
    db = np.full((n_p, n_r), -80.0)
    db[:, depth >= 15] = -60.0
    hard = np.zeros((n_p, n_r), dtype=bool)
    hard[:, :5] = True
    starts = np.datetime64("2018-05-09T20:00:00") + np.arange(3) * np.timedelta64(10, "s")
    e, centres = echogram_at_intervals(ping_time, db, hard, depth, starts, 10.0, depth_step_m=1.0)
    assert e.shape == (3, centres.size)
    assert np.isnan(e[0, 0]) and abs(10 * np.log10(e[0, 1]) + 80) < 1e-6
    series = band_series(e, centres, np.array([16.0, 16.0, 16.0]), 0.6, lag_intervals=1)
    assert np.isnan(series[0]) and abs(10 * np.log10(series[1]) + 60) < 1e-6
    mapping = lag_mapping(starts, np.array([16.0, 16.0, 16.0]), lag_s=60.0, lag_sd_s=10.0)
    assert (mapping["time_camera"] - mapping["time_vessel"] == pd.Timedelta(seconds=60)).all()


def test_build_signals_sums_matching_columns() -> None:
    from echolabel.alignment_stage import build_signals

    series = pd.DataFrame(
        {
            "krill_weighted": [1.0, 2.0],
            "herring_weighted": [0.5, 0.5],
            "unidentified_weighted": [9.0, 9.0],
            "n_op": [3, 4],
        }
    )
    signals = build_signals(
        series,
        {"all": ["*_weighted"], "fish": ["herring_weighted"], "none": ["missing_*"]},
        ["unidentified_weighted"],
    )
    assert set(signals) == {"all", "fish"}
    assert np.allclose(signals["all"], [1.5, 2.5]) and np.allclose(signals["fish"], [0.5, 0.5])


def test_exclusion_and_usable_fraction() -> None:
    echogram = np.full((4, 50), 1e-8)
    centres = np.arange(0.5, 50.5, 1.0)
    echogram[:, :14] = np.nan  # cells above 14 m depth are never usable
    depth = np.array([12.0, 12.0, 30.0, 30.0])
    strict = band_series(echogram, centres, depth, 5.0, 0, exclusion_depth_m=13.5, min_usable_fraction=0.5)
    nearest = band_series(echogram, centres, depth, 5.0, 0, exclusion_depth_m=None, min_usable_fraction=0.0)
    assert np.isnan(strict[0]) and np.isfinite(strict[2])
    assert np.isfinite(nearest[0])  # the cells below the camera are used
    usable = band_usable_fraction(echogram, centres, depth, 5.0, 0)
    assert usable[0] < 0.5 and usable[2] == 1.0
    prior = lag_bounds_per_interval(np.array([2.0, 2.0]), np.array([100.0, 600.0]), (500.0, 1500.0))
    assert abs(prior.loc[0, "lag_prior_lo_s"] - np.sqrt(500**2 - 100**2) / 2) < 1e-6
    assert (
        prior.loc[1, "lag_prior_lo_s"] == 0.0
        and prior.loc[1, "lag_prior_hi_s"] < prior.loc[0, "lag_prior_hi_s"]
    )


def test_alignment_figures_handle_both_modes(tmp_path) -> None:
    from echolabel.alignment_stage import _figures

    lags = np.arange(0, 5)
    rows = []
    for mode in ("strict", "nearest"):
        for f in (38000, 70000):
            for k in lags:
                rows.append(
                    {
                        "signal": "all",
                        "frequency_hz": f,
                        "lag_intervals": k,
                        "lag_s": 10.0 * k,
                        "corr": 0.1 * k,
                        "mode": mode,
                    }
                )
    curves = pd.DataFrame(rows)
    boots = {("all", 38000): {"curves": np.tile(0.1 * lags, (6, 1)), "lag_median": 2.0}}
    paths = _figures(curves, boots, ("all", 38000), 10.0, {}, tmp_path, "test")
    assert paths and paths[0].is_file()


def test_plateaus_detrend_and_events() -> None:
    from echolabel.alignment.lag import detrend, event_lags, find_plateaus, plateau_lag_curve

    depth = np.concatenate(
        [np.linspace(0, 100, 50), 100 + 3 * np.sin(np.arange(200) / 7), np.linspace(100, 0, 50)]
    )
    runs = find_plateaus(depth, min_intervals=60, span_m=10.0)
    assert len(runs) == 1 and runs[0][0] <= 50 and runs[0][1] >= 250  # the ramps' last metres join the run
    assert find_plateaus(depth, min_intervals=60, span_m=2.0) == []
    x = np.arange(100, dtype=float)
    assert np.nanmax(np.abs(detrend(x, 9)[10:-10])) < 1e-9  # a linear trend is removed exactly
    # A band series that is a lagged copy of the camera signal gives the lag back inside the plateau.
    rng = np.random.default_rng(1)
    n, lag_true = 300, 7
    base = np.convolve(rng.normal(size=n + 20), np.ones(5) / 5, mode="same")[:n]
    cam = np.exp(base) * 3
    lags = np.arange(0, 20)
    acoustic = np.vstack([np.roll(base, -k + lag_true) + 0.05 * rng.normal(size=n) for k in lags])
    res = plateau_lag_curve(
        cam, acoustic, lags, 20, 280, detrend_intervals=18, block=12, n_boot=20, prior_s=(0, 150)
    )
    assert res["lag_best_s"] == lag_true * 10 and res["lag_prior_s"] == lag_true * 10
    assert res["lag_q025_s"] <= lag_true * 10 <= res["lag_q975_s"]
    # Events: bursts at stable depth whose strongest patch sits ``lag_true`` intervals earlier.
    bursts = np.zeros(n)
    bursts[[60, 120, 180, 240]] = 50.0
    acoustic_db = np.full((20, n), -80.0)
    for b in (60, 120, 180, 240):
        acoustic_db[lag_true, b - 1 : b + 2] = -50.0  # the patch spans the smoothed burst
    events = event_lags(
        bursts, {38000: acoustic_db}, [(20, 280)], np.full(n, 30.0), np.full(n, 150.0), n_events=3
    )
    assert len(events) == 3 and (np.abs(events["lag_s_38000"] - lag_true * 10) <= 10).all()


def test_plateau_null_separates_an_aligned_series_from_noise() -> None:
    from echolabel.alignment.lag import plateau_lag_curve

    rng = np.random.default_rng(3)
    n, lags = 240, np.arange(0, 60)
    source = rng.normal(size=n + 80)
    # The echogram leads the camera by 30 intervals: the band series at lag k and camera time t
    # is the source at t - k + 30, so the camera series matches it at lag 30.
    acoustic = np.vstack([source[np.arange(n) - k + 60] for k in lags])
    camera = np.exp(source[np.arange(n) + 30] + 0.3 * rng.normal(size=n))
    aligned = plateau_lag_curve(
        camera, acoustic, lags, 20, 220, detrend_intervals=18, n_boot=20, n_null=60, prior_s=(100, 500)
    )
    assert (
        aligned["lag_prior_s"] == 300.0
        and aligned["corr_prior"] > aligned["null_q99"]
        and aligned["p_null"] < 0.05
    )
    noise = plateau_lag_curve(
        np.exp(rng.normal(size=n)),
        acoustic,
        lags,
        20,
        220,
        detrend_intervals=18,
        n_boot=20,
        n_null=60,
        prior_s=(100, 500),
    )
    # Without a relation the observed maximum is an ordinary draw of the null maximum.
    assert noise["corr_prior"] < noise["null_q99"] + 0.1 and noise["p_null"] > 0.01
    assert aligned["n_null"] == 60
    # Under the null the maximum falls anywhere in the searched range.
    assert 100 <= noise["null_lag_q25_s"] < noise["null_lag_q50_s"] < noise["null_lag_q75_s"] <= 500
    assert noise["null_lag_q75_s"] - noise["null_lag_q25_s"] > 100
    # The search itself is unchanged by the calibration.
    plain = plateau_lag_curve(
        camera, acoustic, lags, 20, 220, detrend_intervals=18, n_boot=20, prior_s=(100, 500)
    )
    assert plain["lag_prior_s"] == aligned["lag_prior_s"] and plain["lag_median_s"] == aligned["lag_median_s"]
