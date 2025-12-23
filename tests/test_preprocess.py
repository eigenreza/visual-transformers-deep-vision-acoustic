from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echolabel.acoustics.downsample import aggregate_cells
from echolabel.acoustics.preprocess import (
    HARD_BITS,
    MASK_BOTTOM,
    MASK_IMPULSIVE,
    MASK_LOW_SNR,
    MASK_MISSING,
    MASK_SURFACE,
    absorption_db_per_m,
    background_noise_db,
    block_summary,
    bottom_mask,
    detect_bottom,
    impulsive_ping_mask,
    noise_sv_db,
    preprocess_block,
    subtract_noise,
    surface_exclusion_m,
    to_db,
)

RNG = np.arange(0, 200.0, 0.2)


def test_absorption_matches_published_magnitudes() -> None:
    a38 = absorption_db_per_m(38000, 10.0, 35.0, 0.0, 8.0) * 1000
    assert 8.0 < a38 < 12.0  # about 10 dB per km at 38 kHz
    values = [absorption_db_per_m(f, 7.0, 35.0, 100.0, 8.0) for f in (18e3, 38e3, 70e3, 120e3, 200e3, 333e3)]
    assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))
    assert 0.035 < values[4] < 0.06  # 200 kHz between 35 and 60 dB per km


def test_surface_exclusion_scales_with_pulse() -> None:
    assert surface_exclusion_m(5.0, 0.001024) == 5.0
    assert abs(surface_exclusion_m(5.0, 0.008192) - 4 * 1500 * 0.008192 / 2) < 1e-9
    assert surface_exclusion_m(60.0, None) == 60.0


def _synthetic_bottom(n_p: int = 50) -> np.ndarray:
    db = np.full((n_p, RNG.size), -90.0)
    db[:, (RNG >= 150) & (RNG < 152)] = -20.0
    db[:, RNG >= 152] = -60.0
    db[10:20, (RNG >= 60) & (RNG < 70)] = -40.0  # a school that must not count as seabed
    return db


def test_detect_bottom_and_mask() -> None:
    db = _synthetic_bottom()
    bottom = detect_bottom(db, RNG, min_range_m=30, threshold_db=-30, smooth_pings=5)
    assert np.allclose(bottom, 150.0)
    mask = bottom_mask(RNG, bottom, margin_m=5.0)
    assert mask[:, RNG >= 145].all() and not mask[:, RNG < 145].any()
    gap = _synthetic_bottom()
    gap[30:33, RNG >= 140] = -90.0  # three pings without a seabed echo inside the stretch
    filled = detect_bottom(gap, RNG, min_range_m=30, threshold_db=-30, smooth_pings=5)
    assert np.allclose(filled, 150.0)
    none = detect_bottom(np.full((5, RNG.size), -90.0), RNG)
    assert np.isnan(none).all()
    assert not bottom_mask(RNG, none).any()


def test_background_noise_recovers_level_and_masks_noise() -> None:
    n0, alpha = -120.0, 0.05
    noise_db = noise_sv_db(np.full(120, n0), RNG, alpha)
    sv = 10.0 ** (noise_db / 10.0)
    layer = (RNG >= 30) & (RNG <= 60)
    sv[:, layer] += 1e-6  # a layer at -60 dB
    est = background_noise_db(sv, RNG, alpha, ping_block=60, range_block_m=10, min_range_m=10)
    assert np.all(np.abs(est - n0) < 0.5)
    clean, low = subtract_noise(sv, est, RNG, alpha, min_snr_db=10)
    assert low[:, RNG >= 150].all()
    assert not low[:, layer].any()
    assert np.all(np.abs(to_db(clean[:, layer]) + 60) < 0.5)


def test_impulsive_ping_mask_flags_stripes_not_schools() -> None:
    db = np.full((100, RNG.size), -90.0)
    db[:, (RNG >= 100) & (RNG <= 150)] = -50.0
    db[40] += 15.0  # one ping raised everywhere
    db[60:70, (RNG >= 60) & (RNG <= 80)] = -40.0  # a school over a fifth of the band
    bad, frac = impulsive_ping_mask(db, RNG, lo_m=50, hi_m=190, step_db=10, neighbours=2, min_fraction=0.5)
    assert bad[40] and bad.sum() == 1
    assert frac[40] > 0.9 and frac[65] < 0.3


def _params() -> dict:
    return {
        "floor_db": -130.0,
        "surface_exclusion_m": {38000: 5.0, 200000: 5.0},
        "pulse_factor": 4,
        "bottom": {
            "reference_hz": 38000,
            "min_range_m": 30,
            "threshold_db": -30,
            "smooth_pings": 5,
            "margin_m": 5,
        },
        "noise": {"ping_block": 60, "range_block_m": 10, "min_range_m": 10, "min_snr_db": 10},
        "impulsive": {
            "lo_m": 50,
            "hi_m": 190,
            "step_db": 10,
            "neighbours": 2,
            "bin_m": 1.0,
            "min_fraction": 0.5,
        },
    }


def test_preprocess_block_sets_every_bit() -> None:
    freqs = np.array([38000.0, 200000.0])
    n_p = 120
    db38 = _synthetic_bottom(n_p)
    noise200 = noise_sv_db(np.full(n_p, -125.0), RNG, 0.05)
    sv = np.stack([10.0 ** (db38 / 10.0), 10.0 ** (noise200 / 10.0)]).astype(np.float32)
    sv[1, :, (RNG >= 30) & (RNG <= 60)] += 1e-6
    sv[1, :, RNG > 180] = np.nan
    sv[0, 40] *= 10**1.5  # impulsive ping on 38 kHz
    pulse = np.full((n_p, 2), 0.001024, dtype=np.float32)
    out = preprocess_block(sv, RNG, freqs, _params(), pulse)
    mask = out["mask"]
    assert out["db"].dtype == np.float16 and out["db"].shape == sv.shape
    assert (mask[:, :, RNG < 5] & MASK_SURFACE).all()
    assert (mask[1, :, RNG > 180] & MASK_MISSING).all()
    assert (mask[:, :, RNG >= 145] & MASK_BOTTOM).all()
    assert (mask[0, 40, :] & MASK_IMPULSIVE).all() and not (mask[0, 41, :] & MASK_IMPULSIVE).any()
    assert (mask[1, :, (RNG >= 100) & (RNG <= 140)] & MASK_LOW_SNR).all()
    assert not (mask[1, :, (RNG >= 35) & (RNG <= 55)] & MASK_LOW_SNR).any()
    assert np.all(np.abs(out["noise_db"][1] + 125) < 0.5)
    summary = block_summary(out, freqs)
    assert summary["n_pings"] == n_p and abs(summary["bottom_range_m_median"] - 150) < 1
    assert 0 < summary["per_frequency"]["200000"]["valid_fraction"] < 1


def test_aggregate_cells_means_in_linear_domain() -> None:
    db = np.full((1, 4, 10), -80.0, dtype=np.float16)
    db[0, :2, :5] = -70.0
    mask = np.zeros((1, 4, 10), dtype=np.uint8)
    mask[0, 0, :5] = MASK_SURFACE  # hard bit: excluded from the mean
    mask[0, 2, 5:] = MASK_LOW_SNR  # soft bit: kept in the mean
    out_db, valid, hard = aggregate_cells(db, mask, ping_factor=2, range_factor=5)
    assert out_db.shape == (1, 2, 2)
    assert abs(float(out_db[0, 0, 0]) + 70) < 0.1  # only the unmasked ping contributes
    assert abs(float(out_db[0, 1, 1]) + 80) < 0.1
    assert valid[0, 0, 0] == 128 and valid[0, 1, 1] == 128 and valid[0, 0, 1] == 255
    assert hard[0, 0, 0] == MASK_SURFACE and hard[0, 1, 1] == 0
    assert HARD_BITS & MASK_LOW_SNR == 0


SAMPLE = Path("data/samples/source_data/S2018105_sv.zarr")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sampled store not present")
def test_preprocess_block_on_sampled_store() -> None:
    import xarray as xr

    ds = xr.open_zarr(str(SAMPLE), consolidated=False).isel(ping_time=slice(0, 240))
    sv = ds["sv"].transpose("frequency", "ping_time", "range").values
    rng = ds["range"].values
    freqs = ds["frequency"].values
    params = {
        "floor_db": -130.0,
        "absorption": {"temperature_c": 7.0, "salinity_psu": 35.0, "depth_m": 100.0, "ph": 8.0},
        "surface_exclusion_m": {18000: 60, 38000: 5, 70000: 5, 120000: 5, 200000: 5, 333000: 5},
        "pulse_factor": 4,
        "bottom": {
            "reference_hz": 38000,
            "min_range_m": 280,
            "threshold_db": -30,
            "smooth_pings": 31,
            "margin_m": 5,
        },
        "noise": {
            "ping_block": 60,
            "range_block_m": 10,
            "min_range_m": 10,
            "min_snr_db": 10,
            "max_noise_db": {18000: -151, 38000: -157, 70000: -158, 120000: -148, 200000: -127, 333000: -125},
        },
        "impulsive": {
            "lo_m": 50,
            "hi_m": 300,
            "step_db": 10,
            "neighbours": 2,
            "bin_m": 1.0,
            "min_fraction": 0.5,
        },
    }
    out = preprocess_block(sv, rng, freqs, params, ds["pulse_length"].values)
    summary = block_summary(out, freqs)
    n200 = summary["per_frequency"]["200000"]["noise_db_median"]
    assert -136 < n200 < -122  # the time-varied gain ramp observed at 200 kHz implies about -129 dB
    assert summary["per_frequency"]["38000"]["valid_fraction"] > 0.5
    assert summary["per_frequency"]["38000"]["noise_capped_fraction"] > 0.5
    assert summary["per_frequency"]["200000"]["noise_capped_fraction"] < 0.5
    assert summary["per_frequency"]["18000"]["surface"] > 0.15


def test_camera_band_fractions_count_bits_in_band() -> None:
    import pandas as pd

    from echolabel.preprocess_stage import camera_band_fractions

    freqs = np.array([38000.0, 200000.0])
    n_p = 120
    db38 = _synthetic_bottom(n_p)
    noise200 = noise_sv_db(np.full(n_p, -125.0), RNG, 0.05)
    sv = np.stack([10.0 ** (db38 / 10.0), 10.0 ** (noise200 / 10.0)]).astype(np.float32)
    sv[1, :, (RNG >= 30) & (RNG <= 60)] += 1e-6
    sv[0, 40] *= 10**1.5
    out = preprocess_block(sv, RNG, freqs, _params(), np.full((n_p, 2), 0.001024, dtype=np.float32))
    ping_time = np.datetime64("2018-05-09T20:00:00") + np.arange(n_p) * np.timedelta64(1000, "ms")
    camera = pd.DataFrame({"timestamp": pd.to_datetime(ping_time[::5]), "depth_clean": 108.5})
    band = camera_band_fractions(out["mask"], ping_time, RNG, 8.5, freqs, camera, half_width_m=25.0)
    assert band["n_pings_with_camera_depth"] == n_p
    f38, f200 = band["per_frequency"]["38000"], band["per_frequency"]["200000"]
    assert abs(f38["hard_valid_fraction"] - (n_p - 1) / n_p) < 1e-9  # only the impulsive ping is lost
    assert f200["hard_valid_fraction"] > 0.99 and f200["valid_fraction"] < 0.5  # below the SNR threshold
    assert f200["low_snr_fraction"] > 0.5


def test_aggregate_time_cells_matches_fixed_cells_for_regular_pings() -> None:
    from echolabel.acoustics.downsample import aggregate_cells, aggregate_time_cells

    rng = np.random.default_rng(0)
    db = rng.uniform(-90, -50, size=(2, 12, 20)).astype(np.float32)
    mask = np.zeros((2, 12, 20), dtype=np.uint8)
    mask[0, 3, :5] = 1
    mask[1, :, 10:] = 8
    cells = np.repeat(np.arange(6), 2)
    a = aggregate_cells(db, mask, 2, 5)
    b = aggregate_time_cells(db, mask, cells, 6, 5)
    assert np.allclose(a[0].astype(float), b[0].astype(float), atol=0.01)
    assert (a[1] == b[1]).all() and (a[2] == b[2]).all()
    # A cell without pings is missing.
    gap = aggregate_time_cells(db[:, :10], mask[:, :10], np.repeat(np.arange(5), 2), 6, 5)
    assert (
        np.isnan(gap[0][:, 5].astype(float)).all() and (gap[1][:, 5] == 0).all() and (gap[2][:, 5] & 1).all()
    )
