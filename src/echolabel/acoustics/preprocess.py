"""Preprocessing of the backscatter: decibels, masks and background noise.

Every acoustic product carries, next to the volume backscattering strength in
decibels, a per-sample mask that records why a value should not be trusted:
missing in the store, within the near-surface ringdown, at or below the
seabed, below the noise floor, or within an impulsive-noise ping. The mask is
a bit field so that later stages choose which reasons to apply.

Background noise is estimated as follows: the received
power with the time-varied gain removed is averaged over blocks of pings and
range, the quietest range block gives the noise level of the ping block, the
noise is projected back through the gain and subtracted in the linear domain,
and samples whose signal-to-noise ratio falls below a threshold are flagged.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

MASK_MISSING = 1
MASK_SURFACE = 2
MASK_BOTTOM = 4
MASK_LOW_SNR = 8
MASK_IMPULSIVE = 16

MASK_BITS = {
    "missing": MASK_MISSING,
    "surface": MASK_SURFACE,
    "bottom": MASK_BOTTOM,
    "low_snr": MASK_LOW_SNR,
    "impulsive": MASK_IMPULSIVE,
}
# Bits that exclude a sample from any average; a low signal-to-noise ratio only lowers the value.
HARD_BITS = MASK_MISSING | MASK_SURFACE | MASK_BOTTOM | MASK_IMPULSIVE


def to_db(sv: np.ndarray, floor_db: float = -130.0) -> np.ndarray:
    """10 log10 of linear sv with a floor; NaN is preserved."""
    floor = 10.0 ** (floor_db / 10.0)
    with np.errstate(invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(sv, floor))
    out[np.isnan(sv)] = np.nan
    return out


def absorption_db_per_m(
    frequency_hz: float,
    temperature_c: float = 7.0,
    salinity_psu: float = 35.0,
    depth_m: float = 100.0,
    ph: float = 8.0,
) -> float:
    """Sea-water absorption in dB per metre from temperature, salinity, depth, pH and frequency."""
    f = frequency_hz / 1000.0  # kHz
    t, s, d = temperature_c, salinity_psu, depth_m
    c = 1412.0 + 3.21 * t + 1.19 * s + 0.0167 * d
    theta = 273.0 + t
    # Boric acid
    a1 = 8.86 / c * 10.0 ** (0.78 * ph - 5.0)
    p1 = 1.0
    f1 = 2.8 * np.sqrt(s / 35.0) * 10.0 ** (4.0 - 1245.0 / theta)
    # Magnesium sulphate
    a2 = 21.44 * s / c * (1.0 + 0.025 * t)
    p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d * d
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / theta) / (1.0 + 0.0018 * (s - 35.0))
    # Pure water
    if t <= 20.0:
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t * t - 1.5e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t * t - 6.5e-10 * t**3
    p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d * d
    alpha_db_per_km = (
        a1 * p1 * f1 * f * f / (f * f + f1 * f1) + a2 * p2 * f2 * f * f / (f * f + f2 * f2) + a3 * p3 * f * f
    )
    return float(alpha_db_per_km / 1000.0)


def surface_exclusion_m(
    base_m: float, pulse_length_s: float | None, pulse_factor: float = 4.0, sound_speed_m_s: float = 1500.0
) -> float:
    """Range below the transducer to exclude: the larger of a fixed value and a number of pulse lengths."""
    if pulse_length_s is None or not np.isfinite(pulse_length_s):
        return float(base_m)
    return float(max(base_m, pulse_factor * sound_speed_m_s * pulse_length_s / 2.0))


def detect_bottom(
    db_ref: np.ndarray,
    rng: np.ndarray,
    min_range_m: float = 30.0,
    max_range_m: float | None = None,
    threshold_db: float = -30.0,
    smooth_pings: int = 31,
    max_jump_m: float = 20.0,
) -> np.ndarray:
    """Seabed range per ping from the reference channel, NaN where no seabed is found.

    A ping has a seabed when its strongest return between ``min_range_m`` and
    ``max_range_m`` reaches ``threshold_db``. The seabed range is the start of
    the contiguous run of samples above the threshold that contains that peak.
    Ranges are median-filtered over ``smooth_pings`` pings, single pings that
    depart from the filtered line by more than ``max_jump_m`` take the
    filtered value, and gaps of up to ``smooth_pings`` pings inside a detected
    stretch are interpolated. ``db_ref`` has shape (P, R).
    """
    n_p, _ = db_ref.shape
    sel = rng >= min_range_m
    if max_range_m is not None:
        sel &= rng <= max_range_m
    block = np.where(np.isnan(db_ref[:, sel]), -np.inf, db_ref[:, sel])
    r_sel = rng[sel]
    candidate = np.full(n_p, np.nan)
    if block.shape[1] == 0:
        return candidate
    peak_idx = block.argmax(axis=1)
    peak_val = block[np.arange(n_p), peak_idx]
    above = block >= threshold_db
    for p in np.flatnonzero(peak_val >= threshold_db):
        start = peak_idx[p]
        row = above[p]
        while start > 0 and row[start - 1]:
            start -= 1
        candidate[p] = r_sel[start]
    series = pd.Series(candidate)
    smoothed = series.rolling(smooth_pings, center=True, min_periods=1).median()
    jump = (series - smoothed).abs() > max_jump_m
    out = series.where(~jump, smoothed)
    # Pings that missed the seabed inside a detected stretch take the interpolated range.
    return out.interpolate(limit=smooth_pings, limit_area="inside").to_numpy()


def bottom_mask(rng: np.ndarray, bottom_range: np.ndarray, margin_m: float = 5.0) -> np.ndarray:
    """Boolean (P, R): samples at or beyond the seabed range minus ``margin_m``; False where no seabed."""
    limit = bottom_range[:, None] - margin_m
    with np.errstate(invalid="ignore"):
        mask = rng[None, :] >= limit
    mask[np.isnan(bottom_range)] = False
    return mask


def _tvg_db(rng: np.ndarray, alpha_db_per_m: float) -> np.ndarray:
    """Time-varied gain applied to volume backscatter: 20 log10 r plus two-way absorption."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rng, 1e-3)) + 2.0 * alpha_db_per_m * rng


def background_noise_db(
    sv: np.ndarray,
    rng: np.ndarray,
    alpha_db_per_m: float,
    ping_block: int = 60,
    range_block_m: float = 10.0,
    min_range_m: float = 10.0,
    max_noise_db: float | None = None,
) -> np.ndarray:
    """Background noise per ping, as received power at one metre in dB, with shape (P,).

    ``sv`` is linear volume backscatter of one channel with shape (P, R). The
    power with the gain removed is averaged in the linear domain over blocks of
    ``ping_block`` pings and ``range_block_m`` metres beyond ``min_range_m``;
    the quietest range block gives the block's noise level, capped at
    ``max_noise_db`` when given. Blocks without finite samples give NaN.
    """
    n_p, n_r = sv.shape
    tvg = _tvg_db(rng, alpha_db_per_m)
    power = (sv / (10.0 ** (tvg / 10.0))[None, :]).astype(np.float64)
    power[:, rng < min_range_m] = np.nan
    step = float(np.median(np.diff(rng))) if n_r > 1 else range_block_m
    r_block = max(1, int(round(range_block_m / step)))
    n_rb = n_r // r_block
    out = np.full(n_p, np.nan)
    if n_rb == 0:
        return out
    for p0 in range(0, n_p, ping_block):
        chunk = power[p0 : p0 + ping_block, : n_rb * r_block]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(chunk.reshape(chunk.shape[0], n_rb, r_block), axis=(0, 2))
        if np.all(np.isnan(means)):
            continue
        noise = 10.0 * np.log10(max(float(np.nanmin(means)), 1e-30))
        if max_noise_db is not None:
            noise = min(noise, float(max_noise_db))
        out[p0 : p0 + ping_block] = noise
    return out


def noise_sv_db(noise_db: np.ndarray, rng: np.ndarray, alpha_db_per_m: float) -> np.ndarray:
    """Noise projected through the gain, as volume backscatter in dB with shape (P, R)."""
    return noise_db[:, None] + _tvg_db(rng, alpha_db_per_m)[None, :]


def subtract_noise(
    sv: np.ndarray, noise_db: np.ndarray, rng: np.ndarray, alpha_db_per_m: float, min_snr_db: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-corrected linear sv (floored at zero) and the mask of samples below ``min_snr_db``.

    Pings without a noise estimate are returned unchanged and unmasked.
    """
    has_noise = np.isfinite(noise_db)
    noise_lin = 10.0 ** (noise_sv_db(np.where(has_noise, noise_db, -300.0), rng, alpha_db_per_m) / 10.0)
    with np.errstate(all="ignore"):
        clean = np.where(has_noise[:, None], np.maximum(sv - noise_lin, 0.0), sv).astype(np.float32)
        snr = 10.0 * np.log10(sv / noise_lin)
    low = (snr < min_snr_db) & has_noise[:, None] & np.isfinite(sv)
    clean[np.isnan(sv)] = np.nan
    return clean, low


def impulsive_ping_mask(
    db: np.ndarray,
    rng: np.ndarray,
    lo_m: float = 50.0,
    hi_m: float = 450.0,
    step_db: float = 10.0,
    neighbours: int = 2,
    bin_m: float = 1.0,
    min_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pings dominated by impulsive noise, with the per-ping fraction of impulsive samples.

    A sample is impulsive when, after averaging to ``bin_m`` vertical bins, it
    exceeds both the ping ``neighbours`` before and the ping ``neighbours``
    after by more than ``step_db``. A ping is flagged when the fraction of
    such samples between ``lo_m`` and ``hi_m`` reaches ``min_fraction``. The
    first and last pings compare with the one available neighbour.
    """
    n_p, n_r = db.shape
    step = float(np.median(np.diff(rng))) if n_r > 1 else bin_m
    k = max(1, int(round(bin_m / step)))
    sel = (rng >= lo_m) & (rng <= hi_m)
    band = db[:, sel]
    n_b = band.shape[1] // k
    if n_b == 0 or n_p < 2:
        return np.zeros(n_p, dtype=bool), np.zeros(n_p)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lin = 10.0 ** (band[:, : n_b * k].astype(np.float32) / 10.0)
        binned = 10.0 * np.log10(np.nanmean(lin.reshape(n_p, n_b, k), axis=2))
    before = np.full_like(binned, np.nan)
    after = np.full_like(binned, np.nan)
    before[neighbours:] = binned[:-neighbours]
    after[:-neighbours] = binned[neighbours:]
    before = np.where(np.isnan(before), after, before)
    after = np.where(np.isnan(after), before, after)
    with np.errstate(invalid="ignore"):
        hit = (binned - before > step_db) & (binned - after > step_db)
    finite = np.isfinite(binned) & np.isfinite(before) & np.isfinite(after)
    n_finite = finite.sum(axis=1)
    frac = np.where(n_finite > 0, hit.sum(axis=1) / np.maximum(n_finite, 1), 0.0)
    return frac >= min_fraction, frac


def preprocess_block(
    sv: np.ndarray,
    rng: np.ndarray,
    freqs: np.ndarray,
    params: dict[str, Any],
    pulse_length_s: np.ndarray | None = None,
    bottom_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply every filter to one block of pings at all frequencies.

    ``sv`` has shape (F, P, R) in linear units, ``pulse_length_s`` shape (P, F)
    or None. ``bottom_override`` replaces entries of the seabed settings for
    one transect; the value False disables seabed detection. The result holds
    the noise-corrected backscatter in dB as float16 (``db``), the mask bit
    field (``mask``), the seabed range per ping (``bottom_range_m``), the
    noise level per frequency and ping (``noise_db``), the impulsive fraction
    per frequency and ping, and the absorption used per frequency.
    """
    n_f, n_p, n_r = sv.shape
    floor_db = float(params.get("floor_db", -130.0))
    sound_speed = float(params.get("sound_speed_m_s", 1500.0))
    absorption_env = dict(params.get("absorption", {}))
    surface_base = {float(k): float(v) for k, v in params.get("surface_exclusion_m", {}).items()}
    pulse_factor = float(params.get("pulse_factor", 4.0))
    bottom_cfg = dict(params.get("bottom") or {})
    if bottom_override is False:
        bottom_cfg = {}
    elif bottom_override:
        bottom_cfg.update(bottom_override)
    noise_cfg = params.get("noise")
    impulsive_cfg = params.get("impulsive")

    mask = np.zeros((n_f, n_p, n_r), dtype=np.uint8)
    mask[np.isnan(sv)] |= MASK_MISSING
    clean = np.array(sv, dtype=np.float32, copy=True)
    noise_db = np.full((n_f, n_p), np.nan, dtype=np.float32)
    noise_raw_db = np.full((n_f, n_p), np.nan, dtype=np.float32)
    impulsive_fraction = np.zeros((n_f, n_p), dtype=np.float32)
    alphas = np.array([absorption_db_per_m(float(f), **absorption_env) for f in freqs])
    exclusions = np.zeros(n_f, dtype=np.float32)

    bottom_range = np.full(n_p, np.nan)
    if bottom_cfg:
        ref_hz = float(bottom_cfg.get("reference_hz", 38000.0))
        ref = int(np.argmin(np.abs(np.asarray(freqs, dtype=float) - ref_hz)))
        detect_kwargs = {k: v for k, v in bottom_cfg.items() if k not in ("reference_hz", "margin_m")}
        bottom_range = detect_bottom(to_db(sv[ref], floor_db), rng, **detect_kwargs)
        bmask = bottom_mask(rng, bottom_range, margin_m=float(bottom_cfg.get("margin_m", 5.0)))
        mask[:, bmask] |= MASK_BOTTOM

    for i, f in enumerate(freqs):
        f = float(f)
        pulse = None
        if pulse_length_s is not None:
            col = pulse_length_s[:, i]
            pulse = float(np.nanmax(col)) if np.isfinite(col).any() else None
        excl = surface_exclusion_m(surface_base.get(f, 0.0), pulse, pulse_factor, sound_speed)
        exclusions[i] = excl
        if excl > 0:
            mask[i, :, rng < excl] |= MASK_SURFACE
        if impulsive_cfg:
            bad, frac = impulsive_ping_mask(to_db(sv[i], floor_db), rng, **impulsive_cfg)
            mask[i, bad, :] |= MASK_IMPULSIVE
            impulsive_fraction[i] = frac
        if noise_cfg:
            est_kwargs = {k: v for k, v in noise_cfg.items() if k not in ("min_snr_db", "max_noise_db")}
            caps = noise_cfg.get("max_noise_db")
            cap = caps.get(int(f), caps.get(str(int(f)))) if isinstance(caps, dict) else caps
            min_range = max(float(est_kwargs.pop("min_range_m", 10.0)), excl)
            hard = (mask[i] & HARD_BITS) > 0
            noise_raw_db[i] = background_noise_db(
                np.where(hard, np.nan, sv[i]), rng, alphas[i], min_range_m=min_range, **est_kwargs
            )
            noise_db[i] = noise_raw_db[i] if cap is None else np.fmin(noise_raw_db[i], float(cap))
            clean[i], low = subtract_noise(
                sv[i], noise_db[i], rng, alphas[i], float(noise_cfg.get("min_snr_db", 10.0))
            )
            mask[i][low] |= MASK_LOW_SNR
    db = to_db(clean, floor_db).astype(np.float16)
    return {
        "db": db,
        "mask": mask,
        "bottom_range_m": bottom_range.astype(np.float32),
        "noise_db": noise_db,
        "noise_raw_db": noise_raw_db,
        "impulsive_fraction": impulsive_fraction,
        "absorption_db_per_m": alphas,
        "surface_exclusion_m": exclusions,
    }


def block_summary(result: dict[str, Any], freqs: np.ndarray) -> dict[str, Any]:
    """Fractions of masked samples per reason and frequency, plus seabed and noise statistics."""
    mask = result["mask"]
    bottom = result["bottom_range_m"]
    out: dict[str, Any] = {
        "n_pings": int(mask.shape[1]),
        "bottom_detected_fraction": float(np.isfinite(bottom).mean()) if bottom.size else 0.0,
        "bottom_range_m_median": float(np.nanmedian(bottom)) if np.isfinite(bottom).any() else None,
        "per_frequency": {},
    }
    for i, f in enumerate(freqs):
        m = mask[i]
        nd = result["noise_db"][i]
        raw = result.get("noise_raw_db", result["noise_db"])[i]
        has = bool(np.isfinite(nd).any())
        out["per_frequency"][str(int(f))] = {
            "valid_fraction": float((m == 0).mean()),
            **{name: float(((m & bit) > 0).mean()) for name, bit in MASK_BITS.items()},
            "impulsive_pings_fraction": float(((m[:, -1] & MASK_IMPULSIVE) > 0).mean()),
            "noise_db_median": float(np.nanmedian(nd)) if has else None,
            "noise_db_q10": float(np.nanquantile(nd, 0.1)) if has else None,
            "noise_db_q90": float(np.nanquantile(nd, 0.9)) if has else None,
            "noise_raw_db_median": float(np.nanmedian(raw)) if has else None,
            "noise_capped_fraction": float(np.mean(raw[np.isfinite(raw)] > nd[np.isfinite(raw)] + 1e-3))
            if has
            else None,
            "absorption_db_per_m": float(result["absorption_db_per_m"][i]),
        }
    return out
