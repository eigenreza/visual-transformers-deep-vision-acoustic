from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.detection.calibration import (
    Calibrator,
    calibration_by_slice,
    expected_calibration_error,
    logit,
    reliability_table,
    sigmoid,
)


def _miscalibrated(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.02, 0.98, n)
    # True probability is sharper than the score: logit scaled by 2.
    p_true = sigmoid(2.0 * logit(scores))
    y = (rng.uniform(size=n) < p_true).astype(float)
    return scores, y


def test_temperature_and_platt_recover_sharpening() -> None:
    scores, y = _miscalibrated()
    before = expected_calibration_error(scores, y)
    temp = Calibrator.fit("temperature", scores, y)
    assert 0.4 < temp.params["temperature"] < 0.65
    after = expected_calibration_error(temp.apply(scores), y)
    assert after < before / 2
    platt = Calibrator.fit("platt", scores, y)
    assert 1.6 < platt.params["slope"] < 2.4 and abs(platt.params["intercept"]) < 0.3
    iso = Calibrator.fit("isotonic", scores, y)
    iso_after = expected_calibration_error(iso.apply(scores), y)
    assert iso_after < before / 2
    restored = Calibrator.from_dict(iso.to_dict())
    assert np.allclose(restored.apply(scores[:10]), iso.apply(scores[:10]))
    assert np.all(np.diff(iso.apply(np.linspace(0.02, 0.98, 50))) >= -1e-12)


def test_reliability_and_slices() -> None:
    scores, y = _miscalibrated(n=2000)
    table = reliability_table(scores, y, n_bins=10)
    assert len(table) == 10 and table["count"].sum() == 2000
    matched = pd.DataFrame({"score": scores, "tp": y.astype(bool), "year": ["2018"] * 1000 + ["2022"] * 1000})
    cals = {"identity": Calibrator("identity", {}), "temp": Calibrator.fit("temperature", scores, y)}
    slices = {"all": pd.Series(True, index=matched.index), "2018": matched["year"] == "2018"}
    out = calibration_by_slice(matched, cals, slices)
    assert set(out["slice"]) == {"all", "2018"} and set(out["calibrator"]) == {"identity", "temp"}
    ece_identity = out[(out["calibrator"] == "identity") & (out["slice"] == "all")]["ece"].item()
    ece_temp = out[(out["calibrator"] == "temp") & (out["slice"] == "all")]["ece"].item()
    assert ece_temp < ece_identity
