from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from echolabel.detection.review import contact_sheets, review_summary, sample_frames_by_band


def _detections(n_frames: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_frames):
        frame = f"f{i:03d}"
        for _ in range(rng.integers(1, 4)):
            rows.append(
                {
                    "frame": frame,
                    "class_name": "krill" if rng.random() < 0.8 else "herring",
                    "score_calibrated": float(rng.uniform(0.05, 1.0)),
                    "x0": 10.0,
                    "y0": 10.0,
                    "x1": 60.0,
                    "y1": 50.0,
                }
            )
    return pd.DataFrame(rows)


def test_sampling_is_fixed_and_bands_are_exclusive() -> None:
    dets = _detections()
    bands = [(0.05, 0.3), (0.3, 0.6), (0.6, 1.01)]
    a = sample_frames_by_band(dets, "krill", bands, n_per_band=5, seed=3)
    b = sample_frames_by_band(dets, "krill", bands, n_per_band=5, seed=3)
    assert a.equals(b)
    assert a["frame"].is_unique and a.groupby("band").size().max() <= 5
    for _, r in a.iterrows():
        assert r["band_low"] <= r["max_score"] < r["band_high"]
    assert sample_frames_by_band(dets, "capelin", bands, 5).empty


def test_contact_sheets_are_written(tmp_path) -> None:
    dets = _detections(12)
    bands = [(0.05, 0.5), (0.5, 1.01)]
    sample = sample_frames_by_band(dets, "krill", bands, n_per_band=4, seed=0)
    paths = {}
    for fr in sample["frame"]:
        p = tmp_path / f"{fr}.jpg"
        Image.new("RGB", (120, 100), (30, 60, 90)).save(p)
        paths[fr] = p
    sheets = contact_sheets(sample, dets, paths, "krill", tmp_path / "sheets", per_sheet=3, thumb_width=120)
    assert sheets and all(p.is_file() for p in sheets)
    summary = review_summary(sample, bands)
    assert set(summary["per_band"]) == {"0", "1"}
