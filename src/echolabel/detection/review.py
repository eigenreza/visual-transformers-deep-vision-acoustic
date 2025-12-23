"""Fixed random samples of detected frames for manual review.

For one class, frames with at least one detection are drawn at random within
bands of calibrated confidence, the boxes are drawn on the frames, and the
frames are saved as contact sheets. The sample is fixed by the seed so that a
reviewer always sees the same frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def sample_frames_by_band(
    detections: pd.DataFrame,
    class_name: str,
    bands: list[tuple[float, float]],
    n_per_band: int,
    seed: int = 0,
    score_column: str = "score_calibrated",
) -> pd.DataFrame:
    """One row per sampled frame: the frame, its band and the number of detections of the class in that band.

    A frame enters a band when its highest-confidence detection of the class
    falls in the band, so that every frame belongs to exactly one band.
    """
    d = detections[detections["class_name"] == class_name]
    if d.empty:
        return pd.DataFrame(columns=["frame", "band", "band_low", "band_high", "max_score", "n_in_band"])
    top = d.groupby("frame")[score_column].max()
    counts = d.groupby("frame")[score_column].size()
    rng = np.random.default_rng(seed)
    rows = []
    for b, (lo, hi) in enumerate(bands):
        frames = top[(top >= lo) & (top < hi)].index.to_numpy()
        take = rng.choice(frames, size=min(n_per_band, len(frames)), replace=False) if len(frames) else []
        for fr in sorted(take):
            rows.append(
                {
                    "frame": fr,
                    "band": b,
                    "band_low": lo,
                    "band_high": hi,
                    "max_score": float(top.loc[fr]),
                    "n_in_band": int(counts.loc[fr]),
                    "n_frames_in_band": int(len(frames)),
                }
            )
    return pd.DataFrame(rows)


def draw_boxes(image, boxes: pd.DataFrame, class_name: str, score_column: str = "score_calibrated"):
    """Return a copy of a PIL image with the boxes of ``class_name`` in red and other classes in grey."""
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for _, r in boxes.iterrows():
        same = r["class_name"] == class_name
        colour = (255, 40, 40) if same else (180, 180, 180)
        draw.rectangle([r["x0"], r["y0"], r["x1"], r["y1"]], outline=colour, width=3 if same else 1)
        if same:
            draw.text((r["x0"] + 2, max(r["y0"] - 12, 0)), f"{r[score_column]:.2f}", fill=colour)
    return out


def contact_sheets(
    sample: pd.DataFrame,
    detections: pd.DataFrame,
    frame_paths: dict[str, Path],
    class_name: str,
    out_dir: Path,
    per_sheet: int = 10,
    thumb_width: int = 614,
    score_column: str = "score_calibrated",
) -> list[Path]:
    """Save the sampled frames with their boxes as contact sheets, one set of sheets per band."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cols = 2
    for band, group in sample.groupby("band"):
        frames = group["frame"].tolist()
        lo, hi = float(group["band_low"].iloc[0]), float(group["band_high"].iloc[0])
        for sheet_idx, start in enumerate(range(0, len(frames), per_sheet)):
            chunk = frames[start : start + per_sheet]
            thumbs = []
            for fr in chunk:
                path = frame_paths.get(fr)
                if path is None or not Path(path).is_file():
                    log.warning("frame %s not found", fr)
                    continue
                with Image.open(path) as img:
                    boxes = detections[detections["frame"] == fr]
                    drawn = draw_boxes(img, boxes, class_name, score_column)
                scale = thumb_width / drawn.width
                thumbs.append((fr, drawn.resize((thumb_width, int(drawn.height * scale)))))
            if not thumbs:
                continue
            h = thumbs[0][1].height
            rows = int(np.ceil(len(thumbs) / cols))
            sheet = Image.new("RGB", (cols * thumb_width, rows * (h + 16)), (255, 255, 255))
            from PIL import ImageDraw

            draw = ImageDraw.Draw(sheet)
            for k, (fr, th) in enumerate(thumbs):
                x, y = (k % cols) * thumb_width, (k // cols) * (h + 16)
                sheet.paste(th, (x, y + 16))
                draw.text((x + 4, y + 2), f"{fr}  {class_name}  band {lo:.2f} to {hi:.2f}", fill=(0, 0, 0))
            p = out_dir / f"{class_name}_band{band}_{lo:.2f}-{hi:.2f}_sheet{sheet_idx}.png"
            sheet.save(p)
            written.append(p)
    return written


def review_summary(sample: pd.DataFrame, bands: list[tuple[float, float]]) -> dict[str, Any]:
    """Counts per band for the review manifest."""
    out: dict[str, Any] = {"bands": [list(b) for b in bands], "per_band": {}}
    for b, _ in enumerate(bands):
        g = sample[sample["band"] == b]
        out["per_band"][str(b)] = {
            "n_sampled": int(len(g)),
            "n_frames_available": int(g["n_frames_in_band"].iloc[0]) if len(g) else 0,
        }
    return out
