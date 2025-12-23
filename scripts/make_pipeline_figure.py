"""Draw the overview figure of the pipeline for the README.

Usage: python scripts/make_pipeline_figure.py [output directory]
Writes ``pipeline.pdf`` and ``pipeline.png`` (default: ``docs/figures``).
The detector pipeline is drawn with solid boxes, the acoustic branch that
is future work with dashed ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

CAMERA, FUTURE = "#f4e3c1", "#eeeeee"

# Label, column, row, part of the future work. Columns run left to right, rows top to bottom.
BOXES = {
    "annotations": ("Annotated frames\n42 raw labels, 66 tows", 0, 0, False),
    "dataset": ("Annotation dataset\n14 classes,\nsplits by tow", 1, 0, False),
    "detector": ("Object detector\ntrained, compared,\ncalibrated", 2, 0, False),
    "inference": ("Detections on every\ntransect frame,\nfixed structure removed", 3, 0, False),
    "series": ("Camera series\ncounts and depth\nper 10 s", 4, 0, False),
    "store": ("Echosounder record\nsix channels", 1, 1.55, True),
    "preprocess": ("Acoustic\npreprocessing", 2, 1.55, True),
    "alignment": ("Alignment with\ntrawl position\nand time", 3, 1.55, True),
    "models": ("Acoustic species\nmodels supervised\nby the detections", 4, 1.55, True),
}
ARROWS = [
    ("annotations", "dataset"),
    ("dataset", "detector"),
    ("detector", "inference"),
    ("inference", "series"),
    ("store", "preprocess"),
    ("preprocess", "alignment"),
    ("alignment", "models"),
    ("series", "models"),
]
WIDTH, HEIGHT, DX, DY = 1.6, 0.8, 2.0, 1.05


def centre(name: str) -> tuple[float, float]:
    _, col, row, _ = BOXES[name]
    return col * DX, -row * DY


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    for name, (label, _col, _row, future) in BOXES.items():
        x, y = centre(name)
        ax.add_patch(
            FancyBboxPatch(
                (x - WIDTH / 2, y - HEIGHT / 2),
                WIDTH,
                HEIGHT,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=FUTURE if future else CAMERA,
                edgecolor="0.45" if future else "0.3",
                linewidth=0.8,
                linestyle="--" if future else "-",
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=7.6, color="0.35" if future else "k")
    for a, b in ARROWS:
        (xa, ya), (xb, yb) = centre(a), centre(b)
        future = BOXES[a][3] or BOXES[b][3]
        if abs(ya - yb) < 1e-6:
            start, end = (xa + WIDTH / 2, ya), (xb - WIDTH / 2, yb)
        else:
            start, end = (xa, ya - HEIGHT / 2), (xb, yb + HEIGHT / 2)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color="0.5" if future else "0.3",
                linewidth=0.8,
                linestyle="--" if future else "-",
            )
        )
    # Frame around the branch that is future work.
    left, right = centre("store")[0] - WIDTH / 2 - 0.2, centre("models")[0] + WIDTH / 2 + 0.2
    bottom, top = centre("store")[1] - HEIGHT / 2 - 0.2, centre("store")[1] + HEIGHT / 2 + 0.2
    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor="0.6",
            linewidth=0.7,
            linestyle=":",
        )
    )
    ax.text(left + 0.08, top + 0.07, "future work", fontsize=8, color="0.4", style="italic", va="bottom")
    ax.text(
        -WIDTH / 2,
        HEIGHT / 2 + 0.12,
        "detector pipeline",
        fontsize=8,
        color="0.3",
        style="italic",
        va="bottom",
    )
    ax.set_xlim(-WIDTH / 2 - 0.15, 4 * DX + WIDTH / 2 + 0.35)
    ax.set_ylim(bottom - 0.15, HEIGHT / 2 + 0.45)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    # No creation time or software tag is written, so the files depend on the drawing alone.
    metadata = {"pdf": {"CreationDate": None, "Creator": None, "Producer": None}, "png": {"Software": None}}
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"pipeline.{suffix}", dpi=170, metadata=metadata[suffix])
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "figures")
