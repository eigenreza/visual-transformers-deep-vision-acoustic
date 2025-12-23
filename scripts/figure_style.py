"""Shared style of the figures: fonts, class colours and the saving convention.

Every class keeps one colour in every figure. The eight most frequent classes
of the transects use the Okabe-Ito palette, the others a muted set of similar
lightness. Files are written as PNG and PDF without creation time or software
tags, so that the bytes depend on the drawing alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = [
    "maurolicus",
    "krill",
    "benthosema",
    "redfish",
    "bluewhiting",
    "gelatinous",
    "mackerel",
    "herring",
    "gonatus",
    "sergestes",
    "capelin",
    "pasiphaea",
    "barracudina",
    "unidentified",
]

LABELS = {"bluewhiting": "blue whiting"}

COLOURS = {
    "krill": "#D55E00",
    "herring": "#0072B2",
    "benthosema": "#E69F00",
    "maurolicus": "#009E73",
    "bluewhiting": "#56B4E9",
    "mackerel": "#CC79A7",
    "gonatus": "#332288",
    "gelatinous": "#F0E442",
    "redfish": "#882255",
    "sergestes": "#999933",
    "capelin": "#44AA99",
    "pasiphaea": "#AA4499",
    "barracudina": "#DDCC77",
    "unidentified": "#888888",
    "background": "#BBBBBB",
}

# Experiment name: display name, colour and marker of the three detector families.
FRAMEWORKS = {
    "rtdetrv2_l_1024": ("RT-DETRv2-L", "#000000", "o"),
    "dfine_l_1024": ("D-FINE-L", "#666666", "s"),
    "retinanet_1024": ("RetinaNet", "#A0A0A0", "^"),
}

WIDTH = 7.2


def label(name: str) -> str:
    """Display name of a class."""
    return LABELS.get(name, name)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def save(fig: matplotlib.figure.Figure, out_dir: Path, name: str, dpi: int = 200) -> list[Path]:
    """Write ``name``.png and ``name``.pdf without time or software metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pdf": {"CreationDate": None, "Creator": None, "Producer": None},
        "png": {"Software": None},
    }
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, metadata=metadata[suffix], bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    plt.close(fig)
    return paths
