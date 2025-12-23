"""Figures of the detector evaluation from the result tables under ``results/``.

Six figures for one experiment and seed: the normalised confusion matrix on the
test split, precision-recall curves per class, F1 against calibrated confidence
with the operating thresholds, the reliability diagram before and after
calibration, per-class AP50 with bootstrap intervals on the test split and the
2024 hold-out, and the comparison of the three detector families.

Usage: python scripts/make_detector_figures.py [--results results]
       [--experiment rtdetrv2_l_1024] [--seed 0] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
from figure_style import plt  # noqa: E402


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def confusion_matrix(run: Path, out: Path) -> None:
    table = pd.read_csv(run / "eval_test" / "confusion.csv", index_col=0)
    order = [*fs.CLASSES, "background"]
    table = table.loc[order, order]
    counts = table.to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    fig, ax = plt.subplots(figsize=(fs.WIDTH, fs.WIDTH * 0.92))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    names = [fs.label(c) for c in order]
    ax.set_xticks(range(len(order)), names, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(order)), names)
    ax.set_xlabel("predicted class (background: box not detected)")
    ax.set_ylabel("true class (background: detection without a box)")
    for i in range(len(order)):
        for j in range(len(order)):
            if counts[i, j] == 0:
                continue
            v = share[i, j]
            ax.text(
                j,
                i,
                f"{100 * v:.0f}" if v >= 0.005 else "<1",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if v > 0.55 else "black",
            )
    ax.text(
        0.0,
        1.01,
        "percent of the row; boxes per true class on the right",
        transform=ax.transAxes,
        fontsize=7,
        color="0.3",
        va="bottom",
    )
    for i, n in enumerate(totals[:, 0]):
        ax.text(len(order) - 0.4, i, f"{int(n):,}", va="center", ha="left", fontsize=6, color="0.3")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    fs.save(fig, out, "confusion_test")


def grid(n: int, ncols: int = 5) -> tuple[int, int]:
    return int(np.ceil(n / ncols)), ncols


def pr_curves(run: Path, out: Path) -> None:
    curves = pd.read_csv(run / "eval_test" / "curves.csv")
    slices = pd.read_csv(run / "eval_test" / "slices.csv")
    rows = slices.loc[slices.slice == "all", ["metric", "value"]]
    ap = {m.replace("ap50_", ""): v for m, v in rows.itertuples(index=False) if m.startswith("ap50_")}
    nrows, ncols = grid(len(fs.CLASSES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fs.WIDTH, fs.WIDTH * 0.66), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, fs.CLASSES, strict=False):
        c = curves[curves.class_name == cls].sort_values("threshold")
        n_boxes = int((c.tp + c.fn).iloc[0])
        ax.plot(c.recall, c.precision, color=fs.COLOURS[cls], drawstyle="steps-post")
        ax.set_title(fs.label(cls), pad=2)
        text = f"AP50 {ap[cls]:.2f}\n{n_boxes:,} boxes" if cls in ap else f"{n_boxes:,} boxes"
        ax.text(0.04, 0.06, text, transform=ax.transAxes, fontsize=6.5, va="bottom", color="0.25")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True, color="0.92", linewidth=0.5)
    for ax in axes.flat[len(fs.CLASSES) :]:
        ax.axis("off")
    fig.supxlabel("recall", fontsize=8)
    fig.supylabel("precision", fontsize=8)
    fig.subplots_adjust(hspace=0.45, wspace=0.18, bottom=0.1, left=0.08)
    fs.save(fig, out, "pr_curves_test")


def f1_curves(run: Path, out: Path) -> None:
    curves = pd.read_csv(run / "eval_test" / "curves.csv")
    with open(run / "calibration" / "calibration.json", encoding="utf-8") as fh:
        calib = json.load(fh)["calibrator"]
    thresholds = pd.read_csv(run / "calibration" / "thresholds_calibrated.csv").set_index("class_name")
    params = calib["params"]
    if calib["method"] == "platt":

        def to_calibrated(s: np.ndarray) -> np.ndarray:
            return sigmoid(params["slope"] * logit(s) + params["intercept"])

    elif calib["method"] == "temperature":

        def to_calibrated(s: np.ndarray) -> np.ndarray:
            return sigmoid(logit(s) / params["temperature"])

    else:
        raise SystemExit(f"calibration method {calib['method']} has no closed form here")
    nrows, ncols = grid(len(fs.CLASSES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fs.WIDTH, fs.WIDTH * 0.66), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, fs.CLASSES, strict=False):
        c = curves[curves.class_name == cls].sort_values("threshold")
        x = to_calibrated(c.threshold.to_numpy())
        ax.plot(x, c.f1, color=fs.COLOURS[cls])
        t = float(thresholds.loc[cls, "threshold"])
        if t < 1.0:
            f1_at = float(np.interp(t, x, c.f1))
            ax.axvline(t, color="0.3", linewidth=0.7, linestyle=(0, (3, 2)))
            right = t <= 0.55
            ax.text(
                t + 0.04 if right else t - 0.04,
                0.08,
                f"t = {t:.2f}\nF1 {f1_at:.2f}",
                fontsize=6.5,
                va="bottom",
                ha="left" if right else "right",
                color="0.25",
            )
        else:
            ax.text(
                0.04,
                0.9,
                "no threshold:\nno validation boxes",
                fontsize=6.5,
                va="top",
                color="0.25",
                transform=ax.transAxes,
            )
        ax.set_title(fs.label(cls), pad=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True, color="0.92", linewidth=0.5)
    for ax in axes.flat[len(fs.CLASSES) :]:
        ax.axis("off")
    fig.supxlabel("calibrated confidence threshold", fontsize=8)
    fig.supylabel("F1", fontsize=8)
    fig.subplots_adjust(hspace=0.45, wspace=0.18, bottom=0.1, left=0.08)
    fs.save(fig, out, "f1_thresholds_test")


def reliability(run: Path, out: Path) -> None:
    table = pd.read_csv(run / "calibration" / "reliability.csv")
    with open(run / "calibration" / "calibration.json", encoding="utf-8") as fh:
        chosen = json.load(fh)["chosen"]
    sl = "all_above_0.05"
    before = table[(table.calibrator == "identity") & (table.slice == sl)]
    after = table[(table.calibrator == chosen) & (table.slice == sl)]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(fs.WIDTH, fs.WIDTH * 0.42), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.7, linestyle=(0, (3, 2)))
    series = ((before, "raw score", "#999999", "s"), (after, "after Platt scaling", "#0072B2", "o"))
    for part, name, colour, marker in series:
        p = part[part["count"] > 0]
        ax.plot(p.mean_confidence, p.fraction_positive, color=colour, marker=marker, markersize=4, label=name)
    ax.set_xlabel("mean confidence in the bin")
    ax.set_ylabel("fraction of true positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    width = float((before.bin_high - before.bin_low).iloc[0])
    centres = (before.bin_low + before.bin_high) / 2
    ax2.bar(centres - width * 0.22, before["count"], width=width * 0.42, color="#999999", label="raw score")
    ax2.bar(
        centres + width * 0.22,
        after["count"],
        width=width * 0.42,
        color="#0072B2",
        label="after Platt scaling",
    )
    ax2.set_yscale("log")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("detections in the bin")
    ax2.legend(loc="upper right")
    fig.subplots_adjust(wspace=0.35)
    fs.save(fig, out, "reliability_validation")


def ap50_by_class(results: Path, run: Path, experiment: str, seed: str, out: Path) -> None:
    table = pd.read_csv(results / "detector_summary" / "ap50_by_class.csv")
    table = table[(table.experiment == experiment) & (table.seed == seed) & (table.slice == "all")]
    n_boxes = {}
    for split in ("test", "test_2024"):
        with open(run / f"eval_{split}" / "summary.json", encoding="utf-8") as fh:
            n_boxes[split] = json.load(fh)["n_gt"]
    order = sorted(fs.CLASSES, key=lambda c: -n_boxes["test"].get(c, 0))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(fs.WIDTH, fs.WIDTH * 0.62))
    series = (("test", 0.18, "o", True, "test split"), ("test_2024", -0.18, "D", False, "2024 hold-out"))
    for split, offset, marker, fill, name in series:
        part = table[table.split == split].set_index("class_name")
        for i, cls in enumerate(order):
            n = n_boxes[split].get(cls, 0)
            if cls not in part.index or n == 0:
                continue
            row = part.loc[cls]
            colour = fs.COLOURS[cls]
            ax.errorbar(
                row.ap50,
                y[i] + offset,
                xerr=[[row.ap50 - row.ci_low], [row.ci_high - row.ap50]],
                fmt=marker,
                color=colour,
                markerfacecolor=colour if fill else "white",
                markeredgewidth=1.2,
                markersize=5,
                capsize=2,
                elinewidth=0.9,
            )
            ax.text(1.02, y[i] + offset, f"{n:,}", va="center", fontsize=6.5, color="0.3")
        ax.plot(
            [],
            [],
            marker=marker,
            color="0.2",
            markerfacecolor="0.2" if fill else "white",
            linestyle="none",
            label=name,
        )
    ax.set_yticks(y, [fs.label(c) for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP50 with 95 percent bootstrap interval over images")
    ax.text(1.02, -0.9, "boxes", fontsize=6.5, color="0.3", va="center")
    ax.grid(True, axis="x", color="0.92", linewidth=0.5)
    ax.legend(loc="upper left")
    fs.save(fig, out, "ap50_by_class")


def framework_comparison(results: Path, seed: str, out: Path) -> None:
    table = pd.read_csv(results / "detector_summary" / "map50_by_slice.csv")
    table = table[(table.seed == seed) & (table.metric == "map50")]
    groups = [
        ("val", "all", "validation"),
        ("test", "all", "test"),
        ("test", "2018", "test, 2018 stations"),
        ("test_2024", "all", "2024 hold-out"),
    ]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, fs.WIDTH * 0.45))
    x = np.arange(len(groups))
    offsets = np.linspace(-0.22, 0.22, len(fs.FRAMEWORKS))
    for (exp, (name, colour, marker)), dx in zip(fs.FRAMEWORKS.items(), offsets, strict=True):
        part = table[table.experiment == exp].set_index(["split", "slice"])
        vals = np.array([part.loc[(s, sl), "value"] for s, sl, _ in groups])
        lo = np.array([part.loc[(s, sl), "ci_low"] for s, sl, _ in groups])
        hi = np.array([part.loc[(s, sl), "ci_high"] for s, sl, _ in groups])
        ax.errorbar(
            x + dx,
            vals,
            yerr=[vals - lo, hi - vals],
            fmt=marker,
            color=colour,
            markersize=5,
            capsize=2.5,
            elinewidth=0.9,
            label=name,
        )
    first = table[table.experiment == next(iter(fs.FRAMEWORKS))].set_index(["split", "slice"])
    images = {(s, sl): int(first.loc[(s, sl), "n_images"]) for s, sl, _ in groups}
    ax.set_xticks(x, [f"{name}\n{images[(s, sl)]:,} images" for s, sl, name in groups])
    ax.set_ylabel("macro mAP50, classes with at least 10 boxes")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y", color="0.92", linewidth=0.5)
    ax.legend(loc="upper left", ncol=3)
    fs.save(fig, out, "framework_comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--experiment", default="rtdetrv2_l_1024")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", default=Path("docs/figures"), type=Path)
    args = ap.parse_args()
    fs.apply_style()
    run = args.results / "detector" / args.experiment / f"seed{args.seed}"
    confusion_matrix(run, args.out)
    pr_curves(run, args.out)
    f1_curves(run, args.out)
    reliability(run, args.out)
    ap50_by_class(args.results, run, args.experiment, f"seed{args.seed}", args.out)
    framework_comparison(args.results, f"seed{args.seed}", args.out)


if __name__ == "__main__":
    main()
