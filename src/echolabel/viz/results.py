"""Result figures from the evaluation tables.

Every figure states one comparison: a metric per configuration with its
bootstrap interval, the four fold values of a metric per configuration, or
the paired differences against the reference. Styles are shared so that all
figures look alike.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {
    "baseline": "0.45",
    "cnn": "tab:blue",
    "vit": "tab:orange",
    "hybrid": "tab:green",
    "other": "tab:purple",
}


def _family(experiment: str) -> str:
    for key in ("vit", "hybrid", "cnn"):
        if experiment.startswith(key):
            return key
    return "baseline" if experiment in ("mean",) or experiment.startswith("gbm") else "other"


def interval_plot(
    table: pd.DataFrame,
    metric: str,
    classes: list[str],
    experiments: list[str],
    path: Path,
    lower: bool = True,
    title: str = "",
) -> Path:
    """One panel per class: the metric per experiment with its interval; ``lower`` marks lower-is-better."""
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(3.2 * len(classes), 0.35 * len(experiments) + 1.5),
        sharey=True,
        squeeze=False,
    )
    for ax, class_name in zip(axes[0], classes, strict=True):
        sub = table[table["class_name"] == class_name].set_index("experiment").reindex(experiments)
        y = np.arange(len(experiments))
        for i, exp in enumerate(experiments):
            if exp not in sub.index or pd.isna(sub.loc[exp, metric]):
                continue
            lo, hi = (
                sub.loc[exp].get(f"{metric}_ci_low", np.nan),
                sub.loc[exp].get(f"{metric}_ci_high", np.nan),
            )
            colour = COLOURS[_family(exp)]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=colour, lw=2)
            ax.plot(sub.loc[exp, metric], i, "o", color=colour)
        ax.set_yticks(y)
        ax.set_yticklabels(experiments, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(class_name, fontsize=10)
        ax.set_xlabel(metric + (" (lower is better)" if lower else " (higher is better)"), fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fold_spread_plot(
    by_run: pd.DataFrame, metric: str, class_name: str, experiments: list[str], path: Path, title: str = ""
) -> Path:
    """The metric of every fold and seed per experiment, folds as markers, showing the fold spread."""
    sub = by_run[(by_run["class_name"] == class_name) & by_run["experiment"].isin(experiments)]
    folds = sorted(sub["fold"].unique())
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(experiments)), 3.8))
    for j, fold in enumerate(folds):
        f = sub[sub["fold"] == fold]
        xs = [experiments.index(e) + (j - len(folds) / 2) * 0.12 for e in f["experiment"]]
        ax.scatter(
            xs, f[metric], marker=markers[j % len(markers)], s=22, label=fold.replace("test_", ""), alpha=0.8
        )
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric}, {class_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="test transect", fontsize=7, title_fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def difference_plot(
    differences: pd.DataFrame, metric: str, classes: list[str], path: Path, title: str = ""
) -> Path:
    """Paired differences against the reference per experiment and class with their intervals; zero marked."""
    sub = differences[differences["metric"] == metric]
    experiments = sorted(sub["experiment"].unique())
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(3.2 * len(classes), 0.35 * len(experiments) + 1.5),
        sharey=True,
        squeeze=False,
    )
    for ax, class_name in zip(axes[0], classes, strict=True):
        s = sub[sub["class_name"] == class_name].set_index("experiment").reindex(experiments)
        for i, exp in enumerate(experiments):
            if exp not in s.index or pd.isna(s.loc[exp, "difference"]):
                continue
            colour = COLOURS[_family(exp)]
            ax.plot([s.loc[exp, "ci_low"], s.loc[exp, "ci_high"]], [i, i], color=colour, lw=2)
            ax.plot(s.loc[exp, "difference"], i, "o", color=colour)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(experiments, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(class_name, fontsize=10)
        ax.set_xlabel(f"{metric} minus reference", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def error_factor_plot(
    table: pd.DataFrame, classes: list[str], path: Path, metric: str = "log_error", title: str = ""
) -> Path:
    """One panel per factor: the error per bin with its interval, one line per class.

    ``table`` holds one row per class, factor and bin with ``order`` (bin
    position), the metric and its interval. Bins that differ between classes
    (equal-count bins fitted per class) are labelled by their position.
    """
    factors = list(dict.fromkeys(table["factor"]))
    fig, axes = plt.subplots(1, len(factors), figsize=(3.3 * len(factors), 3.8), squeeze=False)
    for ax, factor in zip(axes[0], factors, strict=True):
        sub = table[table["factor"] == factor]
        positions = sorted(sub["order"].unique())
        ticks = []
        for position in positions:
            names = sub.loc[sub["order"] == position, "bin"].unique()
            ticks.append(str(names[0]) if len(names) == 1 else f"bin {position}")
        for j, class_name in enumerate(classes):
            s = sub[sub["class_name"] == class_name].sort_values("order")
            if s.empty:
                continue
            x = s["order"].to_numpy(dtype=float) + (j - (len(classes) - 1) / 2) * 0.12
            y = s[metric].to_numpy(dtype=float)
            lo = y - s[f"{metric}_ci_low"].to_numpy(dtype=float)
            hi = s[f"{metric}_ci_high"].to_numpy(dtype=float) - y
            ax.errorbar(
                x, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], marker="o", ms=3, lw=1, label=class_name
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
        ax.set_title(factor, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel(metric)
    axes[0][0].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def band_mass_plot(table: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Share of attribution mass on the camera band per network and method, against an even map.

    ``table`` holds one row per experiment, method and class with
    ``band_mass`` and its interval and ``uniform_share``; the row of the
    class ``all`` is drawn with its interval, the single classes as small
    marks, and the share of an even map as a dashed line.
    """
    keys = list(dict.fromkeys(zip(table["experiment"], table["method"], strict=True)))
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(keys) + 1.6))
    for i, (experiment, method) in enumerate(keys):
        sub = table[(table["experiment"] == experiment) & (table["method"] == method)]
        colour = COLOURS[_family(experiment)]
        singles = sub[sub["class_name"] != "all"]
        ax.plot(singles["band_mass"], np.full(len(singles), i), "|", color=colour, ms=7, alpha=0.7)
        pooled = sub[sub["class_name"] == "all"]
        if len(pooled):
            row = pooled.iloc[0]
            ax.plot([row["band_mass_ci_low"], row["band_mass_ci_high"]], [i, i], color=colour, lw=2)
            ax.plot(row["band_mass"], i, "o", color=colour)
    if len(table):
        ax.axvline(float(table["uniform_share"].mean()), color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{e}, {m}" for e, m in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of attribution mass on the camera band (dashed: an even map)", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def echogram_panels(
    sv_db: np.ndarray,
    depth_cells: np.ndarray,
    camera_depth_m: float,
    labels: list[str],
    path: Path,
    title: str = "",
    limits: tuple[float, float] = (-90.0, -50.0),
) -> Path:
    """The echogram of one window at several channels side by side, on one colour scale.

    ``sv_db`` has shape (F, T, D) in dB; cells without a usable sample (NaN)
    are drawn in grey, and the camera depth is marked by a line.
    """
    cmap = plt.get_cmap("viridis").with_extremes(bad="0.75")
    fig, axes = plt.subplots(
        1, len(labels), figsize=(3.6 * len(labels) + 0.8, 3.4), sharey=True, squeeze=False
    )
    extent = (0, sv_db.shape[1], float(depth_cells[-1]), float(depth_cells[0]))
    image = None
    for ax, label, panel in zip(axes[0], labels, sv_db, strict=True):
        image = ax.imshow(
            np.ma.masked_invalid(panel.T),
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        if np.isfinite(camera_depth_m):
            ax.axhline(camera_depth_m, color="white", lw=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("time cell", fontsize=8)
    axes[0][0].set_ylabel("depth (m)", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=list(axes[0]), fraction=0.025, pad=0.02, label="dB")
    if title:
        fig.suptitle(title, fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
