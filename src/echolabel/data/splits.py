"""Tow-level splits of the annotation collection.

A tow (station or transect) is the unit of assignment: all frames of a tow
land in the same split, so that consecutive, near-identical frames never sit
on both sides of a split boundary. One year is held out entirely as an
out-of-distribution test set, some tows can be forced into a split, and the
remaining tows are assigned by a seeded search that keeps the image fractions
near their targets while spreading every class over the splits.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def tow_key(year: str, station: str) -> str:
    return f"{year}/{station}"


def tow_table(images: pd.DataFrame, boxes: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """One row per tow with image counts, negatives and boxes per class.

    ``images`` holds one row per paired annotation file with the columns
    ``key``, ``year``, ``station`` and ``n_boxes``; ``boxes`` holds one row per
    box with ``key`` and ``class_name``.
    """
    img = images.copy()
    img["tow"] = [tow_key(str(y), str(s)) for y, s in zip(img["year"], img["station"], strict=True)]
    per_tow = img.groupby("tow").agg(
        year=("year", "first"),
        station=("station", "first"),
        n_images=("key", "size"),
        n_with_boxes=("n_boxes", lambda s: int((s > 0).sum())),
        n_zero_box=("n_boxes", lambda s: int((s == 0).sum())),
    )
    key_to_tow = dict(zip(img["key"], img["tow"], strict=True))
    b = boxes.copy()
    b["tow"] = b["key"].map(key_to_tow)
    counts = b.groupby(["tow", "class_name"]).size().unstack(fill_value=0)
    for c in classes:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[classes]
    table = per_tow.join(counts, how="left").fillna(0)
    table[classes] = table[classes].astype(int)
    return table.reset_index()


def split_loss(
    assign: np.ndarray,
    n_images: np.ndarray,
    class_counts: np.ndarray,
    years: np.ndarray,
    fractions: np.ndarray,
    min_tows: int,
    tolerance: float = 0.02,
    year_min_tows: dict[str, dict[int, int]] | None = None,
) -> float:
    """Deviation of a tow assignment from the target fractions.

    Six terms are summed. Image fractions: the excess of every split's image
    fraction over its target beyond ``tolerance``, weighted by ten. Class
    fractions: for every class, the absolute deviation of its box fraction in
    every split from the target, summed over classes; classes carried by fewer
    than ``min_tows`` tows cannot be spread and count with a quarter of the
    weight. Presence in training: for every class carried by at least two
    tows, twice the shortfall of its training box count below half of the
    class. Presence in validation and test: for every class carried by at
    least ``min_tows`` tows, twice the shortfall of its box count below the
    larger of ten boxes and five percent of the class, as a fraction of that
    requirement; a class carried by exactly two tows is required in one of
    the two. Year fractions enter like image fractions with a tolerance of
    three percent, for years with at least 200 images, with weight one.
    Minimum tows per year: ``year_min_tows`` maps a year to the minimum number
    of its tows required in a split index; every missing tow costs five.
    """
    total_images = n_images.sum()
    loss = 0.0
    for s in range(3):
        dev = abs(n_images[assign == s].sum() / total_images - fractions[s])
        loss += 10.0 * max(0.0, dev - tolerance)
    class_total = class_counts.sum(axis=0)
    class_tows = (class_counts > 0).sum(axis=0)
    present = class_total > 0
    weights = np.where(class_tows >= min_tows, 1.0, 0.25) * present
    split_counts = np.stack([class_counts[assign == s].sum(axis=0) for s in range(3)])
    frac = np.divide(split_counts, class_total, out=np.zeros_like(split_counts), where=present)
    loss += float(np.sum(weights * np.abs(frac - fractions[:, None])))
    need_train = np.maximum(10.0, 0.5 * class_total)
    short_train = np.clip(need_train - split_counts[0], 0, None) / need_train
    loss += 2.0 * float(short_train[(class_tows >= 2) & present].sum())
    need = np.maximum(10.0, 0.05 * class_total)
    shorts = np.stack([np.clip(need - split_counts[s], 0, None) / need for s in (1, 2)])
    spread = (class_tows >= min_tows) & present
    loss += 2.0 * float(shorts[:, spread].sum())
    two_tows = (class_tows == 2) & present & (min_tows > 2)
    loss += 2.0 * float(shorts[:, two_tows].min(axis=0).sum())
    for year in np.unique(years):
        mask_year = years == year
        year_images = n_images[mask_year].sum()
        if year_images < 200:
            continue
        for s in range(3):
            dev = abs(n_images[mask_year & (assign == s)].sum() / year_images - fractions[s])
            loss += 1.0 * max(0.0, dev - 0.03)
    for year, minima in (year_min_tows or {}).items():
        mask_year = years == year
        for s, minimum in minima.items():
            count = int(np.sum(mask_year & (assign == s)))
            loss += 5.0 * max(0, int(minimum) - count)
    return loss


def assign_splits(
    tows: pd.DataFrame,
    classes: list[str],
    fractions: dict[str, float],
    forced: dict[str, str] | None = None,
    holdout: dict[str, str] | None = None,
    seed: int = 0,
    n_restarts: int = 40,
    n_moves: int = 6000,
    min_tows_for_presence_penalty: int = 3,
    min_tows_per_split: dict[str, dict[str, int]] | None = None,
) -> pd.Series:
    """Assign every tow to a split name.

    ``holdout`` maps a year to a split name that receives all tows of that
    year. ``forced`` maps a tow key to one of train, val or test.
    ``min_tows_per_split`` maps a year to the minimum number of its tows that
    validation or test must contain, for example ``{"2018": {"val": 5}}``;
    forced tows do not count towards these minima. The other
    tows are assigned by a seeded search with ``n_restarts`` starts. Each start
    shuffles the free tows and fills the splits greedily to their image
    quotas, forced tows included, then runs ``n_moves`` proposals that either
    move one tow to another split or swap two tows between splits. A proposal
    that lowers the loss of :func:`split_loss` is always kept; a worse one is
    kept with a probability that decays geometrically over the run (simulated
    annealing). The best assignment seen over all starts is returned as a
    Series indexed by tow key; the result is deterministic for a given seed.
    """
    forced = forced or {}
    holdout = {str(k): v for k, v in (holdout or {}).items()}
    for key, split_name in forced.items():
        if split_name not in SPLITS:
            raise ValueError(f"forced split {split_name!r} for {key} is not one of {SPLITS}")
    result: dict[str, str] = {}
    year_str = tows["year"].astype(str)
    held = year_str.isin(holdout.keys())
    for key, year in zip(tows.loc[held, "tow"], year_str[held], strict=True):
        result[key] = holdout[year]
    rows = tows[~held].reset_index(drop=True)
    n = len(rows)
    if n == 0:
        return pd.Series(result, name="split")

    fr = np.array([fractions[s] for s in SPLITS], dtype=float)
    fr = fr / fr.sum()
    n_images = rows["n_images"].to_numpy(dtype=float)
    class_counts = rows[classes].to_numpy(dtype=float)
    years = rows["year"].astype(str).to_numpy()
    fixed = np.full(n, -1, dtype=int)
    for i, key in enumerate(rows["tow"]):
        if key in forced:
            fixed[i] = SPLITS.index(forced[key])
    free_idx = np.flatnonzero(fixed < 0)
    quotas = fr * n_images.sum()
    filled_fixed = np.array([n_images[fixed == s].sum() for s in range(3)])

    year_min_tows = {
        str(year): {SPLITS.index(split_name): int(minimum) for split_name, minimum in minima.items()}
        for year, minima in (min_tows_per_split or {}).items()
    }

    def loss_of(assign: np.ndarray) -> float:
        return split_loss(
            assign,
            n_images,
            class_counts,
            years,
            fr,
            min_tows_for_presence_penalty,
            year_min_tows=year_min_tows,
        )

    rng = np.random.default_rng(seed)
    best_loss = np.inf
    best_assign = fixed.copy()
    t_start, t_end = 0.05, 0.0005
    for _ in range(max(1, n_restarts)):
        assign = fixed.copy()
        filled = filled_fixed.copy()
        for i in rng.permutation(free_idx):
            s = int(np.argmax(quotas - filled))
            assign[i] = s
            filled[s] += n_images[i]
        current = loss_of(assign)
        if current < best_loss:
            best_loss, best_assign = current, assign.copy()
        if free_idx.size < 2:
            continue
        for step in range(max(0, n_moves)):
            candidate = assign.copy()
            if rng.random() < 0.5:
                i = int(rng.choice(free_idx))
                s_new = int(rng.integers(0, 2))
                candidate[i] = s_new + 1 if s_new >= assign[i] else s_new
            else:
                i, j = (int(v) for v in rng.choice(free_idx, size=2, replace=False))
                if assign[i] == assign[j]:
                    continue
                candidate[i], candidate[j] = assign[j], assign[i]
            cand_loss = loss_of(candidate)
            temperature = t_start * (t_end / t_start) ** (step / max(1, n_moves - 1))
            if cand_loss < current or rng.random() < np.exp(-(cand_loss - current) / temperature):
                assign, current = candidate, cand_loss
                if current < best_loss:
                    best_loss, best_assign = current, assign.copy()
    for key, s in zip(rows["tow"], best_assign, strict=True):
        result[key] = SPLITS[int(s)]
    log.info("split search: %d starts, %d moves each, best loss %.4f", n_restarts, n_moves, best_loss)
    return pd.Series(result, name="split")


def check_split_properties(
    images: pd.DataFrame,
    holdout: dict[str, str],
    forced: dict[str, str],
    protected_tows: list[str],
) -> list[str]:
    """Return the list of violated rules (empty when all hold).

    ``images`` has one row per image with ``key``, ``year``, ``station`` and
    ``split``. Rules: every image has exactly one split; a tow sits in one
    split only; held-out years are confined to their split and nothing else
    enters it; forced tows sit where required; protected tows (those recorded
    during the label-generation transects) never appear in validation or test.
    """
    problems: list[str] = []
    img = images.copy()
    img["tow"] = [tow_key(str(y), str(s)) for y, s in zip(img["year"], img["station"], strict=True)]
    if img["split"].isna().any() or (img["split"].astype(str) == "").any():
        problems.append("images without split")
    if img["key"].duplicated().any():
        problems.append("duplicate image keys")
    per_tow = img.groupby("tow")["split"].nunique()
    if (per_tow > 1).any():
        problems.append(f"tows in more than one split: {per_tow[per_tow > 1].index.tolist()[:5]}")
    for year, split_name in holdout.items():
        in_year = img["year"].astype(str) == str(year)
        if (img.loc[in_year, "split"] != split_name).any():
            problems.append(f"year {year} not confined to {split_name}")
        if (img.loc[~in_year, "split"] == split_name).any():
            problems.append(f"split {split_name} contains images outside year {year}")
    for key, split_name in forced.items():
        sel = img["tow"] == key
        if sel.any() and (img.loc[sel, "split"] != split_name).any():
            problems.append(f"tow {key} not in forced split {split_name}")
    bad = img[img["tow"].isin(protected_tows) & img["split"].isin(["val", "test"])]
    if not bad.empty:
        problems.append(f"protected tows in val or test: {sorted(bad['tow'].unique())}")
    return problems


def class_counts_by_split(images: pd.DataFrame, boxes: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Boxes and images per class and split, plus image totals and negatives."""
    key_to_split = dict(zip(images["key"], images["split"], strict=True))
    b = boxes.copy()
    b["split"] = b["key"].map(key_to_split)
    box_counts = b.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    img_counts = (
        b.drop_duplicates(["key", "class_name"]).groupby(["class_name", "split"]).size().unstack(fill_value=0)
    )
    split_names = list(dict.fromkeys(images["split"]))
    rows: list[dict[str, Any]] = []
    for c in classes:
        row: dict[str, Any] = {"class_name": c}
        for s in split_names:
            row[f"boxes_{s}"] = (
                int(box_counts.at[c, s]) if c in box_counts.index and s in box_counts.columns else 0
            )
            row[f"images_{s}"] = (
                int(img_counts.at[c, s]) if c in img_counts.index and s in img_counts.columns else 0
            )
        rows.append(row)
    total: dict[str, Any] = {"class_name": "all_images"}
    neg: dict[str, Any] = {"class_name": "negatives"}
    for s in split_names:
        sel = images["split"] == s
        total[f"images_{s}"] = int(sel.sum())
        total[f"boxes_{s}"] = int((b["split"] == s).sum())
        neg[f"images_{s}"] = int((sel & (images["n_boxes"] == 0)).sum())
        neg[f"boxes_{s}"] = 0
    rows += [total, neg]
    return pd.DataFrame(rows)
