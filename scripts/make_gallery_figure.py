"""Gallery of transect frames with the detections of the selected detector.

Two steps. ``select`` runs where the frames and the inference tables are (the compute
server): it reads ``detections.parquet`` of every transect under the inference run, keeps
resolved objects at the operating thresholds that are not fixed structure, scores every
frame per class by the confidence and the clutter of its objects, copies the best candidate
frames to a folder and writes their detections next to them. ``render`` draws a chosen
subset of those frames with their boxes and class labels.

Usage:
  python scripts/make_gallery_figure.py select --inference-dir <project_dir>/runs/inference/<run>
         --transects-root <source_data> --out <candidate directory> [--per-class 6]
         [--depth-dir <project_dir>/runs/camera_series/<run>]
  python scripts/make_gallery_figure.py render --candidates <candidate directory>
         --selection docs/figures/gallery_frames.csv [--out docs/figures] [--columns 4]

The selection file lists the chosen frames with the columns ``transect``, ``filename`` and
``caption`` in the order of the tiles.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
from figure_style import plt  # noqa: E402

GALLERY_CLASSES = [
    "krill",
    "herring",
    "benthosema",
    "maurolicus",
    "bluewhiting",
    "mackerel",
    "gonatus",
    "gelatinous",
]
DETECTION_COLUMNS = [
    "transect",
    "frame_index",
    "filename",
    "class_name",
    "score_calibrated",
    "x0",
    "y0",
    "x1",
    "y1",
]


def objects_of(detections: pd.DataFrame) -> pd.DataFrame:
    """Resolved objects that are not fixed structure."""
    d = detections
    if "kept_resolved" in d.columns:
        d = d[d["kept_resolved"]]
    if "static" in d.columns:
        d = d[~d["static"]]
    return d


def select(args: argparse.Namespace) -> None:
    out = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    kept_parts = []
    for det_path in sorted(args.inference_dir.glob("*/Left/detections.parquet")):
        transect = det_path.parents[1].name
        dets = objects_of(pd.read_parquet(det_path))
        if dets.empty:
            continue
        per_frame = dets.groupby("frame_index")
        n_objects = per_frame.size().rename("n_objects")
        min_score = per_frame["score_calibrated"].min().rename("min_score")
        info = dets.drop_duplicates("frame_index").set_index("frame_index")
        info = info[["folder", "filename", "timestamp"]]
        times = pd.to_datetime(info["timestamp"])
        for cls in args.classes:
            d = dets[dets["class_name"] == cls]
            if d.empty:
                continue
            g = d.groupby("frame_index")
            parts = [g.size().rename("n_class"), g["score_calibrated"].mean().rename("mean_class_score")]
            table = pd.concat([*parts, n_objects, min_score], axis=1, join="inner")
            # Frames with few, confident objects of the class and nothing doubtful in the frame;
            # krill is dense by nature, so several animals per frame are wanted there.
            lo, hi, most = (4, 40, 60) if cls == "krill" else (1, 4, 6)
            table = table[(table.n_class >= lo) & (table.n_class <= hi) & (table.n_objects <= most)]
            table = table[table.min_score >= args.min_score]
            if table.empty:
                continue
            table["score"] = table.mean_class_score * table.min_score
            table = table.sort_values("score", ascending=False)
            chosen: list[int] = []
            for idx in table.index:
                # Candidates of one class lie at least two minutes apart.
                if all(abs(times[idx] - times[j]) > pd.Timedelta(minutes=2) for j in chosen):
                    chosen.append(idx)
                if len(chosen) >= args.per_class:
                    break
            for idx in chosen:
                r = table.loc[idx]
                rows.append(
                    {
                        "transect": transect,
                        "class_name": cls,
                        "frame_index": int(idx),
                        "folder": info.loc[idx, "folder"],
                        "filename": info.loc[idx, "filename"],
                        "timestamp": info.loc[idx, "timestamp"],
                        "n_class": int(r.n_class),
                        "n_objects": int(r.n_objects),
                        "mean_class_score": float(r.mean_class_score),
                        "min_score": float(r.min_score),
                    }
                )
        frame_ids = {r["frame_index"] for r in rows if r["transect"] == transect}
        part = dets[dets["frame_index"].isin(frame_ids)].copy()
        if "transect" not in part.columns:
            part.insert(0, "transect", transect)
        kept_parts.append(part[DETECTION_COLUMNS])
    candidates = pd.DataFrame(rows)
    if args.depth_dir is not None and not candidates.empty:
        depths = []
        for p in sorted(args.depth_dir.glob("*/Left/depth_per_frame.parquet")):
            f = pd.read_parquet(p, columns=["timestamp", "depth_clean"])
            f["transect"] = p.parents[1].name
            depths.append(f)
        if depths:
            depth = pd.concat(depths)
            depth["timestamp"] = pd.to_datetime(depth["timestamp"])
            candidates["timestamp"] = pd.to_datetime(candidates["timestamp"])
            candidates = pd.merge_asof(
                candidates.sort_values("timestamp"),
                depth.sort_values("timestamp"),
                on="timestamp",
                by="transect",
                tolerance=pd.Timedelta("100ms"),
                direction="nearest",
            ).sort_values(["class_name", "transect", "timestamp"])
    for r in candidates.itertuples(index=False):
        src = Path(args.transects_root, r.transect, "Left", r.folder, r.filename)
        dst = out / "frames" / r.transect / r.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            shutil.copyfile(src, dst)
    candidates.to_csv(out / "candidates.csv", index=False)
    pd.concat(kept_parts, ignore_index=True).to_csv(out / "detections.csv", index=False)
    print(f"{len(candidates)} candidate frames written to {out}")


def render(args: argparse.Namespace) -> None:
    selection = pd.read_csv(args.selection)
    dets = pd.read_csv(args.candidates / "detections.csv")
    n = len(selection)
    ncols = args.columns
    nrows = int(np.ceil(n / ncols))
    sample = Image.open(args.candidates / "frames" / selection.iloc[0].transect / selection.iloc[0].filename)
    aspect = sample.height / sample.width
    tile_w = fs.WIDTH / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fs.WIDTH, tile_w * aspect * nrows * 1.12))
    for ax, row in zip(np.atleast_1d(axes).flat, selection.itertuples(index=False), strict=False):
        path = args.candidates / "frames" / row.transect / row.filename
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        d = dets[(dets.transect == row.transect) & (dets.filename == row.filename)]
        for b in d.itertuples(index=False):
            colour = fs.COLOURS.get(b.class_name, "#FFFFFF")
            ax.add_patch(
                plt.Rectangle(
                    (b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fill=False, edgecolor=colour, linewidth=1.1
                )
            )
            # Labels sit above the box, anchored to the edge that keeps them inside the frame.
            near_right = b.x0 > 0.72 * sample.width
            ax.text(
                b.x1 - 3 if near_right else b.x0 + 3,
                max(b.y0 - 4, 28),
                f"{fs.label(b.class_name)} {b.score_calibrated:.2f}",
                fontsize=4.6,
                color="black",
                va="bottom",
                ha="right" if near_right else "left",
                bbox={"facecolor": colour, "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
                clip_on=True,
            )
        ax.set_title(str(row.caption), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in np.atleast_1d(axes).flat[n:]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03, hspace=0.18, left=0.01, right=0.99, top=0.96, bottom=0.01)
    fs.save(fig, args.out, "detection_gallery", dpi=220)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    s = sub.add_parser("select")
    s.add_argument("--inference-dir", type=Path, required=True)
    s.add_argument("--transects-root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--depth-dir", type=Path, default=None)
    s.add_argument("--classes", type=lambda v: v.split(","), default=GALLERY_CLASSES)
    s.add_argument("--per-class", type=int, default=6)
    s.add_argument("--min-score", type=float, default=0.6)
    s.set_defaults(func=select)
    r = sub.add_parser("render")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--selection", type=Path, required=True)
    r.add_argument("--out", type=Path, default=Path("docs/figures"))
    r.add_argument("--columns", type=int, default=4)
    r.set_defaults(func=render)
    args = ap.parse_args()
    fs.apply_style()
    args.func(args)


if __name__ == "__main__":
    main()
