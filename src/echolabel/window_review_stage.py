"""Stage 9 command: what the camera and the echosounder show in the worst predicted windows.

For one configuration and class the windows with the largest absolute
error are taken from the error analysis. For each of them the frames of
the window's camera minute are drawn with the counted boxes and their
calibrated scores (a set spread evenly over the minute and a run of
consecutive frames), the counts per frame are summarised, and the echogram
of the window is drawn at several channels. The frame sheets hold camera
images and stay with the run; the tables and the echogram figures are
small.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .manifest import write_manifest
from .models.data import WindowStore
from .models_stage import _window_paths, experiment_config
from .viz.results import echogram_panels

log = logging.getLogger(__name__)


def overlap_share(boxes_by_frame: list[np.ndarray], iou_threshold: float = 0.3) -> float:
    """Share of boxes that overlap a box of the preceding frame at or above ``iou_threshold``.

    ``boxes_by_frame`` holds one array (M, 4) of x0, y0, x1, y1 per frame in
    time order. An object that moves through the image leaves no overlap
    between frames a fifth of a second apart; a high share marks boxes that
    stay in place.
    """
    hits, total = 0, 0
    for prev, cur in zip(boxes_by_frame[:-1], boxes_by_frame[1:], strict=True):
        total += len(cur)
        if len(cur) == 0 or len(prev) == 0:
            continue
        x0 = np.maximum(cur[:, None, 0], prev[None, :, 0])
        y0 = np.maximum(cur[:, None, 1], prev[None, :, 1])
        x1 = np.minimum(cur[:, None, 2], prev[None, :, 2])
        y1 = np.minimum(cur[:, None, 3], prev[None, :, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area_c = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
        area_p = (prev[:, 2] - prev[:, 0]) * (prev[:, 3] - prev[:, 1])
        iou = inter / np.maximum(area_c[:, None] + area_p[None, :] - inter, 1e-9)
        hits += int((iou.max(axis=1) >= iou_threshold).sum())
    return hits / total if total else float("nan")


def frame_summary(counts: np.ndarray, top_share: float = 0.1) -> dict[str, float]:
    """Summary of the counts per frame of one window.

    Returns the number of frames, the sum, mean, median and maximum per
    frame, the occupancy (share of frames with a positive count), the share
    of the sum carried by the ``top_share`` of frames with the highest count,
    and the correlation between the counts of consecutive frames.
    """
    c = np.asarray(counts, dtype=float)
    if c.size == 0:
        return {"n_frames": 0}
    ordered = np.sort(c)[::-1]
    k = max(1, int(np.ceil(top_share * c.size)))
    lag1 = float(np.corrcoef(c[:-1], c[1:])[0, 1]) if c.size > 2 and np.ptp(c) > 0 else float("nan")
    return {
        "n_frames": int(c.size),
        "sum": float(c.sum()),
        "mean_per_frame": float(c.mean()),
        "median_per_frame": float(np.median(c)),
        "max_per_frame": float(c.max()),
        "occupancy": float((c > 0).mean()),
        "share_in_top_frames": float(ordered[:k].sum() / c.sum()) if c.sum() > 0 else float("nan"),
        "lag1_correlation": lag1,
    }


def choose_windows(errors: pd.DataFrame, class_name: str, n_worst: int, n_per_fold: int) -> pd.DataFrame:
    """The ``n_worst`` windows by absolute error of the class, and the ``n_per_fold`` worst of every fold.

    Returns the rows of ``errors`` in order of decreasing error with a
    ``selection`` column: ``overall`` for the first set, ``fold`` for windows
    that enter through their fold only.
    """
    col = f"abs_error_{class_name}"
    ranked = errors.sort_values(col, ascending=False)
    overall = ranked.head(n_worst).assign(selection="overall")
    per_fold = ranked.groupby("fold", sort=False).head(n_per_fold)
    extra = per_fold[~per_fold["window_id"].isin(overall["window_id"])].assign(selection="fold")
    return pd.concat([overall, extra], ignore_index=True)


def _sheet(
    frames: list[tuple[str, Path]], boxes: pd.DataFrame, class_name: str, title: str, path: Path
) -> Path:
    """Two columns of frames with the boxes of ``boxes`` drawn, saved as one JPEG."""
    from PIL import Image, ImageDraw

    from .detection.review import draw_boxes

    thumbs = []
    for frame, source in frames:
        if not source.is_file():
            log.warning("frame %s not found", source)
            continue
        with Image.open(source) as img:
            drawn = draw_boxes(img, boxes[boxes["frame"] == frame], class_name)
        thumbs.append((frame, drawn))
    if not thumbs:
        return path
    width, height = thumbs[0][1].size
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * width, 22 + rows * (height + 18)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for k, (frame, thumb) in enumerate(thumbs):
        x, y = (k % cols) * width, 22 + (k // cols) * (height + 18)
        sheet.paste(thumb, (x, y + 18))
        n = int((boxes["frame"] == frame).sum())
        draw.text((x + 4, y + 3), f"{frame}  boxes drawn: {n}", fill=(0, 0, 0))
    sheet.save(path, quality=88)
    return path


def run_review_windows(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Frame sheets, per-frame summaries and multi-channel echograms of the worst predicted windows."""
    rcfg = cfg.get("window_review", {})
    experiment = str(rcfg.get("experiment", cfg.get("evaluation", {}).get("reference", "cnn")))
    class_name = str(rcfg.get("class_name", "krill"))
    n_worst, n_per_fold = int(rcfg.get("n_worst", 10)), int(rcfg.get("n_worst_per_fold", 3))
    n_spread, n_run = int(rcfg.get("frames_spread", 8)), int(rcfg.get("frames_consecutive", 8))
    iou = float(rcfg.get("match_iou", 0.3))
    channels = [float(f) for f in rcfg.get("channels_hz", [38000, 120000, 200000])]
    formats = list(rcfg.get("formats", ["png"]))
    camera = str(rcfg.get("camera", "Left"))
    project = resolve_path(cfg, "project_dir")
    errors_path = Path(project, "runs", "acoustic_errors", f"windows_{experiment}.parquet")
    exp = experiment_config(cfg, experiment)
    zarr_path, meta_dir, _ = _window_paths(
        cfg, str(exp.get("geometry", "base")), str(exp.get("windows_variant", ""))
    )
    store = WindowStore(zarr_path, meta_dir)
    run = cfg.get("windows_run", {})
    tag = f"{run.get('experiment')}_seed{int(run.get('seed', 0))}"
    span_s = float(store.meta["target_span_s"].iloc[0]) if "target_span_s" in store.meta else 60.0
    target = out_dir / experiment / class_name
    target.mkdir(parents=True, exist_ok=True)
    chosen = choose_windows(pd.read_parquet(errors_path), class_name, n_worst, n_per_fold)
    position = pd.Series(np.arange(len(store.meta)), index=store.meta["window_id"])
    channel_positions = store.channel_positions(channels)
    rows, outputs, sheets = [], [], []
    detections: dict[str, pd.DataFrame] = {}
    frame_tables: dict[str, pd.DataFrame] = {}
    for rank, w in enumerate(chosen.to_dict("records")):
        transect = str(w["transect"])
        if transect not in detections:
            infer = Path(project, "runs", "inference", tag, transect, camera)
            dets = pd.read_parquet(infer / "detections.parquet")
            keep = dets["kept_resolved"].to_numpy(dtype=bool) & ~dets["static"].to_numpy(dtype=bool)
            dets = dets[keep].copy()
            dets["frame"] = dets["folder"].astype(str) + "/" + dets["filename"].astype(str)
            detections[transect] = dets
            frames = pd.read_parquet(infer / "frames.parquet").sort_values("timestamp").reset_index(drop=True)
            frames["frame"] = frames["folder"].astype(str) + "/" + frames["filename"].astype(str)
            frame_tables[transect] = frames
        dets, frames = detections[transect], frame_tables[transect]
        centre = pd.Timestamp(w["time_camera"])
        half = pd.Timedelta(seconds=span_s / 2.0)
        inside = frames[(frames["timestamp"] >= centre - half) & (frames["timestamp"] < centre + half)]
        counts = inside[f"{class_name}_res"].to_numpy(dtype=float)
        boxes = dets[dets["frame"].isin(inside["frame"])]
        of_class = boxes[boxes["class_name"] == class_name]
        by_frame = [
            of_class.loc[of_class["frame"] == f, ["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
            for f in inside["frame"]
        ]
        area = (of_class["x1"] - of_class["x0"]) * (of_class["y1"] - of_class["y0"])
        row = {
            "rank": rank,
            "selection": w["selection"],
            "window_id": int(w["window_id"]),
            "fold": w["fold"],
            "transect": transect,
            "time_camera": str(centre),
            "camera_depth_m": float(w.get("camera_depth_m", np.nan)),
            "count": float(w[f"y_{class_name}"]),
            "prediction": float(w[f"mean_{class_name}"]),
            **frame_summary(counts),
            "overlap_share": overlap_share(by_frame, iou),
            "score_median": float(of_class["score_calibrated"].median()) if len(of_class) else float("nan"),
            "score_share_below_0p6": float((of_class["score_calibrated"] < 0.6).mean())
            if len(of_class)
            else np.nan,
            "box_side_median_px": float(np.sqrt(area).median()) if len(of_class) else float("nan"),
            "other_classes": int((boxes["class_name"] != class_name).sum()),
        }
        rows.append(row)
        label = (
            f"{experiment} {class_name}: window {row['window_id']} of {transect}, {centre:%H:%M:%S} UTC, "
            f"camera at {row['camera_depth_m']:.0f} m, counted {row['count']:.0f}, "
            f"predicted {row['prediction']:.0f}"
        )
        if len(inside):
            transect_dir = resolve_path(cfg, "source_data", transect)
            names = inside["frame"].tolist()
            spread = [names[i] for i in np.linspace(0, len(names) - 1, min(n_spread, len(names))).astype(int)]
            start = max(0, len(names) // 2 - n_run // 2)
            consecutive = names[start : start + n_run]
            for kind, selection in (("spread", spread), ("consecutive", consecutive)):
                path = target / f"frames_{rank:02d}_{transect}_{row['window_id']}_{kind}.jpg"
                _sheet(
                    [(f, Path(transect_dir, camera, f)) for f in selection],
                    boxes,
                    class_name,
                    f"{label}; {kind} frames",
                    path,
                )
                sheets.append(path)
        idx = int(position.loc[row["window_id"]])
        sv = store.group["sv_db"].oindex[idx, channel_positions, :, :].astype(np.float32)
        depth_cells = store.group["depth_cells"][idx, :].astype(np.float32)
        for fmt in formats:
            outputs.append(
                echogram_panels(
                    sv,
                    depth_cells,
                    row["camera_depth_m"],
                    [f"{f / 1000:g} kHz" for f in channels],
                    target / f"echogram_{rank:02d}_{transect}_{row['window_id']}.{fmt}",
                    title=label,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(target / "worst_windows.csv", index=False)
    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment": experiment,
                "class_name": class_name,
                "n_windows": int(len(table)),
                "frame_sheets": [p.name for p in sheets],
                "span_s": span_s,
            },
            fh,
            indent=2,
        )
    write_manifest(
        target,
        "review-windows",
        cfg,
        {"errors": errors_path},
        [target / "worst_windows.csv", target / "summary.json", *outputs],
    )
    log.info("reviewed %d windows of %s, %d frame sheets", len(table), class_name, len(sheets))
    return target
