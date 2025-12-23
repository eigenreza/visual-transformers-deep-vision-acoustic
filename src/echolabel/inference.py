"""Stage 3: detector inference on whole camera transects.

A trained experiment is run on every frame of one camera of one transect.
Detections above a low raw score are stored with their calibrated confidence,
box and frame timestamp; a second table gives, per frame, the number of
detections per class above a grid of raw thresholds and above the per-class
operating thresholds. Folders are processed one at a time and skipped when
their part file exists, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.transects import list_camera_frames
from .detection.calibration import Calibrator
from .detection.evaluate import load_coco
from .detection.objects import effective_thresholds
from .detection.resolve import assign_queries, expected_counts, hypothesis_sums, resolve_cross_class
from .detection.static import TEMPLATE_COLUMNS, flag_static, static_summary, static_templates
from .detector_stage import _cuda_available, _experiment, _paths
from .manifest import write_manifest

log = logging.getLogger(__name__)


def frame_summary(
    detections: pd.DataFrame,
    n_frames: int,
    classes: list[str],
    thresholds: list[float],
    operating: dict[str, float] | None = None,
    floor: float = 0.05,
    iou_threshold: float = 0.7,
    existence: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-frame detection counts per class above raw thresholds and at the operating thresholds.

    ``detections`` needs ``frame_index`` (0 to ``n_frames`` - 1), ``class_name``
    and ``score``; when ``operating`` is given it also needs
    ``score_calibrated``, ``rank`` and the box columns. The result has one row
    per frame and columns ``<class>_ge<threshold>`` and ``n_ge<threshold>``
    (raw score thresholds), ``<class>_op`` (calibrated score at or above the
    class threshold, every row counted), ``<class>_res`` (objects after
    cross-class resolution at ``iou_threshold``, counted in their highest
    class), ``<class>_exp`` (the same objects, each split over its two
    highest classes in proportion to their calibrated scores) and
    ``<class>_hyp`` (diagnostic: sum of calibrated scores of all stored
    hypotheses of rank one and two above ``floor``). Objects exist under the
    ``existence`` thresholds (the chosen object rule), which default to the
    operating thresholds.
    """
    index = pd.RangeIndex(n_frames, name="frame_index")
    blocks: list[pd.DataFrame] = []

    def counts(sub: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if sub.empty:
            table = pd.DataFrame(0, index=index, columns=classes)
        else:
            table = (
                sub.groupby(["frame_index", "class_name"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=index, columns=classes, fill_value=0)
            )
        table = table.astype("int32")
        table[f"n{suffix}"] = table.sum(axis=1).astype("int32")
        table.columns = [f"{c}{suffix}" if c in classes else c for c in table.columns]
        return table

    for thr in thresholds:
        suffix = f"_ge{thr:g}".replace(".", "p")
        blocks.append(counts(detections[detections["score"] >= thr], suffix))
    if operating:
        thr_series = detections["class_name"].map(operating).fillna(1.0)
        blocks.append(counts(detections[detections["score_calibrated"] >= thr_series], "_op"))
        if "rank" in detections.columns and not detections.empty:
            kept = resolve_cross_class(
                detections, existence or operating, "frame_index", iou_threshold=iou_threshold
            )
            blocks.append(counts(detections[kept], "_res"))
            expected = expected_counts(detections, index, classes, kept)
            expected["n_exp"] = expected.sum(axis=1).astype("float32")
            expected.columns = [f"{c}_exp" if c in classes else c for c in expected.columns]
            blocks.append(expected)
            hyp = hypothesis_sums(detections, index, classes, floor=floor)
            hyp["n_hyp"] = hyp.sum(axis=1).astype("float32")
            hyp.columns = [f"{c}_hyp" if c in classes else c for c in hyp.columns]
            blocks.append(hyp)
        else:
            blocks.append(counts(detections.iloc[0:0], "_res"))
    return pd.concat(blocks, axis=1).reset_index()


def _load_model(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, Path], device: str, floor: float):
    weights = paths["weights_dir"] / f"{exp['name']}_seed{exp['seed']}.pth"
    if exp["framework"] == "retinanet":
        from .detection import retinanet

        model = retinanet.load_trained(
            weights, int(cfg["dataset"]["num_classes"]), int(exp["input_size"]), device, score_threshold=floor
        )

        def run(paths_batch: list[Path], batch_size: int, workers: int) -> pd.DataFrame:
            return retinanet.predict(model, paths_batch, device, batch_size=batch_size, num_workers=workers)

        return run
    from .detection.predict import load_detr_model, predict_images

    model, post = load_detr_model(paths["repo"], paths["home_run"] / "framework_config.yml", weights, device)

    def run_detr(paths_batch: list[Path], batch_size: int, workers: int) -> pd.DataFrame:
        return predict_images(
            model,
            post,
            paths_batch,
            int(exp["input_size"]),
            device,
            batch_size=batch_size,
            num_workers=workers,
            score_threshold=floor,
        )

    return run_detr


def run_infer_transect(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    transect: str = "",
    camera: str = "Left",
    folders: str | None = None,
) -> Path:
    """Run one experiment on every frame of one camera of one transect and write the tables."""
    exp = _experiment(cfg, experiment)
    exp["seed"] = seed
    paths = _paths(cfg, exp, seed)
    icfg = cfg.get("inference", {})
    floor = float(icfg.get("score_floor", 0.05))
    thresholds = [float(t) for t in icfg.get("count_thresholds", [0.05, 0.1, 0.2, 0.3, 0.5])]
    transect_dir = Path(cfg["paths"]["source_data"], transect)
    target_dir = Path(paths["home_run"].parents[2], "inference", f"{experiment}_seed{seed}", transect, camera)
    target_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = target_dir / "parts"
    parts_dir.mkdir(exist_ok=True)

    coco = load_coco(paths["coco_dir"] / "val.json")
    names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    classes = [names[k] for k in sorted(names)]
    calibrator = None
    calib_path = paths["home_run"] / "calibration" / "calibration.json"
    if calib_path.is_file():
        with open(calib_path, encoding="utf-8") as fh:
            calibrator = Calibrator.from_dict(json.load(fh)["calibrator"])
    operating: dict[str, float] = {}
    thr_path = paths["home_run"] / "calibration" / "thresholds_calibrated.csv"
    if thr_path.is_file():
        t = pd.read_csv(thr_path)
        operating = dict(zip(t["class_name"], t["threshold"].astype(float), strict=True))
    existence = dict(operating)
    rule_path = paths["home_run"] / "calibration" / "object_rule.json"
    object_rule: dict[str, Any] = {}
    if rule_path.is_file():
        with open(rule_path, encoding="utf-8") as fh:
            object_rule = json.load(fh)
        existence = effective_thresholds(
            str(object_rule.get("chosen", "per_class")),
            operating,
            object_rule.get("pooled_threshold"),
            list(object_rule.get("label_classes", [])),
        )

    frames = list_camera_frames(transect_dir, camera)
    frames = frames[frames["is_image"] & ~frames["archive_duplicate"] & ~frames["is_zip"]].copy()
    frames = frames.sort_values(["folder", "timestamp"]).reset_index(drop=True)
    frames["frame_index"] = np.arange(len(frames))
    wanted = [f.strip() for f in folders.split(",")] if folders else sorted(frames["folder"].unique())
    frames = frames[frames["folder"].isin(wanted)].copy()
    frames["frame_index"] = np.arange(len(frames))

    device = "cuda" if _cuda_available() else "cpu"
    run = _load_model(cfg, exp, paths, device, floor)
    batch_size = int(icfg.get("batch_size", 16))
    workers = int(cfg.get("n_workers", 8))
    timing: dict[str, float] = {}
    for folder, group in frames.groupby("folder", sort=True):
        part = parts_dir / f"{folder}.parquet"
        if part.is_file() and not force:
            continue
        t0 = time.time()
        paths_batch = [Path(transect_dir, camera, folder, fn) for fn in group["filename"]]
        dets = run(paths_batch, batch_size, workers)
        dets["frame_index"] = group["frame_index"].to_numpy()[dets["image_index"].to_numpy()]
        dets = dets.drop(columns=["image_index"])
        dets["class_name"] = dets["category_id"].map(names)
        dets["score_calibrated"] = calibrator.apply(dets["score"].to_numpy()) if calibrator else dets["score"]
        dets = assign_queries(dets, "frame_index")
        dets.to_parquet(part, index=False)
        timing[folder] = time.time() - t0
        log.info(
            "%s %s %s: %d frames, %d detections, %.0f s",
            transect,
            camera,
            folder,
            len(group),
            len(dets),
            timing[folder],
        )

    parts = [
        pd.read_parquet(parts_dir / f"{f}.parquet") for f in wanted if (parts_dir / f"{f}.parquet").is_file()
    ]
    dets = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not dets.empty:
        # Each part numbers its queries from zero, so queries are assigned again over the whole table.
        dets = assign_queries(dets, "frame_index")
    dets = dets.merge(
        frames[["frame_index", "folder", "filename", "timestamp"]], on="frame_index", how="left"
    )
    dets.insert(0, "camera", camera)
    dets.insert(0, "transect", transect)
    det_path = target_dir / "detections.parquet"
    dets.to_parquet(det_path, index=False)

    dets["static"] = False
    templates = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    static_per_class: dict[str, Any] = {}
    if operating and not dets.empty:
        dets["kept_resolved"] = resolve_cross_class(
            dets, existence, "frame_index", iou_threshold=float(icfg.get("resolution_iou", 0.7))
        )
        # Fixed structure (housing, warp) recurs at one box across a folder; such boxes are not counted.
        scfg = icfg.get("static", {})
        templates = static_templates(
            dets[dets["kept_resolved"]],
            frames.groupby("folder").size().to_dict(),
            float(scfg.get("cell_px", 16)),
            float(scfg.get("min_folder_share", 0.2)),
        )
        dets["static"] = flag_static(dets, templates, float(scfg.get("iou", 0.5)))
        objects = dets["kept_resolved"].to_numpy().copy()
        static_per_class = static_summary(dets[objects], dets.loc[objects, "static"].to_numpy())
        dets.loc[dets["static"], "kept_resolved"] = False
        dets.to_parquet(det_path, index=False)
    summary_table = frame_summary(
        dets[~dets["static"]],
        len(frames),
        classes,
        thresholds,
        operating or None,
        floor,
        float(icfg.get("resolution_iou", 0.7)),
        existence or None,
    )
    per_frame = frames[["frame_index", "folder", "filename", "timestamp", "size_bytes"]].merge(
        summary_table, on="frame_index"
    )
    per_frame["n_static"] = 0
    if operating and not dets.empty:
        above = dets["score_calibrated"] >= dets["class_name"].map(operating).fillna(1.0)
        n_static = dets[dets["static"] & above].groupby("frame_index").size()
        per_frame["n_static"] = per_frame["frame_index"].map(n_static).fillna(0).astype("int32")
    per_frame.insert(0, "camera", camera)
    per_frame.insert(0, "transect", transect)
    frame_path = target_dir / "frames.parquet"
    per_frame.to_parquet(frame_path, index=False)

    floor_col = f"n_ge{thresholds[0]:g}".replace(".", "p")
    qc: dict[str, Any] = {
        "object_rule": object_rule.get("chosen", "per_class"),
        "pooled_threshold": object_rule.get("pooled_threshold"),
        "experiment": experiment,
        "seed": seed,
        "transect": transect,
        "camera": camera,
        "n_frames": int(len(frames)),
        "n_detections": int(len(dets)),
        "score_floor": floor,
        "calibrated": calibrator is not None,
        "operating_thresholds": operating,
        "no_detection_rate_at_floor": float((per_frame[floor_col] == 0).mean()) if len(per_frame) else None,
        "no_detection_rate_at_operating": float((per_frame["n_op"] == 0).mean())
        if "n_op" in per_frame
        else None,
        "no_detection_rate_by_folder": per_frame.groupby("folder")[floor_col]
        .apply(lambda s: float((s == 0).mean()))
        .to_dict(),
        "detections_per_class_at_floor": dets["class_name"].value_counts().to_dict() if len(dets) else {},
        "score_quantiles_by_class": {
            c: {q: float(v) for q, v in sub["score"].quantile([0.1, 0.5, 0.9]).round(4).items()}
            for c, sub in dets.groupby("class_name")
        }
        if len(dets)
        else {},
        "static_templates": templates.round(1).to_dict("records"),
        "static_objects_per_class": static_per_class,
        "frames_with_static": int((per_frame["n_static"] > 0).sum()),
        "seconds_by_folder": timing,
        "frames_per_second": float(len(frames) / sum(timing.values())) if timing else None,
    }
    with open(target_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2, default=str)
    write_manifest(
        target_dir,
        "infer-transect",
        {**cfg, "experiment": experiment, "seed": seed, "transect": transect, "camera": camera},
        {"transect": transect_dir, "weights": paths["weights_dir"] / f"{experiment}_seed{seed}.pth"},
        [det_path, frame_path, target_dir / "summary.json"],
    )
    return target_dir
