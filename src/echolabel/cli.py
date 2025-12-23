"""Command line entry point. One sub-command per stage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from . import (
    acoustics_stage,
    alignment_stage,
    attribution_stage,
    audit,
    camera,
    camera_audit,
    dataset,
    detector_stage,
    error_stage,
    eval_stage,
    figures_stage,
    inference,
    models_stage,
    preprocess_stage,
    review_stage,
    sample,
    window_review_stage,
    windows_stage,
)
from .config import load_config, resolve_path
from .detection import calibration

# name -> (runner, default output path parts under project_dir, extra argument names)
STAGES: dict[str, tuple[Any, tuple[str, ...], tuple[str, ...]]] = {
    "audit-annotations": (audit.run_audit_annotations, ("runs", "audit", "annotations"), ()),
    "audit-transects": (audit.run_audit_transects, ("runs", "audit", "transects"), ()),
    "audit-zarr": (audit.run_audit_zarr, ("runs", "audit", "zarr"), ()),
    "pull-sample": (sample.run_pull_sample, ("samples",), ()),
    "build-annotation-dataset": (dataset.run_build_annotation_dataset, ("runs", "dataset"), ()),
    "train-detector": (
        detector_stage.run_train_detector,
        ("runs", "detector"),
        ("experiment", "seed", "resume"),
    ),
    "predict-detector": (
        detector_stage.run_predict_detector,
        ("runs", "detector"),
        ("experiment", "seed", "split", "checkpoint"),
    ),
    "evaluate-detector": (
        detector_stage.run_evaluate_detector,
        ("runs", "detector"),
        ("experiment", "seed", "split"),
    ),
    "calibrate-detector": (
        calibration.run_calibrate_detector,
        ("runs", "detector"),
        ("experiment", "seed"),
    ),
    "summarize-detectors": (detector_stage.run_summarize_detectors, ("runs", "detector_summary"), ()),
    "infer-transect": (
        inference.run_infer_transect,
        ("runs", "inference"),
        ("experiment", "seed", "transect", "camera", "folders"),
    ),
    "build-camera-series": (
        camera.run_build_camera_series,
        ("runs", "camera_series"),
        ("experiment", "seed", "transect", "camera"),
    ),
    "audit-camera-depth": (camera_audit.run_audit_camera_depth, ("runs", "camera_depth_audit"), ()),
    "audit-transect-acoustics": (
        acoustics_stage.run_audit_transect_acoustics,
        ("runs", "acoustics", "exploration"),
        (),
    ),
    "preprocess-transects": (
        preprocess_stage.run_preprocess_transects,
        ("runs", "acoustics", "transects"),
        (),
    ),
    "downsample-cruise": (preprocess_stage.run_downsample_cruise, ("runs", "acoustics", "cruise"), ()),
    "align-transect": (
        alignment_stage.run_align_transect,
        ("runs", "alignment"),
        ("experiment", "seed", "transect", "camera"),
    ),
    "review-detections": (
        review_stage.run_review_detections,
        ("runs", "review"),
        ("experiment", "seed", "transect", "camera"),
    ),
    "build-windows": (
        windows_stage.run_build_windows,
        ("runs", "windows"),
        ("experiment", "seed", "variant"),
    ),
    "make-splits": (windows_stage.run_make_splits, ("runs", "windows"), ("experiment", "seed", "variant")),
    "train-acoustic": (
        models_stage.run_train_acoustic,
        ("runs", "acoustic"),
        ("experiment", "seed", "fold", "protocol"),
    ),
    "pretrain-mae": (models_stage.run_pretrain_mae, ("runs", "acoustic_pretraining"), ("resume",)),
    "evaluate-acoustic": (eval_stage.run_evaluate_acoustic, ("runs", "acoustic_summary"), ()),
    "plot-results": (figures_stage.run_plot_results, ("runs", "acoustic_figures"), ()),
    "analyse-errors": (error_stage.run_analyse_errors, ("runs", "acoustic_errors"), ()),
    "review-windows": (window_review_stage.run_review_windows, ("runs", "window_review"), ()),
    "summarize-attribution": (
        attribution_stage.run_summarize_attribution,
        ("runs", "attribution_summary"),
        (),
    ),
    "attribute-acoustic": (
        attribution_stage.run_attribute_acoustic,
        ("runs", "attribution"),
        ("experiment", "seed", "fold", "protocol"),
    ),
}


def _setup_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echolabel", description="Pipeline stages.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, extras) in STAGES.items():
        p = sub.add_parser(name, help=f"run the {name} stage")
        p.add_argument("--config", required=True, help="stage configuration YAML")
        p.add_argument("--paths", required=True, help="machine paths YAML")
        p.add_argument("--out", default=None, help="output directory (default: under <project_dir>)")
        p.add_argument("--force", action="store_true", help="rerun even if the manifest is current")
        p.add_argument("--log-level", default="INFO")
        if "experiment" in extras:
            p.add_argument("--experiment", required=True, help="experiment name from the configuration")
        if "seed" in extras:
            p.add_argument("--seed", type=int, default=0)
        if "split" in extras:
            p.add_argument("--split", default="val", help="train, val, test or test_2024")
        if "resume" in extras:
            p.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
        if "transect" in extras:
            p.add_argument("--transect", required=True, help="transect directory name")
        if "camera" in extras:
            p.add_argument("--camera", default="Left", help="Left or Right")
        if "folders" in extras:
            p.add_argument("--folders", default=None, help="comma-separated subset of folders (default: all)")
        if "fold" in extras:
            p.add_argument(
                "--fold", required=True, help="fold name of the protocol, for example test_20180505T0637Z"
            )
        if "variant" in extras:
            p.add_argument(
                "--variant",
                default="",
                help="lag-model variant of the mapping (prior, plateau_event, series)",
            )
        if "protocol" in extras:
            p.add_argument("--protocol", default="", help="split protocol (default: the experiment's)")
        if "checkpoint" in extras:
            p.add_argument(
                "--checkpoint", default=None, help="checkpoint path (default: the stored best weights)"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.paths)
    runner, default_sub, extras = STAGES[args.command]
    out_dir = Path(args.out) if args.out else resolve_path(cfg, "project_dir", *default_sub)
    _setup_logging(args.log_level, out_dir / "run.log")
    logging.getLogger(__name__).info("command %s, output %s", args.command, out_dir)
    kwargs = {name: getattr(args, name) for name in extras}
    runner(cfg, out_dir, force=args.force, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
