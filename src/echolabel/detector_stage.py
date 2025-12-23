"""Stage 2: detector training, inference and evaluation.

Training delegates to the external framework of the experiment through a
generated configuration file and records the run in a manifest; inference and
evaluation use the project's own code so that every framework is scored the
same way.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import resolve_path
from .detection.evaluate import evaluate_predictions, load_coco
from .detection.frameworks import (
    best_checkpoint,
    build_framework_config,
    train_command,
    write_framework_config,
)
from .manifest import manifest_is_current, write_manifest

log = logging.getLogger(__name__)


def _experiment(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    experiments = cfg.get("experiments", {})
    if name not in experiments:
        raise KeyError(f"experiment {name!r} not in configuration; known: {sorted(experiments)}")
    exp = dict(cfg.get("defaults", {}))
    exp.update(experiments[name])
    exp["name"] = name
    return exp


def _paths(cfg: dict[str, Any], exp: dict[str, Any], seed: int) -> dict[str, Path]:
    framework = exp["framework"]
    fw = cfg["frameworks"][framework]
    size = exp["size"]
    project = resolve_path(cfg, "project_dir")
    scratch = resolve_path(cfg, "scratch_dir")
    dcfg = cfg["dataset"]
    base_configs = fw.get("base_configs", {})
    pretrained = fw.get("pretrained", {})
    return {
        "repo": Path(project, "external", fw.get("repo", "")),
        "base_config": Path(project, "external", fw.get("repo", ""), base_configs.get(size, "")),
        "pretrained": Path(scratch, pretrained[size])
        if isinstance(pretrained, dict)
        else Path(scratch, "pretrained"),
        "images_root": resolve_path(cfg, "annotated_data", dcfg.get("images_subdir", "images")),
        "coco_dir": Path(project, dcfg.get("coco_dir", "runs/dataset/coco")),
        "scratch_run": Path(scratch, "runs", "detector", exp["name"], f"seed{seed}"),
        "home_run": Path(project, "runs", "detector", exp["name"], f"seed{seed}"),
        "weights_dir": Path(project, "weights", "detector"),
    }


def _parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    if not log_path.is_file():
        return pd.DataFrame()
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    table = pd.DataFrame(rows)
    if "test_coco_eval_bbox" in table.columns:
        stats = table["test_coco_eval_bbox"].apply(lambda v: v if isinstance(v, list) else [None] * 12)
        table["val_map"] = stats.apply(lambda v: v[0])
        table["val_map50"] = stats.apply(lambda v: v[1])
        table["val_map75"] = stats.apply(lambda v: v[2])
        table = table.drop(columns=["test_coco_eval_bbox"])
    return table


def _train_framework(
    cfg: dict[str, Any],
    exp: dict[str, Any],
    paths: dict[str, Path],
    home_run: Path,
    scratch_run: Path,
    framework_out: Path,
    train_json: Path,
    val_json: Path,
    seed: int,
    resume: bool,
) -> Path:
    """Write the framework configuration and run the framework's training script; return the config path."""
    fw_cfg = build_framework_config(
        framework=exp["framework"],
        base_config=paths["base_config"],
        output_dir=framework_out,
        images_root=paths["images_root"],
        train_json=train_json,
        val_json=val_json,
        num_classes=int(cfg["dataset"]["num_classes"]),
        input_size=int(exp["input_size"]),
        batch_size=int(exp["batch_size"]),
        epochs=int(exp["epochs"]),
        lr=float(exp["lr"]),
        backbone_lr=float(exp["backbone_lr"]),
        weight_decay=float(exp.get("weight_decay", 0.0001)),
        warmup_iterations=int(exp.get("warmup_iterations", 500)),
        checkpoint_freq=int(exp.get("checkpoint_freq", 10)),
        num_workers=int(cfg.get("n_workers", 8)),
        aug=dict(exp.get("augmentation", {})),
        stop_augmentation_epochs_before_end=int(exp.get("stop_augmentation_epochs_before_end", 8)),
        pretrained_backbone=False,
    )
    config_path = write_framework_config(fw_cfg, home_run / "framework_config.yml")
    shutil.copy2(config_path, scratch_run / "framework_config.yml")
    resume_ckpt = framework_out / "last.pth" if resume and (framework_out / "last.pth").is_file() else None
    cmd = train_command(
        exp["framework"],
        config_path,
        framework_out,
        seed,
        pretrained=paths["pretrained"],
        resume=resume_ckpt,
        python=sys.executable,
    )
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log.info("launching %s in %s", " ".join(cmd), paths["repo"])
    with open(home_run / "train_stdout.log", "a", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, cwd=str(paths["repo"]), env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"training failed with exit code {proc.returncode}; see {home_run / 'train_stdout.log'}"
        )
    return config_path


def _train_retinanet(
    cfg: dict[str, Any],
    exp: dict[str, Any],
    paths: dict[str, Path],
    home_run: Path,
    framework_out: Path,
    train_json: Path,
    val_json: Path,
    seed: int,
    resume: bool,
) -> Path:
    """Train the torchvision RetinaNet reference in-process; return the path of the recorded settings."""
    from .detection import retinanet

    os.environ.setdefault("TORCH_HOME", str(paths["pretrained"] / "torch"))
    device = "cuda" if _cuda_available() else "cpu"
    settings = {
        "framework": "retinanet",
        "num_classes": int(cfg["dataset"]["num_classes"]),
        "input_size": int(exp["input_size"]),
        "epochs": int(exp["epochs"]),
        "batch_size": int(exp["batch_size"]),
        "lr": float(exp["lr"]),
        "weight_decay": float(exp.get("weight_decay", 0.0001)),
        "warmup_iterations": int(exp.get("warmup_iterations", 500)),
        "augmentation": dict(exp.get("augmentation", {})),
        "seed": seed,
    }
    config_path = home_run / "framework_config.yml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(settings, fh, sort_keys=False)
    retinanet.train(
        train_json=train_json,
        val_json=val_json,
        images_root=paths["images_root"],
        output_dir=framework_out,
        num_classes=settings["num_classes"],
        input_size=settings["input_size"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
        warmup_iterations=settings["warmup_iterations"],
        aug=settings["augmentation"],
        seed=seed,
        device=device,
        num_workers=int(cfg.get("n_workers", 8)),
        pretrained=True,
        resume=resume,
        eval_batch_size=int(cfg.get("evaluation", {}).get("batch_size", 16)),
    )
    return config_path


def run_train_detector(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    resume: bool = False,
) -> Path:
    """Train one experiment with one seed through its framework and record the result."""
    exp = _experiment(cfg, experiment)
    paths = _paths(cfg, exp, seed)
    home_run = paths["home_run"]
    scratch_run = paths["scratch_run"]
    train_json = paths["coco_dir"] / "train.json"
    val_json = paths["coco_dir"] / "val.json"
    inputs: dict[str, str | Path] = {"train": train_json, "val": val_json, "pretrained": paths["pretrained"]}
    if (
        not force
        and not resume
        and manifest_is_current(home_run, {**cfg, "experiment": experiment, "seed": seed}, inputs)
    ):
        log.info("experiment %s seed %d is up to date in %s", experiment, seed, home_run)
        return home_run
    home_run.mkdir(parents=True, exist_ok=True)
    scratch_run.mkdir(parents=True, exist_ok=True)
    framework_out = scratch_run / "framework"
    framework_out.mkdir(parents=True, exist_ok=True)

    if exp["framework"] == "retinanet":
        config_path = _train_retinanet(
            cfg, exp, paths, home_run, framework_out, train_json, val_json, seed, resume
        )
    else:
        config_path = _train_framework(
            cfg, exp, paths, home_run, scratch_run, framework_out, train_json, val_json, seed, resume
        )

    table = _parse_log(framework_out / "log.txt")
    table.to_csv(home_run / "training_log.csv", index=False)
    best = best_checkpoint(exp["framework"], framework_out)
    outputs: list[str | Path] = [config_path, home_run / "training_log.csv"]
    summary: dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "framework": exp["framework"],
        "input_size": exp["input_size"],
        "epochs_run": int(len(table)),
        "best_checkpoint": str(best) if best else None,
    }
    if not table.empty and "val_map" in table.columns:
        i = int(table["val_map"].astype(float).idxmax())
        summary["best_epoch"] = int(table.loc[i, "epoch"]) if "epoch" in table.columns else i
        summary["best_val_map"] = float(table.loc[i, "val_map"])
        summary["best_val_map50"] = float(table.loc[i, "val_map50"])
        summary["last_val_map"] = float(table["val_map"].astype(float).iloc[-1])
    if best is not None:
        paths["weights_dir"].mkdir(parents=True, exist_ok=True)
        target = paths["weights_dir"] / f"{experiment}_seed{seed}.pth"
        shutil.copy2(best, target)
        summary["weights"] = str(target)
        outputs.append(target)
    with open(home_run / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    outputs.append(home_run / "summary.json")
    write_manifest(
        home_run, "train-detector", {**cfg, "experiment": experiment, "seed": seed}, inputs, outputs
    )
    return home_run


def run_predict_detector(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    split: str = "val",
    checkpoint: str | None = None,
) -> Path:
    """Run a trained experiment on one split and store every detection above a low score threshold."""
    from .detection.predict import load_detr_model, predict_coco_split

    exp = _experiment(cfg, experiment)
    paths = _paths(cfg, exp, seed)
    home_run = paths["home_run"]
    coco_path = paths["coco_dir"] / f"{split}.json"
    ckpt = Path(checkpoint) if checkpoint else paths["weights_dir"] / f"{experiment}_seed{seed}.pth"
    target = home_run / f"predictions_{split}.parquet"
    if target.is_file() and not force and target.stat().st_mtime > ckpt.stat().st_mtime:
        log.info("predictions for %s exist in %s", split, target)
        return target
    ecfg = cfg.get("evaluation", {})
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or _cuda_available() else "cpu"
    coco = load_coco(coco_path)
    if exp["framework"] == "retinanet":
        from .detection import retinanet

        model = retinanet.load_trained(
            ckpt,
            int(cfg["dataset"]["num_classes"]),
            int(exp["input_size"]),
            device,
            score_threshold=float(ecfg.get("score_threshold", 0.001)),
        )
        preds = retinanet.predict_coco(
            model,
            coco,
            paths["images_root"],
            device,
            batch_size=int(ecfg.get("batch_size", 16)),
            num_workers=int(cfg.get("n_workers", 8)),
        )
    else:
        model, post = load_detr_model(paths["repo"], home_run / "framework_config.yml", ckpt, device)
        preds = predict_coco_split(
            model,
            post,
            coco,
            paths["images_root"],
            int(exp["input_size"]),
            device,
            batch_size=int(ecfg.get("batch_size", 16)),
            num_workers=int(cfg.get("n_workers", 8)),
            score_threshold=float(ecfg.get("score_threshold", 0.001)),
        )
    preds.to_parquet(target, index=False)
    log.info("wrote %d detections for %d images to %s", len(preds), len(coco["images"]), target)
    return target


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        return False


def run_evaluate_detector(
    cfg: dict[str, Any],
    out_dir: Path,
    force: bool = False,
    experiment: str = "",
    seed: int = 0,
    split: str = "val",
) -> Path:
    """Evaluate stored predictions of one experiment and split; write tables and a summary."""
    exp = _experiment(cfg, experiment)
    paths = _paths(cfg, exp, seed)
    home_run = paths["home_run"]
    coco_path = paths["coco_dir"] / f"{split}.json"
    preds_path = home_run / f"predictions_{split}.parquet"
    eval_dir = home_run / f"eval_{split}"
    inputs: dict[str, str | Path] = {"predictions": preds_path, "ground_truth": coco_path}
    if not force and manifest_is_current(
        eval_dir, {**cfg, "experiment": experiment, "seed": seed, "split": split}, inputs
    ):
        log.info("evaluation of %s is up to date in %s", split, eval_dir)
        return eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    ecfg = cfg.get("evaluation", {})
    preds = pd.read_parquet(preds_path)
    thresholds = None
    thr_path = home_run / "eval_val" / "thresholds.csv"
    if split != "val" and thr_path.is_file():
        t = pd.read_csv(thr_path)
        thresholds = dict(zip(t["category_id"].astype(int), t["threshold"].astype(float), strict=True))
    result = evaluate_predictions(
        coco_path,
        preds,
        iou_threshold=float(ecfg.get("iou_threshold", 0.5)),
        n_boot=int(ecfg.get("n_boot", 200)),
        seed=int(cfg.get("seed", 0)),
        thresholds=thresholds,
        min_gt_for_macro=int(ecfg.get("min_gt_for_macro", 10)),
        class_sets=ecfg.get("class_sets"),
    )
    outputs: list[str | Path] = []
    for name in ("curves", "thresholds", "slices"):
        p = eval_dir / f"{name}.csv"
        result[name].to_csv(p, index=False)
        outputs.append(p)
    result["confusion"].to_csv(eval_dir / "confusion.csv")
    outputs.append(eval_dir / "confusion.csv")
    summary = {
        "experiment": experiment,
        "seed": seed,
        "split": split,
        "coco": result["coco"],
        "n_gt": result["n_gt"],
        "n_images": result["n_images"],
        "n_predictions": result["n_predictions"],
        "thresholds_source": "val" if thresholds is not None else "this split",
    }
    slices = result["slices"]
    summary["map50_by_slice"] = {
        r["slice"]: {
            "value": r["value"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "n_images": int(r["n_images"]),
        }
        for _, r in slices[slices["metric"] == "map50"].iterrows()
    }
    summary["class_set_map50_by_slice"] = {
        f"{r['metric']}@{r['slice']}": {
            "value": r["value"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "n_images": int(r["n_images"]),
            "n_classes": int(r.get("n_classes", 0)),
        }
        for _, r in slices[slices["metric"].str.startswith("map50_")].iterrows()
    }
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    outputs.append(eval_dir / "summary.json")
    write_manifest(
        eval_dir,
        "evaluate-detector",
        {**cfg, "experiment": experiment, "seed": seed, "split": split},
        inputs,
        outputs,
    )
    return eval_dir


def _operating_thresholds(run_dir: Path) -> dict[int, float]:
    """Per-class operating thresholds of one run, on calibrated scores, as inference applies them.

    The thresholds chosen on calibrated validation scores are preferred; the
    raw-score thresholds of the evaluation step are the fallback when no
    calibration was run.
    """
    path = run_dir / "calibration" / "thresholds_calibrated.csv"
    if not path.is_file():
        path = run_dir / "eval_val" / "thresholds.csv"
    if not path.is_file():
        return {}
    t = pd.read_csv(path)
    return dict(zip(t["category_id"].astype(int), t["threshold"].astype(float), strict=True))


def resolution_checks(cfg: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-class resolution on annotated splits: counts before and after resolution, and suppression pairs.

    For each run and each check under ``summary.resolution_checks`` the
    predictions of the split are restricted to the slice, given queries and
    ranks, and resolved at the operating thresholds; the table compares the
    number of boxes with the unresolved and resolved detection counts and the
    expected count per class, and the pair table says which classes suppress
    which.
    """
    from .detection.evaluate import gt_table, image_table, load_coco, year_slices
    from .detection.resolve import (
        assign_queries,
        expected_counts,
        hypothesis_sums,
        pair_counts,
        resolve_cross_class,
    )

    checks = cfg.get("summary", {}).get("resolution_checks", [])
    icfg = cfg.get("inference", {})
    iou = float(icfg.get("resolution_iou", 0.7))
    floor = float(icfg.get("score_floor", 0.05))
    count_rows, pair_rows = [], []
    for run_dir in sorted(root.glob("*/seed*")):
        experiment, seed = run_dir.parent.name, run_dir.name
        if experiment.startswith("smoke"):
            continue
        thresholds_by_id = _operating_thresholds(run_dir)
        calib_path = run_dir / "calibration" / "calibration.json"
        for check in checks:
            split, slice_name = str(check["split"]), str(check["slice"])
            pred_path = run_dir / f"predictions_{split}.parquet"
            try:
                seed_number = int(seed.replace("seed", ""))
                coco_path = (
                    _paths(cfg, _experiment(cfg, experiment), seed_number)["coco_dir"] / f"{split}.json"
                )
            except KeyError:
                continue
            if not (pred_path.is_file() and coco_path.is_file() and thresholds_by_id):
                continue
            coco = load_coco(coco_path)
            images = image_table(coco)
            names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
            slices = year_slices(images)
            if slice_name not in slices:
                continue
            ids = set(int(i) for i in slices[slice_name])
            gts = gt_table(coco)
            gts = gts[(gts["iscrowd"] == 0) & gts["image_id"].isin(ids)]
            preds = pd.read_parquet(pred_path)
            preds = preds[preds["image_id"].isin(ids)].copy()
            preds["class_name"] = preds["category_id"].map(names)
            preds["score_calibrated"] = _apply_calibration(calib_path, preds["score"].to_numpy())
            preds = assign_queries(preds, "image_id")
            thresholds = {names[c]: t for c, t in thresholds_by_id.items() if c in names}
            kept = resolve_cross_class(preds, thresholds, "image_id", iou_threshold=iou)
            thr = preds["class_name"].map(thresholds).fillna(1.0).to_numpy(dtype=float)
            above = preds["score_calibrated"].to_numpy(dtype=float) >= thr
            index = pd.Index(sorted(ids), name="image_id")
            classes = [names[c] for c in sorted(names)]
            expected = expected_counts(preds, index, classes, kept, "image_id").sum(axis=0)
            hypotheses = hypothesis_sums(preds, index, classes, "image_id", floor=floor).sum(axis=0)
            for class_name in classes:
                c_id = [k for k, v in names.items() if v == class_name][0]
                count_rows.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "split": split,
                        "slice": slice_name,
                        "class_name": class_name,
                        "n_boxes": int((gts["category_id"] == c_id).sum()),
                        "unresolved": int((above & (preds["class_name"] == class_name).to_numpy()).sum()),
                        "resolved": int((kept & (preds["class_name"] == class_name).to_numpy()).sum()),
                        "expected": float(expected.get(class_name, 0.0)),
                        "hypotheses": float(hypotheses.get(class_name, 0.0)),
                        "threshold": float(thresholds.get(class_name, 1.0)),
                    }
                )
            pairs = pair_counts(preds, thresholds, kept, "image_id", iou_threshold=iou)
            for r in pairs.itertuples(index=False):
                pair_rows.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "split": split,
                        "slice": slice_name,
                        "kept_class": r.kept_class,
                        "suppressed_class": r.suppressed_class,
                        "count": int(r.count),
                    }
                )
    pair_columns = ["experiment", "seed", "split", "slice", "kept_class", "suppressed_class", "count"]
    return pd.DataFrame(count_rows), pd.DataFrame(pair_rows, columns=pair_columns)


def _apply_calibration(calib_path: Path, scores: np.ndarray) -> np.ndarray:
    """Calibrated scores from a stored calibration file, or the raw scores when there is none."""
    if not calib_path.is_file():
        return scores
    from .detection.calibration import Calibrator

    with open(calib_path, encoding="utf-8") as fh:
        c = json.load(fh)
    try:
        return Calibrator.from_dict(c["calibrator"]).apply(scores)
    except Exception as exc:  # pragma: no cover
        log.warning("calibration could not be applied: %s", exc)
        return scores


def object_rule_reports(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """Boxes, counted, expected, precision and recall per class under both object rules per configured slice.

    The pooled threshold and the chosen rule come from each run's
    ``calibration/object_rule.json`` (validation only); the slices listed under
    ``summary.object_rule_slices`` are then reported for both rules.
    """
    from .detection.evaluate import gt_table, image_table, load_coco, year_slices
    from .detection.objects import RULES, effective_thresholds, rule_table
    from .detection.resolve import assign_queries

    slices_cfg = cfg.get("summary", {}).get("object_rule_slices", [])
    iou_suppress = float(cfg.get("inference", {}).get("resolution_iou", 0.7))
    iou_match = float(cfg.get("evaluation", {}).get("iou_threshold", 0.5))
    rows = []
    for run_dir in sorted(root.glob("*/seed*")):
        experiment, seed = run_dir.parent.name, run_dir.name
        rule_path = run_dir / "calibration" / "object_rule.json"
        if experiment.startswith("smoke") or not rule_path.is_file():
            continue
        with open(rule_path, encoding="utf-8") as fh:
            rule = json.load(fh)
        per_class = {}
        thr_table = run_dir / "calibration" / "thresholds_calibrated.csv"
        if thr_table.is_file():
            t = pd.read_csv(thr_table)
            per_class = dict(zip(t["class_name"], t["threshold"].astype(float), strict=True))
        for spec in slices_cfg:
            split, slice_name = str(spec["split"]), str(spec["slice"])
            pred_path = run_dir / f"predictions_{split}.parquet"
            try:
                seed_number = int(seed.replace("seed", ""))
                coco_path = (
                    _paths(cfg, _experiment(cfg, experiment), seed_number)["coco_dir"] / f"{split}.json"
                )
            except KeyError:
                continue
            if not (pred_path.is_file() and coco_path.is_file()):
                continue
            coco = load_coco(coco_path)
            images = image_table(coco)
            names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
            slices = year_slices(images)
            if slice_name not in slices:
                continue
            gts = gt_table(coco)
            preds = pd.read_parquet(pred_path)
            preds["class_name"] = preds["category_id"].map(names)
            preds["score_calibrated"] = _apply_calibration(
                run_dir / "calibration" / "calibration.json", preds["score"].to_numpy()
            )
            preds = assign_queries(preds, "image_id")
            for name in RULES:
                thr = effective_thresholds(
                    name, per_class, rule.get("pooled_threshold"), list(rule.get("label_classes", []))
                )
                table = rule_table(preds, gts, names, slices[slice_name], thr, iou_suppress, iou_match)
                table.insert(0, "chosen", rule.get("chosen"))
                table.insert(0, "rule", name)
                table.insert(0, "slice", slice_name)
                table.insert(0, "split", split)
                table.insert(0, "seed", seed)
                table.insert(0, "experiment", experiment)
                rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def class_confusion_checks(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """What the boxes of one class in one slice are predicted as, per check in ``summary.confusion_checks``.

    Each row gives, per experiment and seed, the number of boxes and tows and
    the predicted class counts at the operating thresholds and at the score
    floor.
    """
    from .detection.evaluate import image_table, load_coco, predicted_as, year_slices

    checks = cfg.get("summary", {}).get("confusion_checks", [])
    ecfg = cfg.get("evaluation", {})
    floor = float(cfg.get("inference", {}).get("score_floor", 0.05))
    rows = []
    for run_dir in sorted(root.glob("*/seed*")):
        experiment, seed = run_dir.parent.name, run_dir.name
        if experiment.startswith("smoke"):
            continue
        for check in checks:
            split, slice_name, class_name = str(check["split"]), str(check["slice"]), str(check["class"])
            pred_path = run_dir / f"predictions_{split}.parquet"
            try:
                coco_path = (
                    _paths(cfg, _experiment(cfg, experiment), int(seed.replace("seed", "")))["coco_dir"]
                    / f"{split}.json"
                )
            except KeyError:
                continue
            if not (pred_path.is_file() and coco_path.is_file()):
                continue
            coco = load_coco(coco_path)
            images = image_table(coco)
            names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
            ids_by_name = {v: k for k, v in names.items()}
            if class_name not in ids_by_name:
                continue
            slices = year_slices(images)
            if slice_name not in slices:
                continue
            from .detection.evaluate import gt_table

            gts = gt_table(coco)
            preds = pd.read_parquet(pred_path)
            preds["score"] = _apply_calibration(
                run_dir / "calibration" / "calibration.json", preds["score"].to_numpy()
            )
            image_ids = slices[slice_name]
            class_id = ids_by_name[class_name]
            sel = gts[(gts["category_id"] == class_id) & gts["image_id"].isin(image_ids)]
            tows = (
                images.set_index("image_id").loc[sel["image_id"].unique(), "tow"].nunique() if len(sel) else 0
            )
            at_operating = predicted_as(
                preds,
                gts,
                class_id,
                image_ids,
                names,
                _operating_thresholds(run_dir),
                float(ecfg.get("iou_threshold", 0.5)),
            )
            at_floor = predicted_as(
                preds,
                gts,
                class_id,
                image_ids,
                names,
                {c: floor for c in names},
                float(ecfg.get("iou_threshold", 0.5)),
            )
            for regime, counts in (("operating", at_operating), ("floor", at_floor)):
                rows.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "split": split,
                        "slice": slice_name,
                        "class_name": class_name,
                        "n_boxes": int(len(sel)),
                        "n_tows": int(tows),
                        "thresholds": regime,
                        **{
                            f"as_{k}": v
                            for k, v in counts.items()
                            if v > 0 or k in (class_name, "background")
                        },
                    }
                )
    return pd.DataFrame(rows)


def compare_experiments(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    """Paired bootstrap differences between the experiments listed under ``summary.compare`` (same seed).

    Both models are evaluated on the same images with the same resamples, so
    the interval of the difference accounts for the shared test set.
    """
    from .detection.evaluate import (
        bootstrap_ap_by_slice,
        matched_predictions,
        operating_point_by_slice,
        paired_difference,
        year_slices,
    )

    pairs = cfg.get("summary", {}).get("compare", [])
    ecfg = cfg.get("evaluation", {})
    rows = []
    for pair in pairs:
        exp_a, exp_b = str(pair[0]), str(pair[1])
        for seed_dir in sorted(root.glob(f"{exp_a}/seed*")):
            seed = seed_dir.name
            dir_b = root / exp_b / seed
            if not dir_b.is_dir():
                continue
            for split in ("val", "test", "test_2024"):
                pred_a = seed_dir / f"predictions_{split}.parquet"
                pred_b = dir_b / f"predictions_{split}.parquet"
                coco_path = (
                    _paths(cfg, _experiment(cfg, exp_a), int(seed.replace("seed", "")))["coco_dir"]
                    / f"{split}.json"
                )
                if not (pred_a.is_file() and pred_b.is_file() and coco_path.is_file()):
                    continue
                matched_a, gts, images, class_ids, names = matched_predictions(
                    coco_path, pd.read_parquet(pred_a), float(ecfg.get("iou_threshold", 0.5))
                )
                matched_b, _, _, _, _ = matched_predictions(
                    coco_path, pd.read_parquet(pred_b), float(ecfg.get("iou_threshold", 0.5))
                )
                ids_by_name = {v: k for k, v in names.items()}
                sets = {
                    name: [ids_by_name[n] for n in members if n in ids_by_name]
                    for name, members in (ecfg.get("class_sets") or {}).items()
                }
                slices = year_slices(images)
                common = dict(
                    n_boot=int(ecfg.get("n_boot", 200)),
                    seed=int(cfg.get("seed", 0)),
                    min_gt=int(ecfg.get("min_gt_for_macro", 10)),
                    class_sets=sets,
                    return_draws=True,
                )
                _, draws_a = bootstrap_ap_by_slice(matched_a, gts, class_ids, names, slices, **common)
                _, draws_b = bootstrap_ap_by_slice(matched_b, gts, class_ids, names, slices, **common)
                # Recall and precision at each model's own operating thresholds, on calibrated scores.
                for matched, draws, run_dir in ((matched_a, draws_a, seed_dir), (matched_b, draws_b, dir_b)):
                    thr = _operating_thresholds(run_dir)
                    matched = matched.copy()
                    matched["score_calibrated"] = _apply_calibration(
                        run_dir / "calibration" / "calibration.json", matched["score"].to_numpy()
                    )
                    _, op_draws = operating_point_by_slice(
                        matched,
                        gts,
                        class_ids,
                        names,
                        slices,
                        thr,
                        common["n_boot"],
                        common["seed"],
                        score_col="score_calibrated",
                    )
                    for slice_name, metrics in op_draws.items():
                        draws.setdefault(slice_name, {}).update(metrics)
                diff = paired_difference(draws_a, draws_b)
                diff.insert(0, "split", split)
                diff.insert(0, "seed", seed)
                diff.insert(0, "experiment_b", exp_b)
                diff.insert(0, "experiment_a", exp_a)
                rows.append(diff)
                log.info("compared %s and %s on %s %s", exp_a, exp_b, seed, split)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_summarize_detectors(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Collect training, evaluation and calibration summaries of every experiment and seed into tables."""
    project = resolve_path(cfg, "project_dir")
    root = Path(project, "runs", "detector")
    target = Path(project, "runs", "detector_summary")
    target.mkdir(parents=True, exist_ok=True)
    overview_rows, slice_rows, class_rows, calib_rows = [], [], [], []
    for summary_path in sorted(root.glob("*/seed*/summary.json")):
        run_dir = summary_path.parent
        experiment, seed = run_dir.parent.name, run_dir.name
        with open(summary_path, encoding="utf-8") as fh:
            train = json.load(fh)
        row: dict[str, Any] = {
            "experiment": experiment,
            "seed": seed,
            "framework": train.get("framework"),
            "input_size": train.get("input_size"),
            "epochs_run": train.get("epochs_run"),
            "best_epoch": train.get("best_epoch"),
            "best_val_map": train.get("best_val_map"),
        }
        for split in ("val", "test", "test_2024"):
            eval_summary = run_dir / f"eval_{split}" / "summary.json"
            if not eval_summary.is_file():
                continue
            with open(eval_summary, encoding="utf-8") as fh:
                ev = json.load(fh)
            for k in ("map", "map50", "map75", "map_small", "map_medium", "map_large"):
                row[f"{split}_{k}"] = ev.get("coco", {}).get(k)
            for slice_name, v in ev.get("map50_by_slice", {}).items():
                slice_rows.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "split": split,
                        "slice": slice_name,
                        "metric": "map50",
                        **{k: v[k] for k in ("value", "ci_low", "ci_high", "n_images")},
                    }
                )
            slices_csv = run_dir / f"eval_{split}" / "slices.csv"
            if slices_csv.is_file():
                s = pd.read_csv(slices_csv)
                s = s[s["metric"].str.startswith("ap50_")]
                for r in s.itertuples(index=False):
                    class_rows.append(
                        {
                            "experiment": experiment,
                            "seed": seed,
                            "split": split,
                            "slice": r.slice,
                            "class_name": r.metric.replace("ap50_", ""),
                            "ap50": r.value,
                            "ci_low": r.ci_low,
                            "ci_high": r.ci_high,
                        }
                    )
        calib = run_dir / "calibration" / "calibration.json"
        if calib.is_file():
            with open(calib, encoding="utf-8") as fh:
                c = json.load(fh)
            for key, values in c.items():
                if isinstance(values, dict) and key.startswith(("ece", "nll", "cross_fitted")):
                    for name, v in values.items():
                        calib_rows.append(
                            {
                                "experiment": experiment,
                                "seed": seed,
                                "quantity": key,
                                "calibrator": name,
                                "value": v,
                            }
                        )
            row["calibration_chosen"] = c.get("chosen")
        overview_rows.append(row)
    pd.DataFrame(overview_rows).to_csv(target / "overview.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(target / "map50_by_slice.csv", index=False)
    pd.DataFrame(class_rows).to_csv(target / "ap50_by_class.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(target / "calibration.csv", index=False)
    differences = compare_experiments(cfg, root)
    if not differences.empty:
        differences.to_csv(target / "differences.csv", index=False)
    confusion = class_confusion_checks(cfg, root)
    if not confusion.empty:
        confusion.to_csv(target / "class_confusion.csv", index=False)
    counts, pairs = resolution_checks(cfg, root)
    if not counts.empty:
        counts.to_csv(target / "resolution_counts.csv", index=False)
        pairs.to_csv(target / "resolution_pairs.csv", index=False)
    rules = object_rule_reports(cfg, root)
    if not rules.empty:
        rules.to_csv(target / "object_rules.csv", index=False)
    log.info("summarised %d runs into %s", len(overview_rows), target)
    return target
