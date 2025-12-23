"""Configuration and command generation for the external DETR frameworks.

RT-DETRv2 and D-FINE share one configuration system: a YAML file with an
``__include__`` list whose keys override the included files. The functions
here write such a file for one experiment, pointing at the framework's own
model definition and at the COCO files of this project, and build the
training and resume commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FRAMEWORKS: dict[str, dict[str, Any]] = {
    "rtdetrv2": {
        "train_script": "tools/train.py",
        "epochs_key": "epoches",
        "backbone_key": "PResNet",
        "backbone_lr_regex": "^(?=.*backbone)(?!.*norm).*$",
    },
    "dfine": {
        "train_script": "train.py",
        "epochs_key": "epochs",
        "backbone_key": "HGNetv2",
        "backbone_lr_regex": "^(?=.*backbone)(?!.*norm|bn).*$",
    },
}

NORM_REGEX = "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$"


def multiscale_sizes(input_size: int, repeat: int = 3) -> list[int]:
    """Training scales from 75 to 125 percent of the input size in steps of 32, as used by the frameworks."""
    low = int(input_size * 0.75 / 32) * 32
    high = int(input_size * 1.25 / 32) * 32
    n = (input_size - low) // 32
    scales = [low + 32 * i for i in range(n)]
    scales += [input_size] * repeat
    scales += [high - 32 * i for i in reversed(range(n))]
    return scales


def train_transforms(input_size: int, aug: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Training transform list and the names of the strong augmentations switched off near the end."""
    ops: list[dict[str, Any]] = []
    strong: list[str] = []
    photometric_p = float(aug.get("photometric_p", 0.5))
    if photometric_p > 0:
        ops.append({"type": "RandomPhotometricDistort", "p": photometric_p})
        strong.append("RandomPhotometricDistort")
    if bool(aug.get("zoom_out", False)):
        ops.append({"type": "RandomZoomOut", "fill": 0})
        strong.append("RandomZoomOut")
    iou_crop_p = float(aug.get("iou_crop_p", 0.5))
    if iou_crop_p > 0:
        ops.append({"type": "RandomIoUCrop", "p": iou_crop_p})
        strong.append("RandomIoUCrop")
    ops.append({"type": "SanitizeBoundingBoxes", "min_size": 1})
    if bool(aug.get("hflip", True)):
        ops.append({"type": "RandomHorizontalFlip"})
    ops.append({"type": "Resize", "size": [input_size, input_size]})
    ops.append({"type": "SanitizeBoundingBoxes", "min_size": 1})
    ops.append({"type": "ConvertPILImage", "dtype": "float32", "scale": True})
    ops.append({"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True})
    return ops, strong


def build_framework_config(
    framework: str,
    base_config: Path,
    output_dir: Path,
    images_root: Path,
    train_json: Path,
    val_json: Path,
    num_classes: int,
    input_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    warmup_iterations: int,
    checkpoint_freq: int,
    num_workers: int,
    aug: dict[str, Any],
    stop_augmentation_epochs_before_end: int = 8,
    pretrained_backbone: bool = False,
) -> dict[str, Any]:
    """Dictionary for one experiment's framework configuration file."""
    spec = FRAMEWORKS[framework]
    ops, strong = train_transforms(input_size, aug)
    stop_epoch = max(1, epochs - int(stop_augmentation_epochs_before_end))
    collate: dict[str, Any] = {"type": "BatchImageCollateFunction", "stop_epoch": stop_epoch}
    if bool(aug.get("multiscale", True)):
        if framework == "rtdetrv2":
            collate["scales"] = multiscale_sizes(input_size)
        else:
            collate["base_size"] = input_size
            collate["base_size_repeat"] = 3
    elif framework == "dfine":
        collate["base_size"] = input_size
    if framework == "dfine":
        collate["ema_restart_decay"] = 0.9999
    cfg: dict[str, Any] = {
        "__include__": [Path(base_config).as_posix()],
        "output_dir": Path(output_dir).as_posix(),
        "num_classes": int(num_classes),
        "remap_mscoco_category": False,
        "eval_spatial_size": [input_size, input_size],
        "train_dataloader": {
            "dataset": {
                "img_folder": Path(images_root).as_posix(),
                "ann_file": Path(train_json).as_posix(),
                "transforms": {
                    "ops": ops,
                    "policy": {"name": "stop_epoch", "epoch": stop_epoch, "ops": strong},
                },
            },
            "collate_fn": collate,
            "total_batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "shuffle": True,
            "drop_last": True,
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": Path(images_root).as_posix(),
                "ann_file": Path(val_json).as_posix(),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [input_size, input_size]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                },
            },
            "total_batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "shuffle": False,
            "drop_last": False,
        },
        spec["epochs_key"]: int(epochs),
        "checkpoint_freq": int(checkpoint_freq),
        "print_freq": 50,
        "use_amp": True,
        "use_ema": True,
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": spec["backbone_lr_regex"], "lr": float(backbone_lr)},
                {"params": NORM_REGEX, "weight_decay": 0.0},
            ],
            "lr": float(lr),
            "betas": [0.9, 0.999],
            "weight_decay": float(weight_decay),
        },
        "lr_scheduler": {"type": "MultiStepLR", "milestones": [1_000_000], "gamma": 0.1},
        "lr_warmup_scheduler": {"type": "LinearWarmup", "warmup_duration": int(warmup_iterations)},
        spec["backbone_key"]: {"pretrained": bool(pretrained_backbone)},
    }
    return cfg


def write_framework_config(cfg: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def train_command(
    framework: str,
    config_path: Path,
    output_dir: Path,
    seed: int,
    pretrained: Path | None = None,
    resume: Path | None = None,
    python: str = "python",
) -> list[str]:
    """Command line for one training run; run it with the framework repository as working directory."""
    cmd = [
        python,
        FRAMEWORKS[framework]["train_script"],
        "-c",
        str(config_path),
        "--use-amp",
        "--seed",
        str(seed),
    ]
    cmd += ["--output-dir", str(output_dir)]
    if resume is not None:
        cmd += ["-r", str(resume)]
    elif pretrained is not None:
        cmd += ["-t", str(pretrained)]
    return cmd


def best_checkpoint(framework: str, output_dir: Path) -> Path | None:
    """Path of the best checkpoint a finished run left behind, if any."""
    candidates = ["best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth"]
    for name in candidates:
        p = output_dir / name
        if p.is_file():
            return p
    return None
