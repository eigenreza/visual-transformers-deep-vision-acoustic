"""Reference convolutional detector: torchvision RetinaNet with a ResNet-50 FPN backbone.

The model starts from COCO weights, keeps the box regression head and
receives a new classification head for the project's classes. Training uses
the same augmentations as the transformer detectors (photometric distortion,
IoU crop, horizontal flip), mixed precision, AdamW with linear warm-up and
cosine decay, and keeps the checkpoint with the best validation mAP. Boxes
are predicted in pixels of the original image.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import v2 as T

from .evaluate import coco_metrics, load_coco

log = logging.getLogger(__name__)


class CocoBoxes(Dataset):
    """COCO images with boxes as tv_tensors, optionally augmented."""

    def __init__(self, coco: dict[str, Any], images_root: Path, transforms: T.Transform | None) -> None:
        self.images = coco["images"]
        self.root = Path(images_root)
        self.transforms = transforms
        by_image: dict[int, list[dict[str, Any]]] = {}
        for a in coco["annotations"]:
            by_image.setdefault(int(a["image_id"]), []).append(a)
        self.by_image = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        info = self.images[index]
        with Image.open(self.root / info["file_name"]) as im:
            image = tv_tensors.Image(T.functional.pil_to_tensor(im.convert("RGB")))
        anns = self.by_image.get(int(info["id"]), [])
        boxes = torch.tensor(
            [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        labels = torch.tensor([int(a["category_id"]) for a in anns], dtype=torch.int64)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=image.shape[-2:]),
            "labels": labels,
            "image_id": int(info["id"]),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        image = T.functional.to_dtype(image, torch.float32, scale=True)
        return image, {
            "boxes": target["boxes"].as_subclass(torch.Tensor),
            "labels": target["labels"],
            "image_id": target["image_id"],
        }


def train_transforms(aug: dict[str, Any]) -> T.Transform:
    ops: list[T.Transform] = []
    if float(aug.get("photometric_p", 0.5)) > 0:
        ops.append(T.RandomPhotometricDistort(p=float(aug.get("photometric_p", 0.5))))
    if bool(aug.get("zoom_out", False)):
        ops.append(T.RandomZoomOut(fill=0))
    iou_crop_p = float(aug.get("iou_crop_p", 0.5))
    if iou_crop_p > 0:
        ops.append(T.RandomApply([T.RandomIoUCrop()], p=iou_crop_p))
    if bool(aug.get("hflip", True)):
        ops.append(T.RandomHorizontalFlip())
    ops.append(T.SanitizeBoundingBoxes(min_size=1))
    return T.Compose(ops)


def collate(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    return [b[0] for b in batch], [b[1] for b in batch]


def build_model(
    num_classes: int, input_size: int, pretrained: bool = True, score_threshold: float = 0.001
) -> torch.nn.Module:
    """RetinaNet with COCO-pretrained backbone, FPN and regression head, and a fresh classification head."""
    weights = "COCO_V1" if pretrained else None
    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        min_size=input_size,
        max_size=int(round(input_size * 1.25)),
        score_thresh=score_threshold,
        detections_per_img=300,
    )
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(256, num_anchors, num_classes)
    return model


def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    paths: list[Path],
    device: str,
    batch_size: int = 8,
    num_workers: int = 8,
    use_amp: bool = True,
) -> pd.DataFrame:
    """Detections for a list of images, with ``image_index`` referring to the position in the list."""

    class Plain(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, i: int) -> tuple[torch.Tensor, int, tuple[int, int]]:
            with Image.open(paths[i]) as im:
                size = im.size
                t = T.functional.pil_to_tensor(im.convert("RGB"))
            return T.functional.to_dtype(t, torch.float32, scale=True), i, size

    loader = DataLoader(
        Plain(),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b], [x[2] for x in b]),
    )
    model.eval()
    parts = []
    for images, indices, sizes in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")
        ):
            outputs = model(images)
        for out, idx, size in zip(outputs, indices, sizes, strict=True):
            if out["boxes"].numel() == 0:
                continue
            b = out["boxes"].float().cpu().numpy()
            parts.append(
                pd.DataFrame(
                    {
                        "image_index": idx,
                        "width": float(size[0]),
                        "height": float(size[1]),
                        "category_id": out["labels"].cpu().numpy().astype(int),
                        "score": out["scores"].float().cpu().numpy(),
                        "x0": b[:, 0],
                        "y0": b[:, 1],
                        "x1": b[:, 2],
                        "y1": b[:, 3],
                    }
                )
            )
    columns = ["image_index", "width", "height", "category_id", "score", "x0", "y0", "x1", "y1"]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


def predict_coco(
    model: torch.nn.Module, coco: dict[str, Any], images_root: Path, device: str, **kwargs: Any
) -> pd.DataFrame:
    paths = [Path(images_root, im["file_name"]) for im in coco["images"]]
    ids = [int(im["id"]) for im in coco["images"]]
    preds = predict(model, paths, device, **kwargs)
    preds.insert(0, "image_id", preds["image_index"].map(dict(enumerate(ids))).astype(int))
    return preds.drop(columns=["image_index"])


def train(
    train_json: Path,
    val_json: Path,
    images_root: Path,
    output_dir: Path,
    num_classes: int,
    input_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_iterations: int,
    aug: dict[str, Any],
    seed: int,
    device: str,
    num_workers: int = 8,
    pretrained: bool = True,
    resume: bool = False,
    eval_batch_size: int = 8,
) -> dict[str, Any]:
    """Train RetinaNet; keep ``best.pth`` by validation mAP and ``last.pth`` for resuming."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_coco, val_coco = load_coco(train_json), load_coco(val_json)
    dataset = CocoBoxes(train_coco, images_root, train_transforms(aug))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    model = build_model(num_classes, input_size, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    total_steps = epochs * len(loader)
    start_epoch, step, best_map = 0, 0, -1.0
    last = output_dir / "last.pth"
    if resume and last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best_map = state["epoch"] + 1, state["step"], state["best_map"]
        log.info("resumed from epoch %d", state["epoch"])
    log_path = output_dir / "log.txt"
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        losses = []
        for images, targets in loader:
            for g in optimizer.param_groups:
                g["lr"] = _lr_at(step, total_steps, warmup_iterations, lr)
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            step += 1
        preds = predict_coco(
            model, val_coco, images_root, device, batch_size=eval_batch_size, num_workers=num_workers
        )
        stats = coco_metrics(val_json, preds)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_map": stats.get("map", 0.0),
            "val_map50": stats.get("map50", 0.0),
            "val_map75": stats.get("map75", 0.0),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(record)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info(
            "epoch %d: loss %.3f val mAP %.4f mAP50 %.4f (%.0f s)",
            epoch,
            record["train_loss"],
            record["val_map"],
            record["val_map50"],
            record["seconds"],
        )
        if record["val_map"] > best_map:
            best_map = record["val_map"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_map": best_map}, output_dir / "best.pth"
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "best_map": best_map,
            },
            last,
        )
    return {"best_val_map": best_map, "epochs_run": len(history), "history": history}


def load_trained(
    checkpoint: Path, num_classes: int, input_size: int, device: str, score_threshold: float = 0.001
) -> torch.nn.Module:
    model = build_model(num_classes, input_size, pretrained=False, score_threshold=score_threshold)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()
