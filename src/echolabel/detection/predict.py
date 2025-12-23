"""Inference with trained RT-DETRv2 and D-FINE models.

A model is rebuilt from the framework configuration written at training time
and its best checkpoint, switched to deployment mode, and run on a list of
images. Every detection is returned with its class identifier, score and
corner coordinates in pixels of the original image. Only one framework can be
loaded per process because both repositories expose a top-level package named
``src``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class ImageListDataset(Dataset):
    """Images resized to a square tensor in [0, 1], with their original sizes."""

    def __init__(self, paths: list[Path], input_size: int) -> None:
        self.paths = paths
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        with Image.open(self.paths[index]) as im:
            im = im.convert("RGB")
            w, h = im.size
            im = im.resize((self.input_size, self.input_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor, torch.tensor([w, h], dtype=torch.float32), index


def load_detr_model(repo_dir: Path, config_path: Path, checkpoint: Path, device: str) -> tuple[Any, Any]:
    """Model in deployment mode and its post-processor, built from the framework code in ``repo_dir``."""
    repo_dir = Path(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from src.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(str(config_path))
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(weights)
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy()
    return model, postprocessor


@torch.no_grad()
def predict_images(
    model: Any,
    postprocessor: Any,
    paths: list[Path],
    input_size: int,
    device: str,
    batch_size: int = 16,
    num_workers: int = 8,
    score_threshold: float = 0.001,
    use_amp: bool = True,
) -> pd.DataFrame:
    """Detections for every image in ``paths``; ``image_index`` refers to the position in that list."""
    dataset = ImageListDataset(paths, input_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
    )
    parts: list[pd.DataFrame] = []
    for images, sizes, indices in loader:
        images = images.to(device, non_blocking=True)
        sizes = sizes.to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")
        ):
            outputs = model(images)
        labels, boxes, scores = postprocessor(outputs, sizes)
        for k in range(images.shape[0]):
            s = scores[k].float().cpu().numpy()
            keep = s >= score_threshold
            if not keep.any():
                continue
            b = boxes[k].float().cpu().numpy()[keep]
            parts.append(
                pd.DataFrame(
                    {
                        "image_index": int(indices[k]),
                        "width": float(sizes[k, 0]),
                        "height": float(sizes[k, 1]),
                        "category_id": labels[k].cpu().numpy()[keep].astype(int),
                        "score": s[keep],
                        "x0": b[:, 0],
                        "y0": b[:, 1],
                        "x1": b[:, 2],
                        "y1": b[:, 3],
                    }
                )
            )
    columns = ["image_index", "width", "height", "category_id", "score", "x0", "y0", "x1", "y1"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


def predict_coco_split(
    model: Any,
    postprocessor: Any,
    coco: dict[str, Any],
    images_root: Path,
    input_size: int,
    device: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Detections for every image of a COCO file, keyed by the COCO image identifier."""
    paths = [Path(images_root, im["file_name"]) for im in coco["images"]]
    ids = [int(im["id"]) for im in coco["images"]]
    preds = predict_images(model, postprocessor, paths, input_size, device, **kwargs)
    preds.insert(0, "image_id", preds["image_index"].map(dict(enumerate(ids))).astype(int))
    return preds.drop(columns=["image_index"])
