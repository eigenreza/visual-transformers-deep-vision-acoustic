"""Training loop for the acoustic models.

One configuration trains in minutes: AdamW, linear warm-up and cosine decay,
gradient clipping, mixed precision on the GPU, early stopping on a count-space
metric of the validation windows (mean absolute error of the predictive mean,
macro over classes), a checkpoint every epoch and resumption from it. Targets
can be sampled over the stored lag offsets every epoch as augmentation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .counts import nll, predictive_mean, split_params

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    clip: float = 1.0
    patience: int = 10
    amp: bool = True
    tweedie_power: float = 1.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    """Per-window sampling weights: uniform, presence balanced, or growing with the target magnitude."""
    if mode == "uniform":
        return None
    total = y.sum(axis=1)
    if mode == "presence_balanced":
        present = total > 0
        w = np.where(present, 0.5 / max(present.sum(), 1), 0.5 / max((~present).sum(), 1))
        return w / w.sum()
    if mode == "magnitude":
        w = 1.0 + np.log1p(total)
        return w / w.sum()
    raise KeyError(mode)


def _forward(net: nn.Module, xb: torch.Tensor, cb: torch.Tensor | None) -> torch.Tensor:
    return net(xb) if cb is None else net(xb, cb)


@torch.no_grad()
def predict_params(
    net: nn.Module,
    family: str,
    x: np.ndarray,
    n_classes: int,
    device: str,
    batch_size: int = 256,
    context: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Distribution parameters per window and class, as numpy arrays; ``context`` feeds a two-scale net."""
    net.eval()
    parts: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        cb = (
            torch.as_tensor(context[i : i + batch_size], dtype=torch.float32, device=device)
            if context is not None
            else None
        )
        params = split_params(family, _forward(net, xb, cb).float(), n_classes)
        for k, v in params.items():
            parts.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def validation_metric(
    net: nn.Module,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    context: np.ndarray | None = None,
) -> float:
    """Mean absolute error of the predictive mean, macro over classes, in count space."""
    params = predict_params(net, family, x, y.shape[1], device, context=context)
    mean = predictive_mean(family, params)
    return float(np.mean(np.abs(mean - y).mean(axis=0)))


def train_model(
    net: nn.Module,
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray | np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: str,
    seed: int,
    checkpoint_dir: Path,
    y_train_offsets: np.ndarray | None = None,
    offset_weights: np.ndarray | None = None,
    sampler: str = "uniform",
    resume: bool = False,
    c_train: np.ndarray | None = None,
    c_val: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train ``net`` and return the history, the best epoch and the best validation metric.

    ``y_train_offsets`` of shape (N, K, O) enables lag sampling: every epoch
    each window draws one offset with ``offset_weights``. ``c_train`` and
    ``c_val`` are the context arrays of a two-scale network. The best weights
    by validation metric are kept on the network at the end.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(device)
    n_classes = y_val.shape[1]
    steps_per_epoch = max(1, math.ceil(len(x_train) / cfg.batch_size))
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg.epochs * steps_per_epoch, cfg.warmup_steps)
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = checkpoint_dir / "last.pt"
    best = checkpoint_dir / "best.pt"
    history: list[dict[str, float]] = []
    start_epoch, best_metric, best_epoch, bad_epochs = 0, float("inf"), -1, 0
    if resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_metric, best_epoch, history = (
            state["epoch"] + 1,
            state["best_metric"],
            state["best_epoch"],
            state["history"],
        )
        rng = np.random.default_rng(seed + start_epoch)
        log.info("resumed at epoch %d", start_epoch)
    weights = sample_weights(y_train, sampler)
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        t0 = time.time()
        if weights is None:
            order = rng.permutation(len(x_train))
        else:
            order = rng.choice(len(x_train), size=len(x_train), replace=True, p=weights)
        if y_train_offsets is not None:
            pick = rng.choice(y_train_offsets.shape[2], size=len(x_train), p=offset_weights)
            y_epoch = y_train_offsets[np.arange(len(x_train)), :, pick]
        else:
            y_epoch = y_train
        losses = []
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_epoch[idx], dtype=torch.float32, device=device)
            cb = (
                torch.as_tensor(c_train[idx], dtype=torch.float32, device=device)
                if c_train is not None
                else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = _forward(net, xb, cb)
            params = split_params(family, out.float(), n_classes)
            loss = nll(family, params, yb, cfg.tweedie_power).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.item()))
        metric = validation_metric(net, family, x_val, y_val, device, c_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mae": metric,
                "seconds": time.time() - t0,
            }
        )
        improved = metric < best_metric - 1e-6
        if improved:
            best_metric, best_epoch, bad_epochs = metric, epoch, 0
            torch.save({"model": net.state_dict(), "epoch": epoch, "val_mae": metric}, best)
        else:
            bad_epochs += 1
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "history": history,
            },
            last,
        )
        log.info(
            "epoch %d: loss %.4f, val mae %.4f%s",
            epoch,
            history[-1]["train_loss"],
            metric,
            " *" if improved else "",
        )
        if bad_epochs >= cfg.patience:
            log.info("early stop at epoch %d", epoch)
            break
    if best.is_file():
        net.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
    return {"history": history, "best_epoch": best_epoch, "best_val_mae": best_metric}


def load_pretrained_encoder(net: nn.Module, path: Path) -> int:
    """Copy matching encoder weights from a masked autoencoder checkpoint; returns the tensor count."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("encoder", state)
    own = net.state_dict()
    copied = 0
    for k, v in weights.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            copied += 1
    net.load_state_dict(own)
    return copied
