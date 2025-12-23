from __future__ import annotations

import numpy as np
import torch

from echolabel.eval.attribution import (
    attention_rollout,
    class_mean_target,
    floor_baseline,
    grad_cam,
    gradient_weighted_attention,
    integrated_gradients,
    mass_on_band,
)
from echolabel.models.nets import build_network


def _input(n: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(n, 4, 30, 50)
    x[:, 2] = 1.0  # valid channel
    x[:, 3] = torch.linspace(0.3, 0.8, 50)  # depth channel in hundreds of metres
    return x


def test_cnn_attributions() -> None:
    x = _input()
    net = build_network("cnn", 4, 4, {"width": 8, "depth": 2})
    target = class_mean_target("nb", 2, 0)
    ig = integrated_gradients(net, x, target, floor_baseline(x, 1, -2.0), steps=4)
    cam = grad_cam(net, x, target)
    assert ig.shape == (2, 30, 50) and cam.shape == (2, 30, 50) and np.isfinite(ig).all() and (cam >= 0).all()
    depth = np.tile(np.linspace(30, 80, 50), (2, 1))
    share = mass_on_band(np.abs(ig) + 1e-6, depth, np.array([50.0, 50.0]), 5.0)
    assert share.shape == (2,) and 0 <= share[0] <= 1


def test_vit_attributions() -> None:
    x = _input()
    vit = build_network("vit", 4, 4, {"patch": [3, 5], "dim": 32, "depth": 2, "heads": 2})
    roll = attention_rollout(vit, x)
    assert roll.shape == (2, 30, 50) and (roll >= 0).all() and roll.sum() > 0
    weighted = gradient_weighted_attention(vit, x, class_mean_target("nb", 2, 1))
    assert weighted.shape == (2, 30, 50) and np.isfinite(weighted).all()
