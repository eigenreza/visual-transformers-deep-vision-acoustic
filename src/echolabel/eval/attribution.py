"""Attribution maps for the acoustic networks.

For convolutional networks: integrated gradients from a floor baseline and
Grad-CAM on the last convolution block. For the transformer: attention
rollout over the encoder layers and the rollout weighted by the gradient of
the prediction with respect to the tokens. Every map is returned on the cell
grid of the input (T by D), and a helper measures how much of its mass lies
on the depth band where the camera was.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ..models.nets import DepthAwareViT, SmallCNN


def floor_baseline(x: torch.Tensor, n_db_channels: int, floor_value: float) -> torch.Tensor:
    """Baseline input: dB and difference channels at ``floor_value``, valid channels zero, depth unchanged."""
    base = x.clone()
    base[:, :n_db_channels] = floor_value
    base[:, n_db_channels:-1] = 0.0
    return base


def integrated_gradients(
    net: nn.Module,
    x: torch.Tensor,
    target: Callable[[torch.Tensor], torch.Tensor],
    baseline: torch.Tensor,
    steps: int = 32,
) -> np.ndarray:
    """Integrated gradients of ``target(net(x))`` summed over channels: (N, T, D)."""
    net.eval()
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        point = (baseline + (x - baseline) * k / steps).detach().requires_grad_(True)
        out = target(net(point)).sum()
        (grad,) = torch.autograd.grad(out, point)
        total += grad
    attribution = (x - baseline) * total / steps
    return attribution.sum(dim=1).detach().cpu().numpy()


def grad_cam(net: SmallCNN, x: torch.Tensor, target: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
    """Grad-CAM on the last convolution block, upsampled to the input grid: (N, T, D)."""
    net.eval()
    store: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        out.retain_grad()
        store["features"] = out

    handle = net.features[-1].register_forward_hook(hook)
    try:
        out = target(net(x)).sum()
        out.backward()
    finally:
        handle.remove()
    features, grads = store["features"], store["features"].grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().cpu().numpy()


def _attention_maps(vit: DepthAwareViT, x: torch.Tensor) -> list[torch.Tensor]:
    """Head-averaged attention of every encoder layer, class token included: list of (N, L+1, L+1)."""
    maps: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    handles = []
    for layer in vit.encoder.layers:
        handles.append(
            layer.self_attn.register_forward_pre_hook(lambda _m, args, inputs=inputs: inputs.append(args[0]))
        )
    try:
        vit.encode(x)
    finally:
        for h in handles:
            h.remove()
    for layer, tokens in zip(vit.encoder.layers, inputs, strict=True):
        attn = layer.self_attn
        qkv = nn.functional.linear(tokens, attn.in_proj_weight, attn.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)
        n, length, dim = q.shape
        heads = attn.num_heads
        q = q.view(n, length, heads, dim // heads).transpose(1, 2)
        k = k.view(n, length, heads, dim // heads).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(dim // heads)
        maps.append(scores.softmax(dim=-1).mean(dim=1))
    return maps


def attention_rollout(vit: DepthAwareViT, x: torch.Tensor) -> np.ndarray:
    """Rollout of the class token's attention over all layers, on the token grid upsampled to (N, T, D)."""
    maps = _attention_maps(vit, x)
    n, length, _ = maps[0].shape
    eye = torch.eye(length, device=x.device)[None]
    joint = eye.expand(n, -1, -1).clone()
    for a in maps:
        a_res = 0.5 * (a + eye)
        a_res = a_res / a_res.sum(dim=-1, keepdim=True)
        joint = a_res @ joint
    cls_row = joint[:, 0, 1:]
    return _tokens_to_grid(cls_row, vit, x)


def gradient_weighted_attention(
    vit: DepthAwareViT, x: torch.Tensor, target: Callable[[torch.Tensor], torch.Tensor]
) -> np.ndarray:
    """Attention rollout weighted by the gradient norm of the target with respect to each token."""
    tokens_store: dict[str, torch.Tensor] = {}

    def hook(_m: nn.Module, _inp: tuple, out: tuple) -> tuple:
        tokens, depth = out
        tokens.retain_grad()
        tokens_store["tokens"] = tokens
        return out

    handle = vit.tokens.register_forward_hook(hook)
    try:
        out = target(vit(x)).sum()
        out.backward()
    finally:
        handle.remove()
    grad_norm = tokens_store["tokens"].grad.norm(dim=-1)  # (N, L)
    rollout = torch.as_tensor(attention_rollout(vit, x.detach()), device=x.device)
    weights = _tokens_to_grid(grad_norm, vit, x)
    return rollout.cpu().numpy() * weights


def _tokens_to_grid(values: torch.Tensor, vit: DepthAwareViT, x: torch.Tensor) -> np.ndarray:
    """Token values (N, L) laid out on the patch grid and upsampled to the input cells (N, T, D)."""
    pt, pd_ = vit.tokens.patch
    tp, dp = x.shape[2] // pt, x.shape[3] // pd_
    grid = values.reshape(values.shape[0], 1, tp, dp)
    up = nn.functional.interpolate(grid, size=x.shape[2:], mode="nearest")[:, 0]
    return up.detach().cpu().numpy()


def mass_on_band(
    attribution: np.ndarray, depth_cells: np.ndarray, camera_depth_m: np.ndarray, half_width_m: float = 5.0
) -> np.ndarray:
    """Share of absolute attribution mass within ``half_width_m`` of the camera depth, per window."""
    mass = np.abs(attribution)
    total = mass.sum(axis=(1, 2))
    band = np.abs(depth_cells - camera_depth_m[:, None]) <= half_width_m  # (N, D)
    on_band = (mass * band[:, None, :]).sum(axis=(1, 2))
    return np.where(total > 0, on_band / np.maximum(total, 1e-12), np.nan)


def class_mean_target(
    family: str, n_classes: int, class_index: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Target function: the predicted mean count of one class from the network output."""
    from ..models.counts import split_params

    def target(out: torch.Tensor) -> torch.Tensor:
        params = split_params(family, out, n_classes)
        if family == "log1p":
            return torch.expm1(params["log1p_mu"][:, class_index])
        mean = params["log_mu"][:, class_index].exp()
        if family == "zinb":
            mean = mean * torch.sigmoid(-params["logit_pi"][:, class_index])
        return mean

    return target
