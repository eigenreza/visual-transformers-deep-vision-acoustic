"""Networks for the acoustic windows.

All networks take an input of shape (N, C, T, D) with time along T and
depth along D, and return (N, n_out). The transformer and the hybrid add a
depth-aware positional encoding: each token receives Fourier features of its
absolute depth in metres, taken from the depth channel, in addition to a
learned encoding of its position in time. The masked autoencoder reuses the
transformer encoder for self-supervised pretraining on cruise-wide patches.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def fourier_features(x: torch.Tensor, n_features: int, max_period: float = 1000.0) -> torch.Tensor:
    """Sine and cosine features of ``x`` (any shape) at ``n_features // 2`` geometric frequencies."""
    half = n_features // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=x.dtype) / max(half, 1)
    )
    angles = x[..., None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU()]
        if pool:
            layers.append(nn.MaxPool2d(2, ceil_mode=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    """Four convolution blocks with pooling, global average and maximum pooling, and a two-layer head."""

    def __init__(
        self, in_channels: int, n_out: int, width: int = 32, depth: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        channels = [in_channels] + [width * 2**i for i in range(depth)]
        self.features = nn.Sequential(*[ConvBlock(channels[i], channels[i + 1]) for i in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(2 * channels[-1], channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)


class TwoScaleCNN(nn.Module):
    """The small CNN on the local band with a second branch on the coarse full-column context.

    Both branches end in global average and maximum pooling; the pooled
    features are concatenated before the head, so the context informs the
    prediction without being mixed into the local convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        n_out: int,
        width: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        context_width: int = 16,
        context_depth: int = 3,
    ) -> None:
        super().__init__()
        channels = [in_channels] + [width * 2**i for i in range(depth)]
        self.features = nn.Sequential(*[ConvBlock(channels[i], channels[i + 1]) for i in range(depth)])
        c_channels = [in_channels] + [context_width * 2**i for i in range(context_depth)]
        self.context = nn.Sequential(
            *[ConvBlock(c_channels[i], c_channels[i + 1]) for i in range(context_depth)]
        )
        joint = 2 * channels[-1] + 2 * c_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(joint, channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_out),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        c = self.context(context)
        pooled = torch.cat(
            [h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), c.mean(dim=(2, 3)), c.amax(dim=(2, 3))], dim=1
        )
        return self.head(pooled)


class DepthAwareTokens(nn.Module):
    """Patch embedding of (N, C, T, D) into tokens with time and depth encodings.

    The depth channel is expected last in ``C`` and holds depth in hundreds
    of metres; it is averaged per patch, converted to metres and encoded
    with Fourier features, which are added to the token after a linear map.
    """

    def __init__(
        self, in_channels: int, dim: int, patch: tuple[int, int], max_time_patches: int = 256
    ) -> None:
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.time_pos = nn.Parameter(torch.zeros(1, max_time_patches, 1, dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.depth_proj = nn.Linear(32, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.embed(x)  # (N, dim, T', D')
        n, dim, tp, dp = tokens.shape
        depth = (
            nn.functional.avg_pool2d(x[:, -1:], self.patch, ceil_mode=True)[:, 0, :tp, :dp] * 100.0
        )  # metres
        depth_enc = self.depth_proj(fourier_features(depth / 10.0, 32))  # (N, T', D', dim)
        tokens = tokens.permute(0, 2, 3, 1) + self.time_pos[:, :tp] + depth_enc
        return tokens.reshape(n, tp * dp, dim), depth.reshape(n, tp * dp)


class DepthAwareViT(nn.Module):
    """Small vision transformer on acoustic patches with depth-aware positional encoding."""

    def __init__(
        self,
        in_channels: int,
        n_out: int,
        patch: tuple[int, int] = (3, 10),
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokens = DepthAwareTokens(in_channels, dim, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, int(dim * mlp_ratio), dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.tokens(x)
        tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        return self.norm(self.encoder(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, 0])


class ConvTransformer(nn.Module):
    """Two convolution blocks followed by the depth-aware transformer on the reduced grid."""

    def __init__(
        self,
        in_channels: int,
        n_out: int,
        width: int = 32,
        dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(in_channels - 1, width), ConvBlock(width, dim))
        self.vit = DepthAwareViT(
            dim + 1, n_out, patch=(1, 1), dim=dim, depth=depth, heads=heads, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x[:, :-1])
        depth = nn.functional.adaptive_avg_pool2d(x[:, -1:], h.shape[2:])
        return self.vit(torch.cat([h, depth], dim=1))


class MaskedAutoencoder(nn.Module):
    """Masked autoencoding on tokens of the depth-aware transformer; reconstructs the standardised cells."""

    def __init__(
        self,
        in_channels: int,
        patch: tuple[int, int] = (3, 10),
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        decoder_dim: int = 64,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.encoder = DepthAwareViT(in_channels, 1, patch, dim, depth, heads)
        self.mask_ratio = mask_ratio
        self.patch = patch
        self.n_recon = (in_channels - 1) * patch[0] * patch[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            decoder_dim, heads, decoder_dim * 2, 0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.to_decoder = nn.Linear(dim, decoder_dim)
        self.decoder = nn.TransformerEncoder(layer, 2)
        self.decoder_pos = nn.Parameter(torch.zeros(1, 4096, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        self.recon = nn.Linear(decoder_dim, self.n_recon)

    def targets(self, x: torch.Tensor) -> torch.Tensor:
        """Cells of every channel but depth, per patch: (N, L, n_recon)."""
        n, c, t, d = x.shape
        pt, pd = self.patch
        core = x[:, :-1, : (t // pt) * pt, : (d // pd) * pd]
        core = core.reshape(n, c - 1, t // pt, pt, d // pd, pd).permute(0, 2, 4, 1, 3, 5)
        return core.reshape(n, (t // pt) * (d // pd), -1)

    def forward(
        self, x: torch.Tensor, valid_weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss on the masked tokens and the reconstruction; ``valid_weight`` (N, L) weights the patches."""
        tokens, _ = self.encoder.tokens(x)
        n, length, dim = tokens.shape
        n_keep = max(1, int(round(length * (1 - self.mask_ratio))))
        noise = torch.rand(n, length, device=x.device)
        order = noise.argsort(dim=1)
        keep = order[:, :n_keep]
        kept = torch.gather(tokens, 1, keep[..., None].expand(-1, -1, dim))
        kept = torch.cat([self.encoder.cls.expand(n, -1, -1), kept], dim=1)
        encoded = self.encoder.norm(self.encoder.encoder(kept))[:, 1:]
        full = self.mask_token.expand(n, length, -1).clone()
        full = full.scatter(1, keep[..., None].expand(-1, -1, dim), encoded)
        dec = self.decoder(self.to_decoder(full) + self.decoder_pos[:, :length])
        recon = self.recon(dec)
        target = self.targets(x)
        masked = torch.ones(n, length, device=x.device, dtype=torch.bool).scatter(1, keep, False)
        per_token = ((recon - target) ** 2).mean(dim=-1)
        weight = masked.float() * (valid_weight if valid_weight is not None else 1.0)
        loss = (per_token * weight).sum() / weight.sum().clamp(min=1.0)
        return loss, recon


def build_network(name: str, in_channels: int, n_out: int, cfg: dict | None = None) -> nn.Module:
    cfg = dict(cfg or {})
    if name == "cnn":
        if cfg.pop("context", False):
            return TwoScaleCNN(in_channels, n_out, **cfg)
        return SmallCNN(in_channels, n_out, **cfg)
    if name == "vit":
        if "patch" in cfg:
            cfg["patch"] = tuple(cfg["patch"])
        return DepthAwareViT(in_channels, n_out, **cfg)
    if name == "hybrid":
        return ConvTransformer(in_channels, n_out, **cfg)
    raise KeyError(name)
