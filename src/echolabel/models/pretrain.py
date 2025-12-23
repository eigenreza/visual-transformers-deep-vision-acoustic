"""Masked autoencoding on random patches of the whole-cruise product.

Patches are cut from the downsampled cruise store (one ping by 1 m) at the
cell size of the window dataset: pairs of pings in time and 1 m in depth,
with the same channel layout as the fine-tuning inputs (selected channels in
dB with invalid cells filled, their valid fractions, and the depth channel).
Chunks of the store are loaded one at a time and many patches are cut from
each, so that the reading cost is amortised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .data import InputSpec

log = logging.getLogger(__name__)


class CruisePatchSampler:
    """Random patches from the cruise store with the fine-tuning channel layout."""

    def __init__(
        self,
        store_path: str | Path,
        spec: InputSpec,
        n_time: int,
        n_depth: int,
        depth_min_m: float = 10.0,
        depth_max_m: float = 450.0,
        pings_per_cell: int = 2,
        min_valid_fraction: float = 0.3,
        seed: int = 0,
    ) -> None:
        import xarray as xr

        self.ds = xr.open_zarr(str(store_path))
        self.freqs = [float(f) for f in self.ds["frequency"].values]
        self.pos = [int(np.argmin(np.abs(np.asarray(self.freqs) - f))) for f in spec.frequencies_hz]
        self.spec = spec
        self.n_time, self.n_depth, self.pings_per_cell = n_time, n_depth, pings_per_cell
        self.range = self.ds["range"].values.astype(float)
        self.draft = (
            float(np.nanmedian(self.ds["transducer_draft_m"].values))
            if "transducer_draft_m" in self.ds
            else 8.5
        )
        depth = self.range + self.draft
        self.d_lo = int(np.searchsorted(depth, depth_min_m))
        self.d_hi = int(np.searchsorted(depth, depth_max_m)) - n_depth
        self.min_valid = min_valid_fraction
        self.rng = np.random.default_rng(seed)
        self.chunk_len = int(self.ds["sv_db"].chunks[1][0])
        self.n_chunks = int(np.ceil(self.ds.sizes["time"] / self.chunk_len))
        self.stats: dict[str, np.ndarray] | None = None

    def _load_chunk(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0, t1 = k * self.chunk_len, min((k + 1) * self.chunk_len, self.ds.sizes["time"])
        sv = self.ds["sv_db"].isel(frequency=self.pos, time=slice(t0, t1)).values.astype(np.float32)
        valid = (
            self.ds["valid_fraction"].isel(frequency=self.pos, time=slice(t0, t1)).values.astype(np.float32)
            / 255.0
        )
        draft = (
            self.ds["transducer_draft_m"].isel(time=slice(t0, t1)).values
            if "transducer_draft_m" in self.ds
            else np.full(t1 - t0, self.draft)
        )
        return sv, valid, np.asarray(draft, dtype=float)

    def patches_from_chunk(self, k: int, n: int) -> np.ndarray:
        """Up to ``n`` patches (n, C, n_time, n_depth) from chunk ``k`` with enough valid cells."""
        sv, valid, draft = self._load_chunk(k)
        n_t = sv.shape[1] // self.pings_per_cell
        span = self.n_time * self.pings_per_cell
        out = []
        tries = 0
        while len(out) < n and tries < 20 * n:
            tries += 1
            t = int(self.rng.integers(0, max(n_t * self.pings_per_cell - span, 1)))
            d = int(self.rng.integers(self.d_lo, max(self.d_hi, self.d_lo + 1)))
            v = valid[:, t : t + span, d : d + self.n_depth]
            if v.shape[1] < span or v[0].mean() < self.min_valid:
                continue
            s = sv[:, t : t + span, d : d + self.n_depth]
            s = s.reshape(s.shape[0], self.n_time, self.pings_per_cell, self.n_depth)
            v = v.reshape(v.shape[0], self.n_time, self.pings_per_cell, self.n_depth)
            with np.errstate(all="ignore"):
                lin = np.where(np.isfinite(s) & (v > 0), 10.0 ** (s / 10.0), 0.0)
                count = (np.isfinite(s) & (v > 0)).sum(axis=2)
                cell_db = 10.0 * np.log10(np.where(count > 0, lin.sum(axis=2) / np.maximum(count, 1), 1e-13))
            cell_valid = v.mean(axis=2)
            cell_db = np.where(cell_valid > 0, cell_db, self.spec.fill_db)
            depth = (self.range[d : d + self.n_depth] + float(np.nanmedian(draft))) / 100.0
            channels = [cell_db]
            for a, b in self.spec.differences:
                pa, pb = self.spec.frequencies_hz.index(a), self.spec.frequencies_hz.index(b)
                both = (cell_valid[pa] > 0) & (cell_valid[pb] > 0)
                channels.append(np.where(both, cell_db[pa] - cell_db[pb], 0.0)[None])
            if self.spec.include_valid:
                channels.append(cell_valid)
            if self.spec.include_depth:
                channels.append(np.broadcast_to(depth[None, None, :], (1, self.n_time, self.n_depth)))
            out.append(np.concatenate(channels, axis=0).astype(np.float32))
        return (
            np.stack(out)
            if out
            else np.zeros((0, self.spec.n_channels, self.n_time, self.n_depth), dtype=np.float32)
        )

    def fit_stats(self, n_chunks: int = 8, per_chunk: int = 64) -> dict[str, np.ndarray]:
        """Mean and standard deviation per dB and difference channel from a sample of patches."""
        sample = np.concatenate(
            [
                self.patches_from_chunk(int(k), per_chunk)
                for k in self.rng.choice(self.n_chunks, n_chunks, replace=False)
            ]
        )
        n_std = len(self.spec.frequencies_hz) + len(self.spec.differences)
        means, stds = [], []
        for c in range(n_std):
            v = sample[:, c]
            if c < len(self.spec.frequencies_hz) and self.spec.include_valid:
                v = v[sample[:, n_std + c] > 0]
            means.append(float(v.mean()))
            stds.append(float(v.std()) if v.std() > 1e-6 else 1.0)
        self.stats = {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32)}
        return self.stats

    def batch(self, n: int, chunks_per_batch: int = 2) -> np.ndarray:
        """A standardised batch drawn from a few random chunks."""
        parts = [
            self.patches_from_chunk(int(k), int(np.ceil(n / chunks_per_batch)))
            for k in self.rng.choice(self.n_chunks, chunks_per_batch, replace=False)
        ]
        x = np.concatenate(parts)[:n]
        if self.stats is not None:
            n_std = len(self.stats["mean"])
            x[:, :n_std] = (x[:, :n_std] - self.stats["mean"][:, None, None]) / self.stats["std"][
                :, None, None
            ]
        return x


def pretrain(
    sampler: CruisePatchSampler,
    model: Any,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    checkpoint_dir: Path,
    log_every: int = 50,
    resume: bool = False,
) -> list[dict[str, float]]:
    """Masked autoencoding for ``steps`` steps with AdamW and cosine decay; checkpoint every ``log_every``."""
    import time

    import torch

    from .train import make_scheduler

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = make_scheduler(optimizer, steps, max(steps // 20, 1))
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = checkpoint_dir / "mae_last.pt"
    history: list[dict[str, float]] = []
    start = 0
    if resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["step"] + 1, state["history"]
    t0 = time.time()
    running = []
    for step in range(start, steps):
        x = torch.as_tensor(sampler.batch(batch_size), device=device)
        n_f = len(sampler.spec.frequencies_hz)
        valid_weight = None
        if sampler.spec.include_valid:
            pt, pd_ = model.patch
            v = torch.nn.functional.avg_pool2d(
                x[:, n_f + len(sampler.spec.differences) : 2 * n_f + len(sampler.spec.differences)].mean(
                    dim=1, keepdim=True
                ),
                (pt, pd_),
            )
            valid_weight = v.flatten(1)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss, _ = model(x, valid_weight)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running.append(float(loss.item()))
        if (step + 1) % log_every == 0 or step + 1 == steps:
            history.append({"step": step + 1, "loss": float(np.mean(running)), "seconds": time.time() - t0})
            running = []
            log.info("step %d: loss %.4f (%.0f s)", step + 1, history[-1]["loss"], history[-1]["seconds"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "history": history,
                    "encoder": model.encoder.state_dict(),
                    "stats": sampler.stats,
                },
                last,
            )
    return history
