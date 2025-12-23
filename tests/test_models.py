from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from echolabel.models.counts import (
    crps_from_samples,
    interval_coverage,
    log_score,
    nll,
    predictive_mean,
    quantiles,
    sample,
    smearing_factor,
    split_params,
)
from echolabel.models.data import (
    InputSpec,
    TargetSpec,
    WindowStore,
    build_inputs,
    build_targets,
    normalisation_stats,
    window_features,
)
from echolabel.models.nets import MaskedAutoencoder, build_network


@pytest.fixture
def store(tmp_path):
    import zarr

    n, f, t, d = 40, 3, 30, 100
    rng = np.random.default_rng(0)
    g = zarr.open_group(str(tmp_path / "base.zarr"), mode="w", zarr_format=2)
    sv = (-80 + 5 * rng.normal(size=(n, f, t, d))).astype(np.float16)
    valid = np.full((n, f, t, d), 255, dtype=np.uint8)
    valid[:, :, :, :10] = 0
    g.create_array("sv_db", shape=sv.shape, chunks=(8, f, t, d), dtype=sv.dtype)[:] = sv
    g.create_array("valid", shape=valid.shape, chunks=(8, f, t, d), dtype=valid.dtype)[:] = valid
    g.create_array("hard", shape=valid.shape, chunks=(8, f, t, d), dtype=valid.dtype)[:] = np.zeros_like(
        valid
    )
    depth = np.tile(np.linspace(20, 70, d, dtype=np.float32), (n, 1))
    g.create_array("depth_cells", shape=depth.shape, chunks=(8, d), dtype=depth.dtype)[:] = depth
    g.attrs["frequencies_hz"] = [38000.0, 18000.0, 200000.0]
    meta = pd.DataFrame({"window_id": np.arange(n), "transect": ["A"] * 20 + ["B"] * 20})
    targets = pd.DataFrame({"window_id": np.arange(n)})
    for c in ("krill", "herring"):
        for k in range(5):
            targets[f"hard_{c}_k{k}"] = rng.poisson(2.0, n).astype(np.float32)
        targets[f"hard_{c}"] = targets[[f"hard_{c}_k{k}" for k in range(5)]].mean(axis=1)
        targets[f"hard_{c}_at_lag"] = targets[f"hard_{c}_k2"]
    meta.to_parquet(tmp_path / "meta.parquet", index=False)
    targets.to_parquet(tmp_path / "targets.parquet", index=False)
    splits = pd.DataFrame(
        {
            "window_id": np.arange(n),
            "protocol": "loto_all",
            "fold": "test_B",
            "split": ["train"] * 15 + ["val"] * 5 + ["test"] * 20,
        }
    )
    splits.to_parquet(tmp_path / "splits.parquet", index=False)
    return WindowStore(tmp_path / "base.zarr", tmp_path), tmp_path / "splits.parquet"


def test_window_store_inputs_and_targets(store) -> None:
    ws, splits_path = store
    assert ws.classes == ["herring", "krill"]
    parts = ws.splits(splits_path, "loto_all", "test_B")
    assert len(parts["train"]) == 15 and len(parts["test"]) == 20
    spec = InputSpec([38000.0, 200000.0], differences=[(200000.0, 38000.0)])
    assert spec.n_channels == 6
    stats = normalisation_stats(ws, parts["train"], spec)
    x = build_inputs(ws, parts["val"], spec, stats)
    assert x.shape == (5, 6, 30, 100) and np.isfinite(x).all()
    assert abs(x[:, 0][x[:, 3] > 0].mean()) < 0.5  # standardised over valid cells
    assert (
        np.allclose(x[:, 5, 0, 0], 0.2) and x[:, 3, :, :10].max() == 0
    )  # depth in hundreds of metres; invalid cells
    tspec = TargetSpec("hard", ["krill", "herring"], "expected")
    y = build_targets(ws, parts["train"], tspec)
    assert y.shape == (15, 2)
    y_sample = build_targets(
        ws, parts["train"], TargetSpec("hard", ["krill", "herring"], "sample"), np.random.default_rng(1)
    )
    assert y_sample.shape == (15, 2) and (y_sample >= 0).all()
    feats = window_features(ws, parts["train"][:4], spec)
    assert {"f38_mean", "f200_q90", "d200_38_mean", "depth_mean"} <= set(feats.columns)


def test_count_families_losses_and_scores() -> None:
    torch.manual_seed(0)
    y = torch.tensor([[0.0, 3.0], [5.0, 0.0], [1.0, 1.0]])
    for family in ("poisson", "nb", "zinb", "tweedie", "log1p"):
        out = torch.randn(
            3, 2 * {"poisson": 1, "nb": 2, "zinb": 3, "tweedie": 1, "log1p": 1}[family], requires_grad=True
        )
        params = split_params(family, out, 2)
        loss = nll(family, params, y).mean()
        loss.backward()
        assert torch.isfinite(loss) and out.grad is not None
        np_params = {k: v.detach().numpy() for k, v in params.items()}
        mean = predictive_mean(family, np_params)
        assert mean.shape == (3, 2) and (mean >= 0).all()
        draws = sample(family, np_params, 50, np.random.default_rng(0))
        assert draws.shape == (50, 3, 2) and (draws >= 0).all()
        crps = crps_from_samples(draws, y.numpy())
        assert crps.shape == (3, 2) and (crps >= 0).all()
        q = quantiles(family, np_params, (0.05, 0.95), n_samples=50)
        assert q.shape == (2, 3, 2) and (q[1] >= q[0]).all()
        ls = log_score(family, np_params, y.numpy())
        assert ls.shape == (3, 2)
    cov = interval_coverage(np.zeros(4), np.array([1, 1, 1, 1]), np.array([0, 1, 2, 0.5]))
    assert cov["coverage"] == 0.75
    assert abs(smearing_factor(np.log1p(np.array([1.0, 3.0])), np.array([1.0, 3.0])) - 1.0) < 1e-9


@pytest.mark.parametrize("name", ["cnn", "vit", "hybrid"])
def test_networks_forward_and_backward(name) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 30, 100)
    x[:, -1] = 0.4  # depth channel in hundreds of metres
    net = build_network(name, 6, 4)
    out = net(x)
    assert out.shape == (2, 4)
    out.sum().backward()
    assert sum(p.numel() for p in net.parameters()) < 5_000_000


def test_masked_autoencoder_loss() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 30, 100)
    x[:, -1] = 0.5
    mae = MaskedAutoencoder(4, patch=(3, 10), dim=32, depth=1, heads=2, decoder_dim=16)
    loss, recon = mae(x)
    assert torch.isfinite(loss) and recon.shape == (2, 100, 3 * 30)
    loss.backward()
