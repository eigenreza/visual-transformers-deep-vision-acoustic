from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from echolabel.models.baselines import build_baseline
from echolabel.models.counts import n_params, predictive_mean
from echolabel.models.nets import build_network
from echolabel.models.train import TrainConfig, predict_params, sample_weights, train_model


def test_train_model_runs_and_resumes(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 4, 30, 50)).astype(np.float32)
    x[:, -1] = 0.5
    y = rng.poisson(2.0, size=(48, 2)).astype(np.float32)
    offsets = np.stack([y, y + 1, y], axis=2)
    net = build_network("cnn", 4, 2 * n_params("nb"), {"width": 8, "depth": 2})
    cfg = TrainConfig(epochs=2, batch_size=16, warmup_steps=2, patience=5, amp=False)
    out = train_model(
        net,
        "nb",
        x[:32],
        y[:32],
        x[32:],
        y[32:],
        cfg,
        "cpu",
        0,
        tmp_path,
        offsets[:32],
        np.array([0.25, 0.5, 0.25]),
        "magnitude",
    )
    assert len(out["history"]) == 2 and out["best_epoch"] >= 0 and (tmp_path / "last.pt").is_file()
    cfg2 = TrainConfig(epochs=3, batch_size=16, warmup_steps=2, patience=5, amp=False)
    out2 = train_model(net, "nb", x[:32], y[:32], x[32:], y[32:], cfg2, "cpu", 0, tmp_path, resume=True)
    assert len(out2["history"]) == 3  # one more epoch after resumption
    params = predict_params(net, "nb", x[32:], 2, "cpu")
    assert set(params) == {"log_mu", "log_alpha"} and predictive_mean("nb", params).shape == (16, 2)
    assert (
        sample_weights(y, "uniform") is None and abs(sample_weights(y, "presence_balanced").sum() - 1) < 1e-9
    )


def test_baselines() -> None:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    y = np.stack([rng.poisson(np.exp(features["a"].to_numpy())), np.zeros(60)], axis=1)
    mean = build_baseline("mean").fit(features, y)
    p = mean.predict_params(features.iloc[:5])
    assert p["log_mu"].shape == (5, 2) and np.allclose(np.exp(p["log_mu"][0, 0]), y[:, 0].mean())
    gbm = build_baseline("gbm", cfg={"max_iter": 20}).fit(features, y)
    q = gbm.predict_params(features.iloc[:5])
    assert set(q) == {"log_mu", "log_alpha"} and q["log_alpha"].shape == (5, 2)
    assert gbm.models_[1] is None  # a class without any count is skipped
    torch.manual_seed(0)
