"""Baseline predictors of counts.

The mean predictor returns the training mean per class as a Poisson rate.
The frequency-response baseline fits one gradient-boosted regressor with a
Poisson deviance per class on hand-crafted window statistics (levels per
channel, dB differences between channels, depth) and estimates a negative
binomial dispersion per class from the training residuals, so that it yields
a count distribution like the networks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-6


class MeanPredictor:
    family = "poisson"

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> MeanPredictor:
        self.mean_ = np.maximum(y.mean(axis=0), EPS)
        return self

    def predict_params(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"log_mu": np.log(np.tile(self.mean_, (len(features), 1)))}


class FeatureBoosting:
    """Per-class gradient boosting with Poisson loss and a moment estimate of negative binomial dispersion."""

    family = "nb"

    def __init__(
        self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 15, seed: int = 0
    ) -> None:
        self.params: dict[str, Any] = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "loss": "poisson",
            "random_state": seed,
        }
        self.models_: list[HistGradientBoostingRegressor | None] = []
        self.alpha_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> FeatureBoosting:
        self.models_ = []
        alphas = []
        for c in range(y.shape[1]):
            target = y[:, c]
            if target.sum() <= 0:
                self.models_.append(None)
                alphas.append(1.0)
                continue
            model = HistGradientBoostingRegressor(**self.params).fit(features.to_numpy(), target)
            mu = np.maximum(model.predict(features.to_numpy()), EPS)
            excess = np.mean((target - mu) ** 2 - mu)
            alphas.append(float(max(excess / max(np.mean(mu**2), EPS), 1e-3)))
            self.models_.append(model)
        self.alpha_ = np.asarray(alphas)
        return self

    def predict_params(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(features)
        mu = np.full((n, len(self.models_)), EPS)
        for c, model in enumerate(self.models_):
            if model is not None:
                mu[:, c] = np.maximum(model.predict(features.to_numpy()), EPS)
        return {"log_mu": np.log(mu), "log_alpha": np.log(np.tile(self.alpha_, (n, 1)))}


def build_baseline(name: str, seed: int = 0, cfg: dict[str, Any] | None = None):
    cfg = dict(cfg or {})
    if name == "mean":
        return MeanPredictor()
    if name == "gbm":
        return FeatureBoosting(seed=seed, **cfg)
    raise KeyError(name)
