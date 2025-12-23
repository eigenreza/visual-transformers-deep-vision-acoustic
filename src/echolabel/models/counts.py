"""Count distributions for the acoustic models.

A model predicts, per class, the parameters of a distribution over counts
with a log link on the mean: Poisson (one parameter), negative binomial
(mean and dispersion), zero-inflated negative binomial (plus the probability
of a structural zero), Tweedie (mean, fixed power, fitted dispersion for the
likelihood), and the log(1 + y) regression evaluated in count space with a
smearing correction. The same module gives the predictive mean, quantiles,
samples and proper scoring rules so that every head is evaluated the same
way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import stats

FAMILIES = {"poisson": 1, "nb": 2, "zinb": 3, "tweedie": 1, "log1p": 1}
EPS = 1e-6


def n_params(family: str) -> int:
    return FAMILIES[family]


def split_params(family: str, out: torch.Tensor, n_classes: int) -> dict[str, torch.Tensor]:
    """Named parameters from the network output of shape (N, n_classes * n_params)."""
    k = n_params(family)
    p = out.view(out.shape[0], n_classes, k)
    if family == "log1p":
        return {"log1p_mu": p[..., 0]}
    params = {"log_mu": p[..., 0].clamp(-10.0, 12.0)}
    if family in ("nb", "zinb"):
        params["log_alpha"] = p[..., 1].clamp(-8.0, 6.0)
    if family == "zinb":
        params["logit_pi"] = p[..., 2].clamp(-10.0, 10.0)
    return params


def nll(
    family: str, params: dict[str, torch.Tensor], y: torch.Tensor, tweedie_power: float = 1.5
) -> torch.Tensor:
    """Negative log likelihood per sample, averaged over classes (Tweedie: unit deviance)."""
    if family == "log1p":
        return ((params["log1p_mu"] - torch.log1p(y)) ** 2).mean(dim=1)
    log_mu = params["log_mu"]
    mu = log_mu.exp()
    if family == "poisson":
        ll = y * log_mu - mu - torch.lgamma(y + 1)
        return -ll.mean(dim=1)
    if family == "tweedie":
        p = tweedie_power
        dev = 2 * (
            y.clamp(min=0) ** (2 - p) / ((1 - p) * (2 - p))
            - y * mu ** (1 - p) / (1 - p)
            + mu ** (2 - p) / (2 - p)
        )
        return dev.mean(dim=1)
    alpha = params["log_alpha"].exp()
    r = 1.0 / alpha
    log_r = -params["log_alpha"]
    ll = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1)
        + r * (log_r - torch.logaddexp(log_r, log_mu))
        + y * (log_mu - torch.logaddexp(log_r, log_mu))
    )
    if family == "nb":
        return -ll.mean(dim=1)
    logit_pi = params["logit_pi"]
    log_pi = torch.nn.functional.logsigmoid(logit_pi)
    log_1m_pi = torch.nn.functional.logsigmoid(-logit_pi)
    zero = torch.logaddexp(log_pi, log_1m_pi + ll)
    nonzero = log_1m_pi + ll
    return -torch.where(y > 0, nonzero, zero).mean(dim=1)


def predictive_mean(family: str, params: dict[str, np.ndarray], smearing: float = 1.0) -> np.ndarray:
    """Mean count per class from the parameters (numpy)."""
    if family == "log1p":
        return np.maximum(np.expm1(params["log1p_mu"]) * smearing + (smearing - 1.0), 0.0)
    mu = np.exp(params["log_mu"])
    if family == "zinb":
        return (1.0 - _sigmoid(params["logit_pi"])) * mu
    return mu


def sample(
    family: str,
    params: dict[str, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
    tweedie_power: float = 1.5,
    tweedie_phi: float = 1.0,
    log1p_sigma: float = 0.5,
) -> np.ndarray:
    """Samples of shape (n_samples, N, K) from the predictive distribution."""
    if family == "log1p":
        mu = params["log1p_mu"]
        z = rng.normal(size=(n_samples, *mu.shape)) * log1p_sigma
        return np.maximum(np.expm1(mu[None] + z), 0.0)
    mu = np.exp(params["log_mu"])
    if family == "poisson":
        return rng.poisson(np.broadcast_to(mu, (n_samples, *mu.shape)))
    if family == "tweedie":
        return _tweedie_sample(mu, tweedie_power, tweedie_phi, n_samples, rng)
    alpha = np.exp(params["log_alpha"])
    r = 1.0 / alpha
    p = r / (r + mu)
    draws = rng.negative_binomial(
        np.broadcast_to(r, (n_samples, *mu.shape)), np.broadcast_to(p, (n_samples, *mu.shape))
    )
    if family == "zinb":
        pi = _sigmoid(params["logit_pi"])
        zero = rng.random((n_samples, *mu.shape)) < pi
        draws = np.where(zero, 0, draws)
    return draws


def quantiles(family: str, params: dict[str, np.ndarray], qs: tuple[float, ...], **kwargs: Any) -> np.ndarray:
    """Predictive quantiles (len(qs), N, K): closed form for Poisson and negative binomial, else samples."""
    if family == "poisson":
        mu = np.exp(params["log_mu"])
        return np.stack([stats.poisson.ppf(q, mu) for q in qs])
    if family == "nb":
        mu = np.exp(params["log_mu"])
        r = 1.0 / np.exp(params["log_alpha"])
        p = r / (r + mu)
        return np.stack([stats.nbinom.ppf(q, r, p) for q in qs])
    rng = np.random.default_rng(kwargs.pop("seed", 0))
    draws = sample(family, params, int(kwargs.pop("n_samples", 400)), rng, **kwargs)
    return np.quantile(draws, qs, axis=0)


def crps_from_samples(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score per element from samples of shape (S, ...) against ``y``."""
    s = draws.shape[0]
    term1 = np.abs(draws - y[None]).mean(axis=0)
    sorted_draws = np.sort(draws, axis=0)
    weights = (
        (2 * np.arange(1, s + 1) - s - 1)[:, None, None]
        if draws.ndim == 3
        else (2 * np.arange(1, s + 1) - s - 1)[:, None]
    )
    term2 = (weights * sorted_draws).sum(axis=0) / (s * s)
    return term1 - term2


def log_score(family: str, params: dict[str, np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Negative log probability of the observed counts per element (numpy, same formulas as the loss)."""
    t_params = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in params.items()}
    t_y = torch.as_tensor(np.asarray(y, dtype=np.float64))
    with torch.no_grad():
        per_class = _nll_per_class(family, t_params, t_y, float(kwargs.get("tweedie_power", 1.5)))
    return per_class.numpy()


def _nll_per_class(
    family: str, params: dict[str, torch.Tensor], y: torch.Tensor, tweedie_power: float
) -> torch.Tensor:
    k = y.shape[-1]
    out = []
    for c in range(k):
        p = {name: v[..., c : c + 1] for name, v in params.items()}
        out.append(nll(family, p, y[..., c : c + 1], tweedie_power))
    return torch.stack(out, dim=-1)


def smearing_factor(log1p_pred: np.ndarray, y: np.ndarray) -> float:
    """Duan's smearing estimate, exp(residual) averaged over training windows, for log(1 + y) regression."""
    residual = np.log1p(np.maximum(y, 0)) - log1p_pred
    return float(np.mean(np.exp(residual)))


def tweedie_dispersion(mu: np.ndarray, y: np.ndarray, power: float = 1.5) -> float:
    """Moment estimate of the Tweedie dispersion from the training residuals."""
    mu = np.maximum(mu, EPS)
    return float(np.mean((y - mu) ** 2 / mu**power))


def _tweedie_sample(
    mu: np.ndarray, p: float, phi: float, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Compound Poisson-gamma samples with mean ``mu``, power ``p`` in (1, 2) and dispersion ``phi``."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    n = rng.poisson(np.broadcast_to(lam, (n_samples, *mu.shape)))
    return rng.gamma(np.maximum(n * shape, EPS), np.broadcast_to(scale, n.shape)) * (n > 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def interval_coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Coverage and mean width of predictive intervals."""
    inside = (y >= lower) & (y <= upper)
    return {"coverage": float(inside.mean()), "mean_width": float((upper - lower).mean())}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split conformal quantile of nonconformity scores for miscoverage ``alpha``, finite-sample corrected."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])
