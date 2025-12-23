from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.eval.metrics import (
    average_precision,
    best_f1_threshold,
    bootstrap_mean,
    bootstrap_statistic,
    conformal_adjustment,
    evaluate_predictions,
    macro_over_classes,
    paired_bootstrap_difference,
    presence_probability,
    reliability,
)


def _predictions(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    classes = ["krill", "herring"]
    table = pd.DataFrame({"window_id": np.arange(n), "split": "test"})
    for k, c in enumerate(classes):
        mu = np.exp(rng.normal(0.5 if k == 0 else -1.0, 0.5, n))
        y = rng.poisson(mu)
        table[f"y_{c}_at_lag"] = y.astype(float)
        table[f"y_{c}_expected"] = y.astype(float)
        table[f"log_mu_{c}"] = np.log(mu)
        table[f"log_alpha_{c}"] = np.log(0.3)
        table[f"mean_{c}"] = mu
        from scipy import stats

        r = 1 / 0.3
        p = r / (r + mu)
        table[f"q05_{c}"] = stats.nbinom.ppf(0.05, r, p)
        table[f"q95_{c}"] = stats.nbinom.ppf(0.95, r, p)
    return table, classes


def test_presence_and_ap() -> None:
    params = {"log_mu": np.log(np.array([[0.01, 5.0]])), "log_alpha": np.log(np.array([[0.3, 0.3]]))}
    prob = presence_probability("nb", params)
    assert prob[0, 0] < 0.02 and prob[0, 1] > 0.9
    score = np.array([0.9, 0.8, 0.3, 0.1])
    truth = np.array([1, 1, 0, 0])
    assert average_precision(score, truth) == 1.0 and np.isnan(average_precision(score, np.zeros(4)))
    assert 0.3 < best_f1_threshold(score, truth) <= 0.8
    table, ece = reliability(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0, 0, 1, 1]))
    assert len(table) == 2 and ece < 0.11


def test_bootstrap_and_conformal() -> None:
    v = np.arange(100, dtype=float)
    b = bootstrap_mean(v, n_boot=100)
    assert b["ci_low"] < b["value"] < b["ci_high"]
    s = bootstrap_statistic(lambda idx: float(v[idx].mean()), 100, n_boot=50)
    assert abs(s["value"] - 49.5) < 1e-9
    d = paired_bootstrap_difference(v, v + 1.0, n_boot=50)
    assert abs(d["difference"] - 1.0) < 1e-9 and d["ci_low"] > 0.99
    adj = conformal_adjustment(np.zeros((50, 1)), np.ones((50, 1)), np.linspace(0, 2, 50)[:, None], alpha=0.1)
    assert adj.shape == (1,) and 0.7 < adj[0] <= 1.0


def test_evaluate_predictions_table() -> None:
    preds, classes = _predictions()
    table, contributions = evaluate_predictions(preds, classes, "nb", n_samples=100)
    assert set(table["class_name"]) == set(classes)
    row = table.set_index("class_name").loc["krill"]
    assert 0.8 < row["coverage90"] <= 1.0 and row["crps"] < row["mae"] and row["presence_ap"] > 0.5
    assert contributions["abs_error"].shape == (200, 2) and contributions["crps"].shape == (200, 2)
    macro = macro_over_classes(table, classes, ("mae", "crps"))
    assert set(macro) == {"mae", "crps"}
    with_thr = evaluate_predictions(
        preds,
        classes,
        "nb",
        thresholds={"krill": 0.5, "herring": 0.5},
        conformal={"krill": 1.0, "herring": 0.0},
        n_samples=50,
    )[0]
    assert with_thr.set_index("class_name").loc["krill", "width90"] > row["width90"]


def test_parameter_names_with_underscored_class() -> None:
    from echolabel.eval.metrics import parameter_names

    classes = ["krill", "herring_bluewhiting"]
    columns = [f"{p}_{c}" for p in ("log_mu", "log_alpha") for c in classes] + [
        "mean_krill",
        "y_krill_at_lag",
    ]
    assert parameter_names(columns, classes) == ["log_mu", "log_alpha"]
    assert parameter_names(["log1p_mu_krill", "log1p_mu_herring_bluewhiting"], classes) == ["log1p_mu"]
    assert parameter_names(columns, []) == []


def test_block_bootstrap_is_wider_than_window_bootstrap_for_correlated_windows() -> None:
    from echolabel.eval.metrics import block_bootstrap_mean, bootstrap_mean

    rng = np.random.default_rng(0)
    block_effect = np.repeat(rng.normal(size=40), 15)  # 40 blocks of 15 identical-ish windows
    values = block_effect + 0.1 * rng.normal(size=block_effect.size)
    blocks = np.repeat(np.arange(40), 15)
    by_block = block_bootstrap_mean(values, blocks, n_boot=400)
    by_window = bootstrap_mean(values, n_boot=400)
    assert by_block["n_blocks"] == 40 and by_block["n"] == 600
    assert np.isclose(by_block["value"], values.mean())
    assert (by_block["ci_high"] - by_block["ci_low"]) > 2.5 * (by_window["ci_high"] - by_window["ci_low"])


def test_fold_block_bootstrap_weights_folds_equally() -> None:
    from echolabel.eval.metrics import fold_block_bootstrap

    # Two folds of unequal size with constant values: the value is the mean of the fold
    # means, not the mean over windows, and the interval collapses onto it.
    values = np.concatenate([np.full(30, 10.0), np.full(90, 2.0)])
    folds = np.array(["a"] * 30 + ["b"] * 90)
    blocks = np.array([f"{f}|{i // 10}" for i, f in enumerate(folds)])
    out = fold_block_bootstrap(values, blocks, folds, n_boot=200)
    assert out["value"] == 6.0 and out["ci_low"] == 6.0 and out["ci_high"] == 6.0
    assert out["n_folds"] == 2 and out["n_blocks"] == 12
    # Blocks are drawn within their fold: a fold never receives the windows of another.
    rng = np.random.default_rng(0)
    noisy = values + rng.normal(0, 1, values.size)
    spread = fold_block_bootstrap(noisy, blocks, folds, n_boot=400)
    assert 5.0 < spread["ci_low"] < spread["value"] < spread["ci_high"] < 7.0
    assert np.isnan(fold_block_bootstrap(np.array([]), np.array([]), np.array([]))["value"])
