from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.error_stage import (
    edge_bins,
    quantile_bins,
    run_analyse_errors,
    tracking_rows,
    window_errors,
)
from echolabel.models_stage import run_train_acoustic
from tests.test_models_stage import window_run  # noqa: F401


def test_edge_and_quantile_bins() -> None:
    labels, order = edge_bins(np.array([0, 5, 50, 2000, np.nan, -1]), [0, 1, 10, 100, 1000, float("inf")])
    assert order == ["0 to 1", "1 to 10", "10 to 100", "100 to 1000", "1000+"]
    assert labels.tolist() == ["0 to 1", "1 to 10", "10 to 100", "1000+", "", ""]
    bins, names = quantile_bins(np.array([*np.arange(9.0), np.nan]), 3)
    assert len(names) == 3 and bins[-1] == "" and [list(bins[:9]).count(n) for n in names] == [3, 3, 3]
    assert quantile_bins(np.array([np.nan, 1.0]), 3)[1] == []


def test_window_errors_average_over_seeds() -> None:
    seeds = [
        pd.DataFrame({"window_id": [1, 2], "y_krill": [10.0, 0.0], "mean_krill": [12.0, 1.0]}),
        pd.DataFrame({"window_id": [2, 1], "y_krill": [0.0, 10.0], "mean_krill": [3.0, 6.0]}),
    ]
    out = window_errors(seeds, ["krill"]).set_index("window_id")
    # Window 1: errors +2 and -4, so the absolute error is 3 and the signed error -1.
    assert out.loc[1, "abs_error_krill"] == 3.0 and out.loc[1, "error_krill"] == -1.0
    assert out.loc[2, "mean_krill"] == 2.0 and out.loc[2, "abs_error_krill"] == 2.0
    assert np.isclose(out.loc[2, "log_error_krill"], (np.log(2.0) + np.log(4.0)) / 2)


def test_tracking_separates_order_from_level() -> None:
    # Fold a follows the truth at a tenth of its level, fold b predicts a constant.
    y = np.arange(40.0)
    windows = pd.DataFrame(
        {
            "fold": ["a"] * 20 + ["b"] * 20,
            "y_krill": y,
            "mean_krill": np.concatenate([y[:20] / 10, np.full(20, 5.0)]),
        }
    )
    blocks = np.array([f"{f}|{i // 4}" for i, f in enumerate(windows["fold"])])
    rows = {r["fold"]: r for r in tracking_rows(windows, "krill", blocks, n_boot=50)}
    assert rows["a"]["spearman"] == 1.0 and np.isnan(rows["b"]["spearman"])
    assert rows["mean over folds"]["spearman"] == 1.0


def test_analyse_errors_writes_tables(window_run) -> None:  # noqa: F811
    cfg, project = window_run
    cfg["evaluation"] = {"n_boot": 20}
    cfg["errors"] = {"experiments": ["mean"], "formats": ["png"], "classes": ["krill"]}
    for seed in (0, 1):
        run_train_acoustic(cfg, project / "runs" / "acoustic", experiment="mean", seed=seed, fold="test_B")
    out = run_analyse_errors(cfg, project / "runs" / "acoustic_errors")
    table = pd.read_csv(out / "errors_by_factor.csv")
    assert {"transect", "camera_depth_m", "true_count"} <= set(table["factor"])
    by_transect = table[table["factor"] == "transect"]
    assert by_transect["n_windows"].sum() == 30 and (by_transect["mae_ci_low"] <= by_transect["mae"]).all()
    assert (out / "errors_mean.png").is_file() and (out / "windows_mean.parquet").is_file()
    tracking = pd.read_csv(out / "tracking.csv")
    assert set(tracking["fold"]) == {"test_B"} or "mean over folds" in set(tracking["fold"])


def test_paired_errors_on_shared_windows() -> None:
    from echolabel.error_stage import paired_errors

    first = pd.DataFrame(
        {
            "window_id": [1, 2, 3, 4],
            "transect": "A",
            "abs_error_krill": [1.0, 2.0, 3.0, 4.0],
            "abs_error_herring": [0.0, 0.0, 1.0, 1.0],
        }
    )
    second = pd.DataFrame(
        {
            "window_id": [2, 3, 4, 5],
            "transect": "A",
            "abs_error_krill": [4.0, 5.0, 6.0, 0.0],
            "abs_error_herring": [0.0, 1.0, 1.0, 9.0],
        }
    )
    rows = {r["class_name"]: r for r in paired_errors(first, second, ["krill", "herring"], 300.0, 50)}
    # Three shared windows; krill differs by -2 in each, herring by nothing.
    assert rows["krill"]["n_windows"] == 3 and rows["krill"]["difference"] == -2.0
    assert rows["krill"]["mae"] == 3.0 and rows["krill"]["mae_other"] == 5.0
    assert rows["herring"]["difference"] == 0.0 and rows["mean over classes"]["difference"] == -1.0
    assert paired_errors(first, second[second["window_id"] > 4], ["krill"], 300.0, 10) == []
