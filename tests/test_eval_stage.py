from __future__ import annotations

import pandas as pd

from echolabel.eval_stage import run_evaluate_acoustic
from echolabel.models_stage import run_train_acoustic
from tests.test_models_stage import window_run  # noqa: F401


def test_evaluate_acoustic_aggregates_runs(window_run) -> None:  # noqa: F811
    cfg, project = window_run
    cfg["evaluation"] = {"n_samples": 50, "n_boot": 30, "reference": "mean", "contrasts": [["cnn", "mean"]]}
    for experiment in ("mean", "cnn"):
        run_train_acoustic(cfg, project / "runs" / "acoustic", experiment=experiment, seed=0, fold="test_B")
    out = run_evaluate_acoustic(cfg, project / "runs" / "acoustic_summary")
    by_run = pd.read_csv(out / "metrics_by_run.csv")
    assert set(by_run["experiment"]) == {"mean", "cnn"} and {
        "mae",
        "crps",
        "coverage90",
        "presence_ap",
    } <= set(by_run.columns)
    agg = pd.read_csv(out / "metrics_aggregated.csv")
    assert len(agg) == 4 and {"mae_ci_low", "presence_ap_ci_low"} <= set(agg.columns)
    # The value and its interval describe the same quantity, the mean over folds.
    assert ((agg["mae_ci_low"] <= agg["mae"] + 1e-9) & (agg["mae"] <= agg["mae_ci_high"] + 1e-9)).all()
    macro = pd.read_csv(out / "macro.csv")
    assert {"label_classes", "without_krill"} <= set(macro["subset"]) and "crps_ci_low" in macro.columns
    contrasts = pd.read_csv(out / "contrasts.csv")
    assert set(contrasts["experiment"]) == {"cnn"} and set(contrasts["reference"]) == {"mean"}
    diff = pd.read_csv(out / "differences.csv")
    assert set(diff["experiment"]) == {"cnn"} and set(diff["metric"]) == {"mae", "crps", "log_score"}
    assert (
        project
        / "runs"
        / "acoustic"
        / "cnn"
        / "loto_unflagged"
        / "test_B"
        / "seed0"
        / "eval"
        / "choices.json"
    ).is_file()


def test_apply_truth_replaces_form_specific_targets() -> None:
    import pandas as pd

    from echolabel.eval_stage import apply_truth

    preds = pd.DataFrame({"window_id": [3, 1], "y_krill_at_lag": [0.4, 2.6], "mean_krill": [1.0, 2.0]})
    truth = pd.DataFrame(
        {"window_key": ["A|10", "A|20", "A|30"], "y_krill_at_lag": [5.0, 0.0, 1.0]},
        index=pd.Index([1, 2, 3], name="window_id"),
    )
    out = apply_truth(preds, truth, ["krill"])
    assert out["y_krill_at_lag"].tolist() == [1.0, 5.0] and out["window_key"].tolist() == ["A|30", "A|10"]
    # Without a dataset on disk the table keeps its own truth and pairs by window identifier.
    kept = apply_truth(preds, None, ["krill"])
    assert kept["y_krill_at_lag"].tolist() == [0.4, 2.6] and kept["window_key"].tolist() == ["3", "1"]


def test_truth_form_of_an_experiment_selects_its_truth(window_run) -> None:  # noqa: F811
    from echolabel.eval_stage import window_truth

    cfg, project = window_run
    meta_dir = project / "runs" / "windows" / "det_seed0" / "base"
    targets = pd.read_parquet(meta_dir / "targets.parquet")
    targets["occupied_krill_at_lag"] = 7.0
    targets["occupied_herring_at_lag"] = 0.0
    targets.to_parquet(meta_dir / "targets.parquet", index=False)
    cfg["experiments"]["cnn_frames"] = {
        "network": "cnn",
        "targets": {"form": "hard", "truth_form": "occupied"},
    }
    cache: dict = {}
    usual = window_truth(cfg, "cnn", cache)
    other = window_truth(cfg, "cnn_frames", cache)
    assert (other["y_krill_at_lag"] == 7.0).all() and set(other["truth_form"]) == {"occupied"}
    assert not (usual["y_krill_at_lag"] == 7.0).all() and set(usual["truth_form"]) == {"hard"}
