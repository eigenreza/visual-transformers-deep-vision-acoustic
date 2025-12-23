from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.figures_stage import run_plot_results


def test_plot_results_writes_group_figures(tmp_path) -> None:
    rng = np.random.default_rng(0)
    tables = tmp_path / "acoustic_summary"
    tables.mkdir()
    experiments = ["mean", "cnn", "cnn_one_channel", "vit"]
    classes = ["krill", "herring"]
    agg, runs, diffs = [], [], []
    for e in experiments:
        for c in classes:
            v = rng.uniform(1, 3)
            row = {"experiment": e, "protocol": "loto_unflagged", "class_name": c}
            for m in ("mae", "crps", "coverage90"):
                row.update({m: v, f"{m}_ci_low": v - 0.2, f"{m}_ci_high": v + 0.2})
            agg.append(row)
            for fold in ("test_A", "test_B"):
                for seed in (0, 1):
                    runs.append(
                        {
                            "experiment": e,
                            "protocol": "loto_unflagged",
                            "class_name": c,
                            "fold": fold,
                            "seed": seed,
                            "mae": v,
                        }
                    )
        if e != "cnn":
            for c in ("macro_label_classes", "macro_without_krill"):
                for m in ("mae", "crps"):
                    diffs.append(
                        {
                            "experiment": e,
                            "reference": "cnn",
                            "protocol": "loto_unflagged",
                            "class_name": c,
                            "metric": m,
                            "difference": 0.1,
                            "ci_low": -0.1,
                            "ci_high": 0.3,
                        }
                    )
    pd.DataFrame(agg).to_csv(tables / "metrics_aggregated.csv", index=False)
    pd.DataFrame(runs).to_csv(tables / "metrics_by_run.csv", index=False)
    pd.DataFrame(diffs).to_csv(tables / "differences.csv", index=False)
    cfg = {
        "figures": {
            "groups": {"main": ["mean", "cnn", "vit"], "channels": ["cnn_one_channel", "cnn"]},
            "classes": classes,
            "formats": ["png"],
            "fold_classes": ["krill"],
        }
    }
    out = run_plot_results(cfg, tmp_path / "acoustic_figures")
    names = {p.name for p in out.glob("*.png")}
    assert {"main_mae.png", "main_crps.png", "main_coverage90.png", "main_folds_krill.png"} <= names
    assert {"channels_mae.png", "channels_mae_difference.png", "main_mae_difference.png"} <= names
