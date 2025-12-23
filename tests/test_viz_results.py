from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.viz.results import difference_plot, fold_spread_plot, interval_plot


def test_result_figures(tmp_path) -> None:
    rng = np.random.default_rng(0)
    experiments = ["mean", "gbm", "cnn", "vit"]
    rows = []
    for e in experiments:
        for c in ("krill", "herring"):
            v = rng.uniform(1, 3)
            rows.append(
                {"experiment": e, "class_name": c, "mae": v, "mae_ci_low": v - 0.2, "mae_ci_high": v + 0.2}
            )
    table = pd.DataFrame(rows)
    p1 = interval_plot(
        table, "mae", ["krill", "herring"], experiments, tmp_path / "intervals.png", title="test"
    )
    by_run = pd.DataFrame(
        [
            {"experiment": e, "class_name": "krill", "fold": f, "seed": 0, "mae": rng.uniform(1, 3)}
            for e in experiments
            for f in ("test_A", "test_B")
        ]
    )
    p2 = fold_spread_plot(by_run, "mae", "krill", experiments, tmp_path / "folds.png")
    diff = pd.DataFrame(
        [
            {
                "experiment": e,
                "class_name": c,
                "metric": "mae",
                "difference": -0.1,
                "ci_low": -0.3,
                "ci_high": 0.1,
            }
            for e in experiments[1:]
            for c in ("krill", "herring")
        ]
    )
    p3 = difference_plot(diff, "mae", ["krill", "herring"], tmp_path / "diff.png")
    assert p1.is_file() and p2.is_file() and p3.is_file()
