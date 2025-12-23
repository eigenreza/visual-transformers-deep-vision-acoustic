from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.attribution_stage import run_attribute_acoustic, run_summarize_attribution
from echolabel.models_stage import run_train_acoustic
from tests.test_models_stage import window_run  # noqa: F401


def test_attribute_acoustic_writes_cases(window_run) -> None:  # noqa: F811
    cfg, project = window_run
    cfg["attribution"] = {"n_cases_per_class": 1, "batch_size": 16}
    run_train_acoustic(cfg, project / "runs" / "acoustic", experiment="cnn", seed=0, fold="test_B")
    out = run_attribute_acoustic(
        cfg, project / "runs" / "attribution", experiment="cnn", seed=0, fold="test_B"
    )
    mass = pd.read_csv(out / "band_mass.csv")
    assert set(mass["method"]) == {"integrated_gradients", "grad_cam"} and set(mass["group"]) == {
        "all",
        "better_half",
        "worse_half",
    }
    cases = pd.read_csv(out / "cases.csv")
    assert len(cases) > 0 and all((out / f).is_file() for f in cases["figure"])
    cfg["evaluation"] = {"n_boot": 20}
    cfg["attribution"]["formats"] = ["png"]
    summary_dir = run_summarize_attribution(cfg, project / "runs" / "attribution_summary")
    summary = pd.read_csv(summary_dir / "band_mass_summary.csv")
    assert set(summary["class_name"]) == {"krill", "herring", "all"}
    # The band of 5 m on either side of the camera covers a fifth of the 50 m window.
    assert np.allclose(summary["uniform_share"], 0.2, atol=0.03)
    assert {"excess_ci_low", "worse_minus_better"} <= set(summary.columns)
    assert (summary_dir / "band_mass.png").is_file()
