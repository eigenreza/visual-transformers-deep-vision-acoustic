from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from echolabel.models_stage import experiment_config, run_train_acoustic


@pytest.fixture
def window_run(tmp_path):
    """A tiny window dataset of one geometry with splits, laid out as the stages write it."""
    import zarr

    tag = "det_seed0"
    scratch = tmp_path / "scratch"
    project = tmp_path / "project"
    gdir = project / "runs" / "windows" / tag / "base"
    gdir.mkdir(parents=True)
    (project / "runs" / "windows" / tag / "splits").mkdir(parents=True)
    n, f, t, d = 60, 2, 30, 50
    rng = np.random.default_rng(0)
    g = zarr.open_group(str(scratch / "windows" / tag / "base.zarr"), mode="w", zarr_format=2)
    signal = rng.normal(size=(n, 1, 1, 1))
    sv = (-80 + 4 * signal + rng.normal(size=(n, f, t, d))).astype(np.float16)
    g.create_array("sv_db", shape=sv.shape, chunks=(16, f, t, d), dtype=sv.dtype)[:] = sv
    valid = np.full((n, f, t, d), 255, dtype=np.uint8)
    g.create_array("valid", shape=valid.shape, chunks=(16, f, t, d), dtype=valid.dtype)[:] = valid
    g.create_array("hard", shape=valid.shape, chunks=(16, f, t, d), dtype=valid.dtype)[:] = np.zeros_like(
        valid
    )
    depth = np.tile(np.linspace(30, 80, d, dtype=np.float32), (n, 1))
    g.create_array("depth_cells", shape=depth.shape, chunks=(16, d), dtype=depth.dtype)[:] = depth
    g.attrs["frequencies_hz"] = [38000.0, 200000.0]
    # Coarse context of the two-scale geometry: 6 time cells by 20 depth cells.
    ctx = (-85 + 3 * signal + rng.normal(size=(n, f, 6, 20))).astype(np.float16)
    g.create_array("context_sv_db", shape=ctx.shape, chunks=(16, f, 6, 20), dtype=ctx.dtype)[:] = ctx
    cvalid = np.full((n, f, 6, 20), 255, dtype=np.uint8)
    g.create_array("context_valid", shape=cvalid.shape, chunks=(16, f, 6, 20), dtype=cvalid.dtype)[:] = cvalid
    cdepth = np.tile(np.linspace(10, 490, 20, dtype=np.float32), (n, 1))
    g.create_array("context_depth_cells", shape=cdepth.shape, chunks=(16, 20), dtype=cdepth.dtype)[:] = cdepth
    meta = pd.DataFrame(
        {
            "window_id": np.arange(n),
            "transect": ["A"] * 30 + ["B"] * 30,
            "unflagged": True,
            "camera_depth_m": 55.0,
        }
    )
    targets = pd.DataFrame({"window_id": np.arange(n)})
    counts = rng.poisson(np.exp(0.7 + 0.5 * signal[:, 0, 0, 0]))
    for c in ("krill", "herring"):
        base = counts if c == "krill" else np.zeros(n, dtype=int)
        for k in range(5):
            targets[f"hard_{c}_k{k}"] = base.astype(np.float32)
        targets[f"hard_{c}"] = base.astype(np.float32)
        targets[f"hard_{c}_at_lag"] = base.astype(np.float32)
    meta.to_parquet(gdir / "meta.parquet", index=False)
    targets.to_parquet(gdir / "targets.parquet", index=False)
    splits = pd.DataFrame(
        {
            "window_id": np.arange(n),
            "protocol": "loto_unflagged",
            "fold": "test_B",
            "split": ["train"] * 24 + ["val"] * 6 + ["test"] * 30,
        }
    )
    splits.to_parquet(project / "runs" / "windows" / tag / "splits" / "base.parquet", index=False)
    cfg = {
        "paths": {"project_dir": str(project), "scratch_dir": str(scratch)},
        "windows_run": {"experiment": "det", "seed": 0},
        "defaults": {
            "geometry": "base",
            "protocol": "loto_unflagged",
            "family": "nb",
            "inputs": {"frequencies_hz": [38000, 200000], "differences": [[200000, 38000]]},
            "targets": {"form": "hard", "classes": ["krill", "herring"], "lag_offsets": "expected"},
            "sampler": "uniform",
            "train": {"epochs": 2, "batch_size": 8, "warmup_steps": 2, "patience": 3, "amp": False},
            "net": {},
        },
        "experiments": {
            "mean": {"network": "mean"},
            "gbm": {"network": "gbm", "net": {"max_iter": 10}},
            "cnn": {"network": "cnn", "net": {"width": 8, "depth": 2}},
            "cnn_twoscale": {
                "network": "cnn",
                "net": {"width": 8, "depth": 2, "context": True, "context_width": 4},
            },
            "cnn_sample": {
                "network": "cnn",
                "net": {"width": 8, "depth": 2},
                "targets": {"lag_offsets": "sample"},
                "family": "poisson",
            },
        },
    }
    return cfg, project


@pytest.mark.parametrize("experiment", ["mean", "gbm", "cnn", "cnn_sample", "cnn_twoscale"])
def test_train_acoustic_writes_predictions(window_run, experiment) -> None:
    cfg, project = window_run
    exp = experiment_config(cfg, experiment)
    assert exp["geometry"] == "base"
    out = run_train_acoustic(cfg, project / "runs" / "acoustic", experiment=experiment, seed=0, fold="test_B")
    preds = pd.read_parquet(out / "predictions_test.parquet")
    assert len(preds) == 30 and {"mean_krill", "q05_krill", "q95_krill", "y_krill_at_lag"} <= set(
        preds.columns
    )
    assert (preds["q95_krill"] >= preds["q05_krill"]).all() and (preds["mean_krill"] >= 0).all()
    summary = json.load(open(out / "summary.json"))
    assert summary["n_windows"] == {"train": 24, "val": 6, "test": 30} and "test_mae_macro" in summary


def test_baseline_on_named_window_properties(window_run) -> None:
    from echolabel.models.data import WindowStore, context_features
    from echolabel.models_stage import _window_paths

    cfg, project = window_run
    zarr_path, meta_dir, _ = _window_paths(cfg, "base")
    meta = pd.read_parquet(meta_dir / "meta.parquet")
    meta["camera_depth_m"] = np.linspace(20, 300, len(meta))
    meta["time_camera"] = pd.date_range("2018-05-09T06:30:00", periods=len(meta), freq="20s")
    meta.to_parquet(meta_dir / "meta.parquet", index=False)
    feats = context_features(WindowStore(zarr_path, meta_dir), np.array([0, 3]), ["depth", "hour"])
    assert feats.columns.tolist() == ["depth", "hour"] and feats["depth"].iloc[0] == 20.0
    assert np.isclose(feats["hour"].iloc[1], 6.5 + 60 / 3600)
    cfg["experiments"]["gbm_named_features"] = {
        "network": "gbm",
        "features": ["depth"],
        "net": {"max_iter": 10},
    }
    out = run_train_acoustic(
        cfg, project / "runs" / "acoustic", experiment="gbm_named_features", seed=0, fold="test_B"
    )
    preds = pd.read_parquet(out / "predictions_test.parquet")
    assert len(preds) == 30 and np.isfinite(preds["mean_krill"]).all()
