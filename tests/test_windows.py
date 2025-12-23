from __future__ import annotations

import numpy as np
import pandas as pd

from echolabel.windows.build import Geometry, TransectPatches, band_top, window_grid
from echolabel.windows.splits import (
    block_splits,
    buffer_by_transect,
    buffer_by_window,
    buffer_seconds,
    check_splits,
    loto_folds,
    split_counts,
)
from echolabel.windows.targets import build_targets, interval_sums, log1p_targets


def _store(n_p: int = 400, n_r: int = 500):
    ping_time = np.datetime64("2018-05-09T20:00:00") + (np.arange(n_p) * 1100).astype("timedelta64[ms]")
    depth = 8.5 + np.arange(n_r) * 0.2
    sv = np.full((2, n_p, n_r), -80.0, dtype=np.float16)
    sv[:, :, depth > 60] = -70.0
    mask = np.zeros((2, n_p, n_r), dtype=np.uint8)
    mask[:, :, depth < 13.5] = 2  # surface bit
    return TransectPatches(sv, mask, ping_time, depth, time_step_s=2.0, depth_step_m=0.5)


def test_patch_shapes_and_values() -> None:
    tp = _store()
    start = np.datetime64("2018-05-09T20:01:00")
    end = start + np.timedelta64(60, "s")
    patch = tp.patch(start, end, top_m=40.0, height_m=50.0, n_time=30, n_depth=100)
    assert patch["sv_db"].shape == (2, 30, 100) and patch["valid"].shape == (2, 30, 100)
    assert abs(float(patch["sv_db"][0, 0, 0]) + 80) < 0.1 and abs(float(patch["sv_db"][0, 0, 60]) + 70) < 0.1
    assert patch["depth_cells"][0] == 40.25 and 54 <= patch["n_pings"] <= 56
    shallow = tp.patch(start, end, top_m=5.0, height_m=50.0, n_time=30, n_depth=100)
    assert shallow["valid"][0, :, :10].max() == 0  # cells inside the surface exclusion are not valid
    assert band_top(12.0, 50.0, "surface", 13.5) == 13.5 and band_top(100.0, 50.0, "surface", 13.5) == 75.0
    late = tp.patch(
        np.datetime64("2018-05-09T20:07:00"), np.datetime64("2018-05-09T20:08:00"), 40.0, 50.0, 30, 100
    )
    assert late is not None and late["n_pings"] < 54 and late["valid"][0, -1, :].max() == 0  # padded tail
    geometry = Geometry.from_config(
        "surface", {"length_s": 60, "height_m": 50, "stride_s": 20, "anchor": "surface"}
    )
    assert geometry.anchor == "surface" and geometry.context is None


def _mapping(n: int = 120, lag_s: float = 300.0) -> pd.DataFrame:
    t = pd.date_range("2018-05-09T20:00:00", periods=n, freq="10s")
    return pd.DataFrame(
        {
            "time_camera": t,
            "time_vessel": t - pd.Timedelta(seconds=lag_s),
            "camera_depth_m": np.linspace(10, 100, n),
            "lag_s": lag_s,
            "lag_sd_s": 20.0,
            "camera_in_exclusion_38000": np.linspace(10, 100, n) < 13.5,
            "band_usable_fraction_38000": 1.0,
        }
    )


def test_window_grid_and_targets() -> None:
    mapping = _mapping()
    grid = window_grid(mapping, stride_s=20, length_s=60)
    assert len(grid) > 50 and grid["n_intervals"].iloc[0] == 6
    assert (grid["time_camera"] - grid["time_vessel"] == pd.Timedelta(seconds=300)).all()
    series = pd.DataFrame(
        {"time": mapping["time_camera"], "krill_res": 1.0, "herring_res": 0.0, "n_frames": 49}
    )
    series.loc[series.index[:30], "herring_res"] = 2.0
    sums = interval_sums(series, grid["time_camera"].to_numpy()[:1], 60.0, ["krill_res"])
    assert sums[0, 0] == 6.0
    targets, record = build_targets(
        series,
        grid,
        60.0,
        {"hard": "_res"},
        ["krill", "herring"],
        merged={"both": ["krill", "herring"]},
        lag_offsets_sd=(-1.0, 0.0, 1.0),
    )
    assert record["distribution"] == "normal" and len(record["weights"]) == 3
    assert np.allclose(targets["hard_krill_at_lag"], 6.0)
    assert set(targets.columns) >= {"hard_krill_k0", "hard_both", "presence_herring", "frame_fraction"}
    assert np.allclose(
        targets["hard_both_at_lag"], targets["hard_krill_at_lag"] + targets["hard_herring_at_lag"]
    )
    assert np.isclose(targets["frame_fraction"].iloc[0], 6 * 49 / (4.9 * 60))
    logged = log1p_targets(targets, ("hard_",))
    assert "log1p_hard_krill" in logged.columns


def _meta() -> pd.DataFrame:
    rows = []
    wid = 0
    for transect in ("A", "B", "C"):
        t0 = np.datetime64("2018-05-05T06:00:00") if transect == "A" else np.datetime64("2018-05-09T00:00:00")
        for k in range(200):
            rows.append(
                {
                    "window_id": wid,
                    "transect": transect,
                    "time_vessel": t0 + np.timedelta64(20 * k, "s"),
                    "length_s": 60.0,
                    "lag_sd_s": 20.0,
                }
            )
            wid += 1
    return pd.DataFrame(rows)


def test_splits_have_no_leakage() -> None:
    meta = _meta()
    buffer = buffer_seconds(meta, 2.0)
    assert buffer == 100.0
    loto = loto_folds(meta, val_fraction=0.2, buffer_s=buffer)
    assert set(loto["fold"]) == {"test_A", "test_B", "test_C"}
    fold = loto[loto["fold"] == "test_A"]
    assert (fold[fold["window_id"].isin(meta[meta["transect"] == "A"]["window_id"])]["split"] == "test").all()
    assert check_splits(meta, loto, buffer).empty
    blocks = block_splits(
        meta, block_s=600, pattern=["train", "train", "val", "train", "train", "test"], buffer_s=buffer
    )
    assert check_splits(meta, blocks, buffer).empty
    counts = split_counts(blocks, meta)
    assert set(counts["split"]) >= {"train", "val", "test", "dropped"}
    # A deliberately leaky assignment is caught.
    leaky = blocks.copy()
    leaky.loc[leaky.index[:3], "split"] = "test"
    leaky.loc[leaky.index[3:6], "split"] = "train"
    assert not check_splits(meta, leaky, buffer).empty


def test_targets_uniform_and_fixed() -> None:
    mapping = _mapping()
    mapping["lag_lo_s"] = 200.0
    mapping["lag_hi_s"] = 400.0
    mapping["lag_distribution"] = "uniform"
    grid = window_grid(mapping, stride_s=20, length_s=60)
    assert grid["lag_distribution"].iloc[0] == "uniform" and grid["lag_hi_s"].iloc[0] == 400.0
    series = pd.DataFrame({"time": mapping["time_camera"], "krill_res": 1.0, "n_frames": 49})
    uniform, record = build_targets(
        series, grid, 60.0, {"hard": "_res"}, ["krill"], distribution="uniform", n_uniform_offsets=5
    )
    assert record["distribution"] == "uniform" and np.allclose(record["weights"], 0.2)
    # A constant series: every offset and the exact average give one object per interval away from the ends.
    inner = slice(15, -15)
    assert np.allclose(uniform["hard_krill_k0"].iloc[inner], 6.0)
    assert np.allclose(uniform["hard_krill"].iloc[inner], 6.0)
    # A step in the series: the exact average over the 200 s support is the share of intervals after the step.
    series["krill_res"] = (series["time"] >= series["time"].iloc[40]).astype(float)
    uniform, _ = build_targets(
        series, grid, 60.0, {"hard": "_res"}, ["krill"], distribution="uniform", n_uniform_offsets=5
    )
    assert 0.0 < uniform["hard_krill"].iloc[18] < 6.0  # support 290 to 490 s straddles the step at 400 s
    fixed, record = build_targets(series, grid, 60.0, {"hard": "_res"}, ["krill"], distribution="fixed")
    assert record["distribution"] == "fixed" and "hard_krill_k1" not in fixed.columns
    assert np.allclose(fixed["hard_krill"], fixed["hard_krill_at_lag"])
    # The buffer covers the whole support when the mapping carries one.
    meta = pd.DataFrame(
        {
            "length_s": [60.0],
            "lag_sd_s": [30.0],
            "lag_lo_s": [200.0],
            "lag_hi_s": [400.0],
            "target_distribution": ["uniform"],
        }
    )
    assert buffer_seconds(meta, 2.0) == 260.0
    assert buffer_seconds(meta.assign(target_distribution="normal"), 2.0) == 120.0
    assert buffer_seconds(meta.assign(target_distribution="fixed"), 2.0) == 60.0
    assert buffer_seconds(meta.drop(columns=["target_distribution"]), 2.0) == 120.0


def test_buffers_per_transect_and_protocol_counts() -> None:
    meta = _meta()
    meta["lag_sd_s"] = np.where(meta["transect"] == "A", 30.0, 100.0)
    meta["target_distribution"] = "normal"
    buffers = buffer_by_transect(meta, 2.0)
    assert buffers["A"] == meta["length_s"].max() + 60.0 and buffers["B"] == meta["length_s"].max() + 200.0
    loto = loto_folds(meta, 0.3, buffers, val_position="middle")
    loto["protocol"] = "loto_all"
    # The validation block sits in the middle of each training transect: training on both sides.
    a_rows = loto[loto["fold"] == "test_C"].merge(meta[["window_id", "transect", "time_vessel"]])
    a_rows = a_rows[a_rows["transect"] == "A"].sort_values("time_vessel")
    first_val = a_rows.index[a_rows["split"] == "val"][0]
    assert (a_rows.loc[:first_val, "split"] == "train").any()
    assert (a_rows.loc[first_val:, "split"] == "train").any()
    assert check_splits(meta, loto, buffers).empty
    # The transect with the smaller buffer keeps more validation windows.
    val = loto[(loto["fold"] == "test_C") & (loto["split"] == "val")].merge(meta[["window_id", "transect"]])
    assert (val["transect"] == "A").sum() > (val["transect"] == "B").sum()
    counts = split_counts(loto, meta)
    assert "protocol" in counts.columns and set(counts["split"]) >= {"train", "val", "test"}


def test_buffer_per_window_uniform_support() -> None:
    meta = _meta()
    meta["lag_sd_s"] = 50.0
    meta["target_distribution"] = "uniform"
    meta["lag_lo_s"] = 200.0
    # One transect has a wide support in a few windows only; the buffer follows the window, not the transect.
    meta["lag_hi_s"] = np.where((meta["transect"] == "B") & (np.arange(len(meta)) % 50 == 0), 2000.0, 500.0)
    buffers = buffer_by_window(meta, 2.0)
    assert buffers.loc[meta.index[0]] == meta["length_s"].iloc[0] + 300.0
    assert (buffers == meta["length_s"] + 300.0).mean() > 0.95 and buffers.max() > 1800.0
    assert buffer_seconds(meta, 2.0) == meta["length_s"].iloc[0] + 300.0  # median, not maximum
    loto = loto_folds(meta, 0.3, buffers)
    loto["protocol"] = "loto_all"
    assert check_splits(meta, loto, buffers).empty
    assert (loto["split"] == "val").sum() > 0


def test_frame_statistics_and_frame_forms() -> None:
    from echolabel.windows.targets import build_targets, frame_statistics

    # Five frames per second for two minutes; krill in every fourth frame, three per frame.
    t0 = np.datetime64("2018-05-09T20:00:00")
    stamps = t0 + (np.arange(600) * 200).astype("timedelta64[ms]")
    krill = np.where(np.arange(600) % 4 == 0, 3.0, 0.0)
    frames = pd.DataFrame({"timestamp": stamps, "krill_res": krill, "herring_res": 1.0})
    centre = np.array([t0 + np.timedelta64(60, "s")])
    stats = frame_statistics(frames.rename(columns={"krill_res": "krill"}), centre, 60.0, ["krill"])
    assert stats["n_frames"][0] == 300
    assert np.isclose(stats["framemean"][0, 0], 0.75) and stats["framemedian"][0, 0] == 0.0
    assert np.isclose(stats["occupancy"][0, 0], 0.25) and stats["occupied"][0, 0] == 75
    # A span without frames gives zeros.
    empty = frame_statistics(
        frames.rename(columns={"krill_res": "krill"}), centre + np.timedelta64(1, "h"), 60.0, ["krill"]
    )
    assert empty["n_frames"][0] == 0 and empty["occupancy"][0, 0] == 0.0
    series = pd.DataFrame(
        {
            "time": pd.date_range("2018-05-09T20:00:00", periods=12, freq="10s"),
            "krill_res": 37.5,
            "herring_res": 50.0,
        }
    )
    grid = pd.DataFrame({"time_camera": centre, "lag_sd_s": [5.0]})
    targets, _ = build_targets(
        series,
        grid,
        60.0,
        {"hard": "_res"},
        ["herring", "krill"],
        {"fish": ["herring"]},
        frames=frames,
        frame_forms=("framemean", "framemedian", "occupancy", "occupied"),
    )
    assert np.isclose(targets["occupancy_krill_at_lag"].iloc[0], 0.25)
    assert (
        np.isclose(targets["occupancy_herring"].iloc[0], 1.0)
        and targets["framemedian_fish_at_lag"].iloc[0] == 1.0
    )
    # The mean per frame times the frames of the span is the sum target.
    assert np.isclose(targets["framemean_krill_at_lag"].iloc[0] * 300, targets["hard_krill_at_lag"].iloc[0])
    assert "occupancy_krill_k0" in targets.columns and "occupancy_krill_k4" in targets.columns
    assert (
        targets["occupied_krill_at_lag"].iloc[0] == 75 and targets["occupied_herring_at_lag"].iloc[0] == 300
    )


def test_patch_rows_keep_their_depth_above_the_first_sample() -> None:
    # Range samples every 0.2 m from the transducer at 8.5 m; a layer of -60 dB from 30 m down.
    n_p, n_r = 80, 600
    ping_time = np.datetime64("2018-05-09T20:00:00") + (np.arange(n_p) * 1000).astype("timedelta64[ms]")
    depth = 8.5 + np.arange(n_r) * 0.2
    sv = np.where(depth >= 30.0, -60.0, -80.0)[None, None, :] * np.ones((1, n_p, 1))
    patches = TransectPatches(
        sv.astype(np.float32), np.zeros((1, n_p, n_r), dtype=np.uint8), ping_time, depth, 2.0, 1.0
    )
    start = np.datetime64("2018-05-09T20:00:00")
    patch = patches.patch(
        start, start + np.timedelta64(60, "s"), top_m=-5.0, height_m=50.0, n_time=30, n_depth=50
    )
    cells = patch["depth_cells"]
    assert cells[0] == -4.5 and cells[-1] == 44.5
    column = patch["sv_db"][0, 0].astype(float)
    # Rows above the transducer are missing, and the layer starts in the row that holds 30 m.
    assert np.isnan(column[cells < 8.0]).all() and (patch["valid"][0, 0][cells < 8.0] == 0).all()
    assert abs(column[cells == 29.5][0] + 80) < 0.2 and abs(column[cells == 31.5][0] + 60) < 0.2
    # A band inside the sampled range is cut as before.
    deep = patches.patch(
        start, start + np.timedelta64(60, "s"), top_m=20.0, height_m=50.0, n_time=30, n_depth=50
    )
    assert abs(float(deep["sv_db"][0, 0, 9]) + 80) < 0.2 and abs(float(deep["sv_db"][0, 0, 11]) + 60) < 0.2


def test_patch_time_cells_follow_the_clock_not_the_ping_count() -> None:
    # One ping every 1.5 s; the level rises from -80 to -60 dB at 50 s into the window.
    n_p, n_r = 60, 300
    start = np.datetime64("2018-05-09T20:00:00")
    ping_time = start + (np.arange(n_p) * 1500).astype("timedelta64[ms]")
    depth = 8.5 + np.arange(n_r) * 0.2
    seconds = np.arange(n_p) * 1.5
    sv = np.where(seconds >= 50.0, -60.0, -80.0)[None, :, None] * np.ones((1, 1, n_r))
    patches = TransectPatches(
        sv.astype(np.float32), np.zeros((1, n_p, n_r), dtype=np.uint8), ping_time, depth, 2.0, 1.0
    )
    patch = patches.patch(
        start, start + np.timedelta64(60, "s"), top_m=20.0, height_m=20.0, n_time=30, n_depth=20
    )
    row = patch["sv_db"][0, :, 5].astype(float)
    # The whole minute is in the patch: 40 pings, and the rise sits in the cell that starts at 50 s.
    assert patch["n_pings"] == 40 and patch["sv_db"].shape == (1, 30, 20)
    assert np.allclose(row[:24], -80.0, atol=0.2) and np.allclose(row[26:], -60.0, atol=0.2)
    assert (patch["valid"][0] == 255).all()
    # A window that runs past the last ping has missing cells at its end only.
    late = patches.patch(
        start + np.timedelta64(60, "s"), start + np.timedelta64(120, "s"), 20.0, 20.0, 30, 20
    )
    assert late["n_pings"] == 20 and np.isfinite(late["sv_db"][0, :14, 5].astype(float)).all()
    assert np.isnan(late["sv_db"][0, 15:, 5].astype(float)).all() and (late["valid"][0, 15:] == 0).all()
