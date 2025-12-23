from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from echolabel.data.transects import (
    classify_folder,
    join_frames_to_metadata,
    list_camera_frames,
    parse_deepvision_xml,
    parse_frame_timestamp,
    summarize_timestamps,
)

DV_XML = """<?xml version="1.0" encoding="utf-8"?>
<deepvision xmlns:ns="http://www.deepvision.no/formats/deepvision/v1">
  <info><group name="System"><item name="Serial">DV0001</item></group></info>
  <frames fileformat="jpg">
    <frame time="20180509193833601" depth="33,5" folder="193833Z_active" active="true" />
    <frame time="20180509193833959" depth="abc" folder="193833Z_active" active="true" />
    <frame time="20180509193834162" depth="19.2" folder="193833Z_active" active="true" />
    <frame time="20180509193900000" depth="20.0" folder="193833Z_empty" active="false" />
  </frames>
</deepvision>"""


def test_parse_frame_timestamp() -> None:
    ts = parse_frame_timestamp("20180509193833601.jpg")
    assert ts == np.datetime64("2018-05-09T19:38:33.601", "ms")
    assert parse_frame_timestamp("calib") is None
    assert parse_frame_timestamp("20189999193833601.jpg") is None


def test_classify_folder() -> None:
    assert classify_folder("193833Z_active") == ("active", False)
    assert classify_folder("155025Z_active.zip") == ("active", True)
    assert classify_folder("155025Z_empty") == ("empty", False)
    assert classify_folder("misc") == ("other", False)


def test_list_and_join(tmp_path: Path) -> None:
    left = tmp_path / "T" / "Left"
    active = left / "193833Z_active"
    active.mkdir(parents=True)
    for name in (
        "20180509193833601.jpg",
        "20180509193833959.jpg",
        "20180509193834162.jpg",
        "20180509193835000.jpg",
    ):
        (active / name).write_bytes(b"x" * 10)
    (left / "193833Z_empty").mkdir()
    with zipfile.ZipFile(left / "194519Z_active.zip", "w") as zf:
        zf.writestr("194519Z_active/20180509194519000.jpg", b"y" * 5)
    with zipfile.ZipFile(left / "193833Z_active.zip", "w") as zf:
        zf.writestr("193833Z_active/20180509193833601.jpg", b"x" * 10)
    (tmp_path / "T" / "deepvision.xml").write_text(DV_XML, encoding="utf-8")

    frames = list_camera_frames(tmp_path / "T", "Left")
    assert len(frames) == 6
    assert frames["is_zip"].sum() == 2
    assert frames["archive_duplicate"].sum() == 1
    assert frames.loc[frames["archive_duplicate"], "folder"].iloc[0] == "193833Z_active.zip"
    frames = frames[~frames["archive_duplicate"]]
    assert set(frames["kind"]) == {"active"}
    assert frames["timestamp"].notna().all()

    meta, info = parse_deepvision_xml(tmp_path / "T" / "deepvision.xml")
    assert info["n_frames"] == 4
    assert info["fileformat"] == "jpg"
    assert meta["depth_num"].isna().sum() == 1
    assert meta["depth_num"].iloc[0] == 33.5
    assert info["decimal_comma_values"] == {"depth": 1}
    assert meta["active_flag"].tolist() == [True, True, True, False]

    joined = join_frames_to_metadata(frames[frames["is_image"]], meta, tolerance_ms=50)
    assert joined["matched"].sum() == 3
    unmatched = joined.loc[~joined["matched"], "filename"].tolist()
    assert sorted(unmatched) == ["20180509193835000.jpg", "20180509194519000.jpg"]
    assert joined.loc[joined["matched"], "folder_agrees"].all()


def test_summarize_timestamps() -> None:
    ts = pd.to_datetime(
        [
            "2018-05-09T19:38:33.601",
            "2018-05-09T19:38:34.000",
            "2018-05-09T19:38:33.800",
            "2018-05-09T19:40:00.000",
        ]
    )
    s = summarize_timestamps(ts, (1, 5, 60))
    assert s["n"] == 4
    assert s["n_nonmonotonic_steps"] == 1
    assert s["gaps"]["60"] == 1
    assert abs(s["interval_s"]["p50"] - 0.2) < 1e-6
