"""Camera transects: frame listings, Deep Vision metadata and their join.

A transect directory holds ``Left/`` and ``Right/`` camera trees with
time-stamped subfolders of JPEG frames (some deployments store a subfolder as a
zip archive), a calibration file and ``deepvision.xml`` with one record per
frame. Frame names and metadata records both carry a UTC timestamp with
millisecond resolution, ``YYYYMMDDhhmmssSSS``, which is the join key.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree

log = logging.getLogger(__name__)

FRAME_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{3})")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
NUMERIC_FRAME_FIELDS = ("depth", "rawdepth", "voltage", "temp", "pitch", "roll", "lat", "lon", "length")
FRAME_COLUMNS = [
    "folder",
    "kind",
    "is_zip",
    "filename",
    "size_bytes",
    "is_image",
    "timestamp",
    "archive_duplicate",
]


def parse_frame_timestamp(name: str) -> np.datetime64 | None:
    """Timestamp encoded at the start of a frame name, as ``datetime64[ms]``, or None."""
    m = FRAME_TS_RE.match(PurePosixPath(name).stem)
    if m is None:
        return None
    y, mo, d, h, mi, s, ms = m.groups()
    try:
        return np.datetime64(f"{y}-{mo}-{d}T{h}:{mi}:{s}.{ms}", "ms")
    except ValueError:
        return None


def classify_folder(name: str) -> tuple[str, bool]:
    """Return (kind, is_zip) where kind is ``active``, ``empty`` or ``other``."""
    is_zip = name.lower().endswith(".zip")
    base = name[:-4] if is_zip else name
    if base.endswith("_active"):
        return "active", is_zip
    if base.endswith("_empty"):
        return "empty", is_zip
    return "other", is_zip


def list_camera_frames(transect_dir: str | Path, camera: str = "Left") -> pd.DataFrame:
    """List every file under one camera tree, including entries of zip archives.

    Columns: folder, kind, is_zip, filename, size_bytes, is_image, timestamp,
    archive_duplicate. Zip archives are listed from their central directory
    without extraction; entries of an archive whose extracted folder also
    exists are flagged as ``archive_duplicate``.
    """
    cam_dir = Path(transect_dir, camera)
    rows: list[tuple[str, str, bool, str, int]] = []
    if not cam_dir.is_dir():
        return pd.DataFrame(columns=FRAME_COLUMNS)
    for entry in sorted(os.scandir(cam_dir), key=lambda e: e.name):
        kind, is_zip = classify_folder(entry.name)
        if entry.is_dir():
            with os.scandir(entry.path) as it:
                for f in it:
                    if f.is_file():
                        rows.append((entry.name, kind, False, f.name, f.stat().st_size))
        elif is_zip and entry.is_file():
            try:
                with zipfile.ZipFile(entry.path) as zf:
                    for info in zf.infolist():
                        if not info.is_dir():
                            rows.append(
                                (entry.name, kind, True, PurePosixPath(info.filename).name, info.file_size)
                            )
            except zipfile.BadZipFile:
                log.warning("bad zip archive %s", entry.path)
                rows.append((entry.name, kind, True, "", 0))
        else:
            rows.append(
                (entry.name, "other", False, entry.name, entry.stat().st_size if entry.is_file() else 0)
            )
    df = pd.DataFrame(rows, columns=["folder", "kind", "is_zip", "filename", "size_bytes"])
    df["is_image"] = df["filename"].str.lower().str.endswith(IMAGE_SUFFIXES)
    df["timestamp"] = pd.to_datetime([parse_frame_timestamp(f) for f in df["filename"]], errors="coerce")
    extracted = set(df.loc[~df["is_zip"], "folder"])
    df["archive_duplicate"] = df["is_zip"] & df["folder"].str.replace(r"\.zip$", "", regex=True).isin(
        extracted
    )
    return df


def parse_deepvision_xml(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read ``deepvision.xml`` into a frame table and an information dictionary.

    Every frame attribute is kept as a string column; numeric fields listed in
    ``NUMERIC_FRAME_FIELDS`` are additionally coerced to float in a column with
    the suffix ``_num``, with invalid text becoming NaN. Some deployments wrote
    numbers with a decimal comma; the comma is replaced before conversion and
    the number of such values is reported in ``info["decimal_comma_values"]``.
    ``timestamp`` is parsed from the ``time`` attribute.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=True)
    root = etree.parse(str(path), parser).getroot()
    info: dict[str, Any] = {"groups": []}
    if root is not None:
        for group in root.iter("group"):
            items = {item.get("name", ""): (item.text or "").strip() for item in group.findall("item")}
            info["groups"].append({"name": group.get("name", ""), "items": items})
        frames_el = root.find("frames")
        info["fileformat"] = frames_el.get("fileformat") if frames_el is not None else None
        rows = [dict(frame.attrib) for frame in root.iter("frame")]
    else:
        rows = []
    df = pd.DataFrame(rows)
    info["n_frames"] = len(df)
    info["attributes"] = list(df.columns)
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime([parse_frame_timestamp(str(t)) for t in df["time"]], errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    info["decimal_comma_values"] = {}
    for col in NUMERIC_FRAME_FIELDS:
        if col in df.columns:
            raw = df[col].astype(str)
            n_comma = int(raw.str.contains(",", regex=False).sum())
            if n_comma:
                info["decimal_comma_values"][col] = n_comma
            df[col + "_num"] = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
    if "active" in df.columns:
        df["active_flag"] = df["active"].str.lower().map({"true": True, "false": False})
    return df, info


def join_frames_to_metadata(
    frames: pd.DataFrame,
    meta: pd.DataFrame,
    tolerance_ms: int = 50,
) -> pd.DataFrame:
    """Attach the nearest metadata record to every frame by timestamp.

    Frames without a record within ``tolerance_ms`` get ``matched == False``.
    ``folder_agrees`` compares the frame's folder (zip suffix removed) with the
    folder recorded in the metadata. Metadata columns are prefixed ``meta_``.
    """
    f = frames.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    m = meta.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    m = m.add_prefix("meta_").rename(columns={"meta_timestamp": "timestamp"})
    m["meta_row"] = np.arange(len(m))
    joined = pd.merge_asof(
        f, m, on="timestamp", direction="nearest", tolerance=pd.Timedelta(milliseconds=tolerance_ms)
    )
    joined["matched"] = joined["meta_row"].notna()
    folder_plain = joined["folder"].str.replace(r"\.zip$", "", regex=True)
    if "meta_folder" in joined.columns:
        joined["folder_agrees"] = joined["matched"] & (folder_plain == joined["meta_folder"])
    else:
        joined["folder_agrees"] = False
    return joined


def summarize_timestamps(
    ts: pd.Series | np.ndarray | pd.DatetimeIndex, gap_thresholds_s: tuple[float, ...] = (1, 5, 60)
) -> dict[str, Any]:
    """Interval and gap statistics of a timestamp sequence in its given order."""
    t = pd.to_datetime(pd.Series(ts)).dropna()
    out: dict[str, Any] = {"n": int(len(t)), "n_missing_timestamp": int(len(ts) - len(t))}
    if len(t) < 2:
        return out
    ordered_diff = t.diff().dt.total_seconds().to_numpy()[1:]
    out["n_nonmonotonic_steps"] = int((ordered_diff < 0).sum())
    out["n_duplicate_timestamps"] = int(t.duplicated().sum())
    t_sorted = t.sort_values()
    d = t_sorted.diff().dt.total_seconds().to_numpy()[1:]
    out["first"] = str(t_sorted.iloc[0])
    out["last"] = str(t_sorted.iloc[-1])
    out["span_s"] = float((t_sorted.iloc[-1] - t_sorted.iloc[0]).total_seconds())
    out["interval_s"] = {f"p{q}": float(np.percentile(d, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}
    out["gaps"] = {str(thr): int((d > thr).sum()) for thr in gap_thresholds_s}
    order = np.argsort(d)[::-1][:10]
    out["largest_gaps"] = [
        {"start": str(t_sorted.iloc[i]), "end": str(t_sorted.iloc[i + 1]), "seconds": float(d[i])}
        for i in order
    ]
    return out
