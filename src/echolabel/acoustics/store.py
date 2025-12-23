"""Access to the volume backscattering store.

The store is a Zarr group with the main array ``sv`` of shape
(frequency, ping_time, range): linear volume backscattering coefficient
(1/m) per acoustic channel, ping and range sample. Channels are always
selected by their frequency value in hertz, never by position, because the
frequency coordinate is not stored in ascending order.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import xarray as xr

CHANNEL_FREQ_RE = re.compile(r"\bES(\d+)")


def open_sv_store(path: str | Path, chunks: dict | None = None) -> xr.Dataset:
    """Open the store lazily with dask, keeping the on-disk chunking by default."""
    return xr.open_zarr(str(path), consolidated=True, chunks={} if chunks is None else chunks)


def channel_frequency_hz(channel_id: str) -> float | None:
    """Frequency in hertz implied by a transducer name such as ``ES38-7`` (kilohertz in the name)."""
    m = CHANNEL_FREQ_RE.search(channel_id)
    return float(m.group(1)) * 1000.0 if m else None


def channel_frequency_table(ds: xr.Dataset) -> pd.DataFrame:
    """Frequency coordinate next to the frequency parsed from ``channel_id``, with an agreement flag."""
    rows = []
    for pos, (f, cid) in enumerate(zip(ds["frequency"].values, ds["channel_id"].values, strict=True)):
        parsed = channel_frequency_hz(str(cid))
        rows.append(
            {
                "position": pos,
                "frequency_hz": float(f),
                "channel_id": str(cid),
                "parsed_hz": parsed,
                "agrees": parsed is not None and abs(parsed - float(f)) < 0.5,
            }
        )
    return pd.DataFrame(rows)


def frequency_position(ds: xr.Dataset, frequency_hz: float, tolerance_hz: float = 1.0) -> int:
    """Index along the frequency dimension of the channel closest to ``frequency_hz``."""
    freqs = ds["frequency"].values
    diffs = abs(freqs - frequency_hz)
    pos = int(diffs.argmin())
    if diffs[pos] > tolerance_hz:
        raise KeyError(
            f"no channel within {tolerance_hz} Hz of {frequency_hz} Hz; available: {freqs.tolist()}"
        )
    return pos


def select_frequency(da: xr.DataArray, frequency_hz: float, tolerance_hz: float = 1.0) -> xr.DataArray:
    """Select one channel by frequency value.

    Positional selection is used because label-based nearest lookup requires a
    sorted coordinate, which the store does not provide.
    """
    freqs = da["frequency"].values
    diffs = abs(freqs - frequency_hz)
    pos = int(diffs.argmin())
    if diffs[pos] > tolerance_hz:
        raise KeyError(
            f"no channel within {tolerance_hz} Hz of {frequency_hz} Hz; available: {freqs.tolist()}"
        )
    return da.isel(frequency=pos)
