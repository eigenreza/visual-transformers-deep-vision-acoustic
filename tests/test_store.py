from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from echolabel.acoustics.store import (
    channel_frequency_hz,
    channel_frequency_table,
    frequency_position,
    select_frequency,
)


def _dataset() -> xr.Dataset:
    freqs = np.array([38000.0, 18000.0, 70000.0])
    sv = xr.DataArray(
        np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5), dims=("frequency", "ping_time", "range")
    )
    return xr.Dataset(
        {"sv": sv},
        coords={
            "frequency": freqs,
            "channel_id": ("frequency", np.array(["WBT 1 ES38-7", "WBT 2 ES18", "WBT 3 ES70-7C"])),
        },
    )


def test_channel_frequency_parsing() -> None:
    assert channel_frequency_hz("WBT 714596-15 ES38-7") == 38000.0
    assert channel_frequency_hz("WBT 714581-15 ES18") == 18000.0
    assert channel_frequency_hz("WBT 714597-15 ES333-7C") == 333000.0
    assert channel_frequency_hz("no name") is None


def test_frequency_selection_ignores_order() -> None:
    ds = _dataset()
    assert frequency_position(ds, 18000.0) == 1
    assert frequency_position(ds, 38000.0) == 0
    with pytest.raises(KeyError):
        frequency_position(ds, 120000.0)
    sel = select_frequency(ds["sv"], 70000.0)
    assert float(sel[0, 0]) == 40.0
    table = channel_frequency_table(ds)
    assert table["agrees"].all()
