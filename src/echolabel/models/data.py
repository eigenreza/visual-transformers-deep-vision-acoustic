"""Window dataset access for the acoustic models.

A geometry's windows live in a Zarr group (``sv_db``, ``valid``, ``hard``,
``depth_cells`` and, for the two-scale geometry, ``context_sv_db`` and
``context_valid``) with metadata, targets and split assignments in Parquet
and CSV tables. This module turns a fold of a protocol into index sets,
computes the input normalisation on the training windows of that fold only,
builds the input tensor (selected channels, frequency differences, validity
channel, depth channel) and the target vector, and samples the lag offsets
of the targets when asked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FLOOR_DB = -130.0


@dataclass
class InputSpec:
    """Which channels enter the model and how."""

    frequencies_hz: list[float]
    differences: list[tuple[float, float]] = field(default_factory=list)
    include_valid: bool = True
    include_depth: bool = True
    fill_db: float = -100.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> InputSpec:
        return cls(
            frequencies_hz=[float(f) for f in cfg.get("frequencies_hz", [38000])],
            differences=[(float(a), float(b)) for a, b in cfg.get("differences", [])],
            include_valid=bool(cfg.get("include_valid", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            fill_db=float(cfg.get("fill_db", -100.0)),
        )

    @property
    def n_channels(self) -> int:
        n = len(self.frequencies_hz) + len(self.differences)
        n += len(self.frequencies_hz) if self.include_valid else 0
        n += 1 if self.include_depth else 0
        return n


@dataclass
class TargetSpec:
    """Which targets to predict."""

    form: str = "hard"
    classes: list[str] = field(default_factory=list)
    lag_offsets: str = "expected"  # "expected", "at_lag" or "sample"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TargetSpec:
        return cls(
            form=str(cfg.get("form", "hard")),
            classes=[str(c) for c in cfg.get("classes", [])],
            lag_offsets=str(cfg.get("lag_offsets", "expected")),
        )


class WindowStore:
    """Arrays, metadata, targets and splits of one geometry, read lazily."""

    def __init__(self, zarr_path: str | Path, meta_dir: str | Path) -> None:
        import zarr

        self.group = zarr.open_group(str(zarr_path), mode="r")
        self.meta = pd.read_parquet(Path(meta_dir, "meta.parquet"))
        self.targets = pd.read_parquet(Path(meta_dir, "targets.parquet"))
        offsets_path = Path(meta_dir, "lag_offsets.json")
        # Weights of the stored lag offsets, as the window stage drew them; normal weights otherwise.
        self.lag_weights: np.ndarray | None = None
        if offsets_path.is_file():
            with open(offsets_path, encoding="utf-8") as fh:
                record = json.load(fh)
            if record.get("weights"):
                self.lag_weights = np.asarray(record["weights"], dtype=float)
        self.frequencies = [float(f) for f in self.group.attrs.get("frequencies_hz", [])]
        self.classes = sorted(
            {
                c.split("_", 1)[1].rsplit("_at_lag", 1)[0]
                for c in self.targets.columns
                if c.startswith("hard_") and c.endswith("_at_lag")
            }
        )

    def __len__(self) -> int:
        return int(len(self.meta))

    def splits(self, splits_path: str | Path, protocol: str, fold: str) -> dict[str, np.ndarray]:
        """Row indices per split for one protocol and fold; windows without assignment are left out."""
        table = pd.read_parquet(splits_path)
        table = table[(table["protocol"] == protocol) & (table["fold"] == fold)].dropna(subset=["split"])
        position = pd.Series(np.arange(len(self.meta)), index=self.meta["window_id"])
        return {s: position.loc[g["window_id"].to_numpy()].to_numpy() for s, g in table.groupby("split")}

    def channel_positions(self, frequencies_hz: list[float]) -> list[int]:
        return [int(np.argmin(np.abs(np.asarray(self.frequencies) - f))) for f in frequencies_hz]

    def target_columns(self, spec: TargetSpec) -> list[str]:
        if spec.lag_offsets == "at_lag":
            return [f"{spec.form}_{c}_at_lag" for c in spec.classes]
        return [f"{spec.form}_{c}" for c in spec.classes]

    def target_offset_columns(self, spec: TargetSpec) -> list[list[str]]:
        """Per offset, the columns of every class, for lag sampling."""
        k = 0
        out = []
        while f"{spec.form}_{spec.classes[0]}_k{k}" in self.targets.columns:
            out.append([f"{spec.form}_{c}_k{k}" for c in spec.classes])
            k += 1
        return out


def normalisation_stats(
    store: WindowStore,
    train_index: np.ndarray,
    spec: InputSpec,
    max_windows: int = 2000,
    context: bool = False,
) -> dict[str, np.ndarray]:
    """Mean and standard deviation per input channel over the valid cells of a sample of training windows.

    With ``context`` the statistics are those of the coarse context arrays.
    """
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(train_index, size=min(max_windows, len(train_index)), replace=False))
    x = build_inputs(store, idx, spec, stats=None, context=context)
    valid_mask = None
    if spec.include_valid:
        n_f = len(spec.frequencies_hz)
        valid_mask = x[:, n_f + len(spec.differences) : 2 * n_f + len(spec.differences)] > 0
    means, stds = [], []
    for c in range(x.shape[1]):
        v = x[:, c]
        if valid_mask is not None and c < len(spec.frequencies_hz):
            v = v[valid_mask[:, c]]
        elif valid_mask is not None and c < len(spec.frequencies_hz) + len(spec.differences):
            a, b = spec.differences[c - len(spec.frequencies_hz)]
            pa, pb = spec.frequencies_hz.index(a), spec.frequencies_hz.index(b)
            v = v[valid_mask[:, pa] & valid_mask[:, pb]]
        v = v[np.isfinite(v)]
        means.append(float(v.mean()) if v.size else 0.0)
        stds.append(float(v.std()) if v.size and v.std() > 1e-6 else 1.0)
    return {"mean": np.asarray(means, dtype=np.float32), "std": np.asarray(stds, dtype=np.float32)}


def build_inputs(
    store: WindowStore,
    index: np.ndarray,
    spec: InputSpec,
    stats: dict[str, np.ndarray] | None,
    context: bool = False,
) -> np.ndarray:
    """Input array (N, C, T, D) for the given window rows.

    Channels, in order: the selected frequencies in dB with invalid cells set
    to ``fill_db``; the dB differences between frequency pairs (zero where
    either is invalid); the valid fraction of each selected frequency; the
    depth of each cell row in hundreds of metres. With ``stats`` the dB and
    difference channels are standardised. With ``context`` the arrays are the
    coarse full-column context of the two-scale geometry, with the same
    channel scheme.
    """
    prefix = "context_" if context else ""
    pos = store.channel_positions(spec.frequencies_hz)
    sv = store.group[f"{prefix}sv_db"].oindex[np.asarray(index), pos, :, :].astype(np.float32)
    valid = store.group[f"{prefix}valid"].oindex[np.asarray(index), pos, :, :].astype(np.float32) / 255.0
    sv = np.where(np.isfinite(sv) & (valid > 0), sv, spec.fill_db)
    channels = [sv]
    for a, b in spec.differences:
        pa, pb = spec.frequencies_hz.index(a), spec.frequencies_hz.index(b)
        both = (valid[:, pa] > 0) & (valid[:, pb] > 0)
        channels.append(np.where(both, sv[:, pa] - sv[:, pb], 0.0)[:, None])
    if spec.include_valid:
        channels.append(valid)
    if spec.include_depth:
        depth = store.group[f"{prefix}depth_cells"].oindex[np.asarray(index), :].astype(np.float32) / 100.0
        depth_channel = np.broadcast_to(depth[:, None, None, :], (len(index), 1, sv.shape[2], sv.shape[3]))
        channels.append(np.ascontiguousarray(depth_channel))
    x = np.concatenate(channels, axis=1)
    if stats is not None:
        n_std = len(spec.frequencies_hz) + len(spec.differences)
        x[:, :n_std] = (x[:, :n_std] - stats["mean"][:n_std, None, None]) / stats["std"][:n_std, None, None]
    return x


def build_targets(
    store: WindowStore, index: np.ndarray, spec: TargetSpec, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Target array (N, K) for the rows; with ``lag_offsets == "sample"`` one offset is drawn per row."""
    if spec.lag_offsets == "sample" and rng is not None:
        columns = store.target_offset_columns(spec)
        weights = (
            store.lag_weights
            if store.lag_weights is not None and len(store.lag_weights) == len(columns)
            else _normal_weights(len(columns))
        )
        choice = rng.choice(len(columns), size=len(index), p=weights)
        out = np.zeros((len(index), len(spec.classes)), dtype=np.float32)
        for k, cols in enumerate(columns):
            sel = choice == k
            if sel.any():
                out[sel] = store.targets.loc[index[sel], cols].to_numpy(dtype=np.float32)
        return out
    return store.targets.loc[index, store.target_columns(spec)].to_numpy(dtype=np.float32)


def _normal_weights(k: int) -> np.ndarray:
    from scipy.stats import norm

    z = np.linspace(-2, 2, k) if k > 1 else np.zeros(1)
    w = norm.pdf(z)
    return w / w.sum()


def window_features(store: WindowStore, index: np.ndarray, spec: InputSpec) -> pd.DataFrame:
    """Hand-crafted features per window for the baselines: level statistics per channel and differences.

    Per selected frequency: mean, 90th percentile and maximum of the valid
    cells in dB, the valid fraction, and the mean over the central third of
    the depth band; per difference pair: mean and standard deviation of the
    difference where both channels are valid; plus the mean cell depth.
    """
    x = build_inputs(
        store, index, InputSpec(spec.frequencies_hz, spec.differences, True, True, spec.fill_db), None
    )
    n_f, n_d = len(spec.frequencies_hz), len(spec.differences)
    rows = []
    for i in range(len(index)):
        row: dict[str, float] = {}
        valid = x[i, n_f + n_d : 2 * n_f + n_d] > 0
        depth = x[i, -1, 0, :]
        d0, d1 = depth.size // 3, 2 * depth.size // 3
        for c, f in enumerate(spec.frequencies_hz):
            v = x[i, c][valid[c]]
            name = f"f{int(f / 1000)}"
            row[f"{name}_mean"] = float(v.mean()) if v.size else spec.fill_db
            row[f"{name}_q90"] = float(np.quantile(v, 0.9)) if v.size else spec.fill_db
            row[f"{name}_max"] = float(v.max()) if v.size else spec.fill_db
            row[f"{name}_valid"] = float(valid[c].mean())
            centre = x[i, c][:, d0:d1][valid[c][:, d0:d1]]
            row[f"{name}_centre_mean"] = float(centre.mean()) if centre.size else spec.fill_db
        for j, (a, b) in enumerate(spec.differences):
            pa, pb = spec.frequencies_hz.index(a), spec.frequencies_hz.index(b)
            both = valid[pa] & valid[pb]
            d = x[i, n_f + j][both]
            name = f"d{int(a / 1000)}_{int(b / 1000)}"
            row[f"{name}_mean"] = float(d.mean()) if d.size else 0.0
            row[f"{name}_std"] = float(d.std()) if d.size else 0.0
        row["depth_mean"] = float(depth.mean()) * 100.0
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def context_features(store: WindowStore, index: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Properties of a window that need no echogram, as inputs of a baseline.

    ``depth`` is the camera depth of the window in metres and ``hour`` the
    hour of day of its camera time in UTC, as a decimal number.
    """
    rows = store.meta.loc[np.asarray(index)]
    out = pd.DataFrame(index=np.asarray(index))
    for name in names:
        if name == "depth":
            out["depth"] = rows["camera_depth_m"].to_numpy(dtype=float)
        elif name == "hour":
            stamp = pd.to_datetime(rows["time_camera"])
            out["hour"] = (stamp.dt.hour + stamp.dt.minute / 60.0 + stamp.dt.second / 3600.0).to_numpy()
        else:
            raise KeyError(f"unknown feature {name}")
    return out
