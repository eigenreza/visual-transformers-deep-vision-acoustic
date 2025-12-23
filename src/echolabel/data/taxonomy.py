"""Mapping of raw annotation labels to detector classes.

The taxonomy lives in a YAML file with an ordered list of class names and,
for each class, the raw labels that map to it. Class identifiers used in the
exported datasets are the one-based positions in that list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_yaml


@dataclass
class Taxonomy:
    """Ordered classes and the raw-label mapping that produces them."""

    classes: list[str]
    raw_to_class: dict[str, str]
    source: Path | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Taxonomy:
        data = load_yaml(path)
        classes = list(data["classes"])
        mapping: dict[str, str] = {}
        for cls_name, raws in data["mapping"].items():
            if cls_name not in classes:
                raise ValueError(f"mapping target {cls_name!r} is not a listed class")
            for raw in raws:
                if raw in mapping:
                    raise ValueError(f"raw label {raw!r} mapped twice ({mapping[raw]} and {cls_name})")
                mapping[str(raw)] = cls_name
        missing = [c for c in classes if c not in data["mapping"]]
        if missing:
            raise ValueError(f"classes without mapping entry: {missing}")
        return cls(classes=classes, raw_to_class=mapping, source=Path(path))

    def class_id(self, name: str) -> int:
        """One-based identifier of a class."""
        return self.classes.index(name) + 1

    def map_label(self, raw: str) -> str:
        """Class name for a raw label. Raises ``KeyError`` for an unmapped label."""
        try:
            return self.raw_to_class[raw]
        except KeyError as exc:
            raise KeyError(f"raw label {raw!r} is not in the taxonomy") from exc

    def unmapped(self, raw_labels: list[str] | pd.Series) -> list[str]:
        """Raw labels that the taxonomy does not cover."""
        return sorted({str(r) for r in raw_labels if str(r) not in self.raw_to_class})


def apply_taxonomy(boxes: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Add ``class_name`` and ``class_id`` columns to a box table.

    Raises ``KeyError`` listing every raw label that is not covered, so that a
    new label in the collection cannot pass unnoticed.
    """
    missing = taxonomy.unmapped(boxes["label_raw"])
    if missing:
        raise KeyError(f"raw labels not in taxonomy: {missing}")
    out = boxes.copy()
    out["class_name"] = out["label_raw"].map(taxonomy.raw_to_class)
    out["class_id"] = out["class_name"].map(taxonomy.class_id)
    return out


def mapping_table(boxes: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Raw label to class with the number of boxes per year, for documentation."""
    counts = boxes.groupby(["label_raw", "year"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.reset_index()
    counts.insert(1, "class_name", counts["label_raw"].map(taxonomy.raw_to_class))
    order = {c: i for i, c in enumerate(taxonomy.classes)}
    counts["order"] = counts["class_name"].map(order)
    return (
        counts.sort_values(["order", "total"], ascending=[True, False])
        .drop(columns="order")
        .reset_index(drop=True)
    )
