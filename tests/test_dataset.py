from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from echolabel.data.coco import to_coco
from echolabel.data.splits import (
    assign_splits,
    check_split_properties,
    class_counts_by_split,
    tow_table,
)
from echolabel.data.taxonomy import Taxonomy, apply_taxonomy, mapping_table

REPO = Path(__file__).resolve().parents[1]
TAXONOMY = REPO / "configs" / "taxonomy.yaml"
AUDIT_LABELS = REPO / "results" / "audit" / "annotations" / "labels_by_year.csv"


def test_taxonomy_covers_audited_labels() -> None:
    tax = Taxonomy.from_yaml(TAXONOMY)
    assert len(tax.classes) == 14
    assert tax.classes[-1] == "unidentified"
    assert tax.class_id("maurolicus") == 1
    assert tax.map_label("Krill") == "krill"
    assert tax.map_label("periphylla") == "gelatinous"
    assert tax.map_label("cod") == "unidentified"
    with pytest.raises(KeyError):
        tax.map_label("dragon")
    if AUDIT_LABELS.is_file():
        labels = pd.read_csv(AUDIT_LABELS)["label_raw"].astype(str).tolist()
        assert tax.unmapped(labels) == []


def test_apply_taxonomy_and_mapping_table() -> None:
    tax = Taxonomy.from_yaml(TAXONOMY)
    boxes = pd.DataFrame(
        {"key": ["a", "a", "b"], "year": ["2018", "2018", "2020"], "label_raw": ["krill", "Krill", "UID"]}
    )
    out = apply_taxonomy(boxes, tax)
    assert out["class_name"].tolist() == ["krill", "krill", "unidentified"]
    assert out["class_id"].tolist() == [2, 2, 14]
    table = mapping_table(out, tax)
    assert set(table["label_raw"]) == {"krill", "Krill", "UID"}
    assert table.loc[table["label_raw"] == "Krill", "class_name"].item() == "krill"
    with pytest.raises(KeyError):
        apply_taxonomy(pd.DataFrame({"key": ["c"], "year": ["2018"], "label_raw": ["dragon"]}), tax)


def _synthetic(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    classes = ["krill", "herring", "gonatus", "unidentified"]
    rows, boxes = [], []
    tows = [("2017", f"ST{i:03d}", 15) for i in range(6)] + [("2022", f"2022T{i}", 300) for i in range(6)]
    tows += [("2018", "ST12", 8), ("2018", "ST5", 10), ("2024", "2024T1", 50)]
    for year, station, n in tows:
        for j in range(n):
            key = f"{year}/{station}/Left/{j}"
            n_boxes = int(rng.integers(0, 4))
            rows.append({"key": key, "year": year, "station": station, "n_boxes": n_boxes,
                         "image_rel": key + ".jpg", "width_used": 100, "height_used": 80})  # fmt: skip
            for _ in range(n_boxes):
                cls = (
                    "gonatus"
                    if station == "2022T0"
                    else str(rng.choice(["krill", "herring", "unidentified"]))
                )
                x0, y0 = rng.uniform(-5, 90), rng.uniform(0, 70)
                boxes.append(
                    {"key": key, "class_name": cls, "xmin": x0, "ymin": y0, "xmax": x0 + 12, "ymax": y0 + 9}
                )
    return pd.DataFrame(rows), pd.DataFrame(boxes), classes


def test_assign_splits_properties() -> None:
    images, boxes, classes = _synthetic()
    tows = tow_table(images, boxes, classes)
    assert tows["n_images"].sum() == len(images)
    holdout = {"2024": "test_2024"}
    forced = {"2018/ST12": "train"}
    targets = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignment = assign_splits(tows, classes, targets, forced, holdout, seed=1, n_restarts=5, n_moves=300)
    again = assign_splits(tows, classes, targets, forced, holdout, seed=1, n_restarts=5, n_moves=300)
    assert assignment.equals(again)
    assert assignment["2024/2024T1"] == "test_2024"
    assert assignment["2018/ST12"] == "train"
    images["split"] = images.apply(lambda r: assignment[f"{r.year}/{r.station}"], axis=1)
    non_hold = images[images["year"] != "2024"]
    frac = non_hold["split"].value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.85
    assert frac["val"] > 0.05 and frac["test"] > 0.05
    assert check_split_properties(images, holdout, forced, ["2018/ST12"]) == []
    counts = class_counts_by_split(images, boxes, classes)
    assert counts.loc[counts["class_name"] == "all_images", "images_train"].item() == int(
        (images["split"] == "train").sum()
    )
    # Violations are detected.
    broken = images.copy()
    broken.loc[broken["station"] == "ST12", "split"] = "test"
    problems = check_split_properties(broken, holdout, forced, ["2018/ST12"])
    assert any("forced" in p for p in problems) and any("protected" in p for p in problems)
    broken2 = images.copy()
    broken2.loc[broken2.index[0], "split"] = "test_2024"
    assert any("outside year" in p for p in check_split_properties(broken2, holdout, forced, []))


def test_coco_export_clips_and_keeps_negatives() -> None:
    tax = Taxonomy.from_yaml(TAXONOMY)
    images = pd.DataFrame({
        "key": ["a", "b"], "image_rel": ["2018/ST1/Left/a.jpg", "2018/ST1/Left/b.jpg"],
        "width_used": [100, 100], "height_used": [80, 80],
    })  # fmt: skip
    boxes = pd.DataFrame({
        "key": ["a", "a", "a"], "class_name": ["krill", "herring", "krill"],
        "xmin": [-5.0, 10.0, 50.0], "ymin": [0.0, 10.0, 50.0],
        "xmax": [20.0, 30.0, 51.0], "ymax": [10.0, 30.0, 60.0],
    })  # fmt: skip
    coco, stats = to_coco(images, boxes, tax, clip_boxes=True, min_box_side_px=2)
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
    assert stats == {"clipped": 1, "dropped": 1}
    first = coco["annotations"][0]
    assert first["bbox"] == [0.0, 0.0, 20.0, 10.0]
    assert first["category_id"] == tax.class_id("krill")
    assert [c["name"] for c in coco["categories"]] == tax.classes


def test_assign_splits_minimum_tows_per_year() -> None:
    images, boxes, classes = _synthetic()
    tows = tow_table(images, boxes, classes)
    targets = {"train": 0.7, "val": 0.15, "test": 0.15}
    minima = {"2017": {"val": 2, "test": 2}}
    assignment = assign_splits(
        tows,
        classes,
        targets,
        {},
        {"2024": "test_2024"},
        seed=2,
        n_restarts=5,
        n_moves=400,
        min_tows_per_split=minima,
    )
    per_split = tows.assign(split=tows["tow"].map(assignment))
    counts = per_split[per_split["year"] == "2017"]["split"].value_counts()
    assert counts.get("val", 0) >= 2 and counts.get("test", 0) >= 2
