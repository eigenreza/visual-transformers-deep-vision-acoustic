from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from echolabel.detection.evaluate import (
    PRED_COLUMNS,
    average_precision,
    best_thresholds,
    box_iou,
    confusion_with_background,
    evaluate_predictions,
    gt_table,
    match_detections,
    precision_recall_curves,
    year_slices,
)


def _coco(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    images, annotations = [], []
    ann_id = 1
    rng = np.random.default_rng(0)
    for i in range(1, 21):
        year = "2018" if i <= 10 else "2022"
        images.append({"id": i, "file_name": f"{year}/T{i % 3}/Left/{i}.jpg", "width": 200, "height": 100})
        for _ in range(2):
            x, y = rng.uniform(0, 150), rng.uniform(0, 60)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1 + (ann_id % 2),
                    "bbox": [x, y, 30, 20],
                    "area": 600,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "krill"}, {"id": 2, "name": "herring"}],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(coco))
    gts = gt_table(coco)
    return path, gts


def _perfect_predictions(gts: pd.DataFrame) -> pd.DataFrame:
    p = gts[["image_id", "category_id", "x0", "y0", "x1", "y1"]].copy()
    p["score"] = 0.9
    return p[PRED_COLUMNS]


def test_box_iou() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    iou = box_iou(a, b)[0]
    assert abs(iou[0] - 1.0) < 1e-9 and abs(iou[1] - 25 / 175) < 1e-9 and iou[2] == 0


def test_matching_and_ap(tmp_path: Path) -> None:
    _, gts = _coco(tmp_path)
    preds = _perfect_predictions(gts)
    matched = match_detections(preds, gts)
    assert matched["tp"].all()
    n_gt = gts["category_id"].value_counts().to_dict()
    assert average_precision(matched[matched["category_id"] == 1], n_gt[1]) == 1.0
    # One false positive with the highest score lowers precision at the top of the ranking.
    fp = pd.DataFrame(
        [{"image_id": 1, "category_id": 1, "score": 0.99, "x0": 150, "y0": 70, "x1": 180, "y1": 90}]
    )
    matched2 = match_detections(pd.concat([preds, fp]), gts)
    ap = average_precision(matched2[matched2["category_id"] == 1], n_gt[1])
    assert 0.9 < ap < 1.0
    curves = precision_recall_curves(matched2, n_gt)
    row = curves[(curves["category_id"] == 1) & (curves["threshold"] == 0.95)].iloc[0]
    assert row["tp"] == 0 and row["fp"] == 1 and row["recall"] == 0.0
    best = best_thresholds(curves)
    assert set(best["category_id"]) == {1, 2}
    assert best.loc[best["category_id"] == 2, "f1"].item() == 1.0


def test_confusion_and_full_evaluation(tmp_path: Path) -> None:
    gt_path, gts = _coco(tmp_path)
    preds = _perfect_predictions(gts)
    # Mislabel the first box and drop the second one entirely.
    preds.loc[preds.index[0], "category_id"] = 3 - preds.loc[preds.index[0], "category_id"]
    preds = preds.drop(index=preds.index[1]).reset_index(drop=True)
    conf = confusion_with_background(preds, gts, [1, 2], {1: 0.5, 2: 0.5})
    assert conf.values.sum() == len(gts)
    assert conf.loc["background", "background"] == 0
    off_diagonal = conf.loc["1", "2"] + conf.loc["2", "1"]
    assert off_diagonal == 1
    assert conf["background"].sum() == 1
    result = evaluate_predictions(gt_path, preds, n_boot=20, seed=0)
    slices = result["slices"]
    assert set(slices["slice"]) >= {"all", "2018", "2022", "2020-2023"}
    overall = slices[(slices["slice"] == "all") & (slices["metric"] == "map50")].iloc[0]
    assert 0.8 < overall["value"] <= 1.0
    assert overall["ci_low"] <= overall["value"] <= overall["ci_high"]
    assert result["confusion"].shape == (3, 3)
    assert "map50" in result["coco"]
    assert result["coco"]["map50"] > 0.8


def test_year_slices() -> None:
    images = pd.DataFrame({"image_id": [1, 2, 3], "year": ["2018", "2022", "2024"]})
    s = year_slices(images)
    assert list(s["2018"]) == [1] and list(s["2020-2023"]) == [2] and len(s["all"]) == 3


def test_class_set_macro_matches_members() -> None:
    import numpy as np
    import pandas as pd

    from echolabel.detection.evaluate import bootstrap_ap_by_slice

    images = np.arange(1, 21)
    gts = pd.DataFrame(
        {"image_id": np.concatenate([images, images]), "category_id": [1] * 20 + [2] * 20, "iscrowd": 0}
    )
    matched = pd.DataFrame(
        {
            "image_id": np.concatenate([images, images]),
            "category_id": [1] * 20 + [2] * 20,
            "score": np.linspace(1.0, 0.1, 40),
            "tp": [1.0] * 20 + [1.0] * 10 + [0.0] * 10,
        }
    )
    table = bootstrap_ap_by_slice(
        matched,
        gts,
        [1, 2],
        {1: "a", 2: "b"},
        {"all": images},
        n_boot=20,
        min_gt=10,
        class_sets={"only_a": [1], "both": [1, 2]},
    )
    value = table.set_index("metric")["value"]
    assert abs(value["map50_only_a"] - value["ap50_a"]) < 1e-9
    assert abs(value["map50_both"] - value["map50"]) < 1e-9
    assert table.set_index("metric").loc["map50_only_a", "n_classes"] == 1


def test_paired_difference_uses_shared_resamples() -> None:
    import numpy as np
    import pandas as pd

    from echolabel.detection.evaluate import bootstrap_ap_by_slice, paired_difference

    images = np.arange(1, 41)
    gts = pd.DataFrame({"image_id": images, "category_id": 1, "iscrowd": 0})
    good = pd.DataFrame({"image_id": images, "category_id": 1, "score": np.linspace(1, 0.5, 40), "tp": 1.0})
    worse = good.copy()
    worse.loc[worse["image_id"] > 30, "tp"] = 0.0
    common = dict(n_boot=30, seed=1, min_gt=10, return_draws=True)
    _, draws_a = bootstrap_ap_by_slice(worse, gts, [1], {1: "a"}, {"all": images}, **common)
    _, draws_b = bootstrap_ap_by_slice(good, gts, [1], {1: "a"}, {"all": images}, **common)
    diff = paired_difference(draws_a, draws_b)
    row = diff.set_index("metric").loc["ap50_a"]
    assert row["difference"] > 0 and row["ci_low"] > 0 and row["n_boot"] == 30
    same = paired_difference(draws_a, draws_a)
    assert (same["difference"] == 0).all() and (same["ci_low"] == 0).all()


def test_operating_point_and_predicted_as() -> None:
    import numpy as np
    import pandas as pd

    from echolabel.detection.evaluate import operating_point_by_slice, paired_difference, predicted_as

    images = np.arange(1, 21)
    gts = pd.DataFrame(
        {
            "image_id": images,
            "category_id": 1,
            "iscrowd": 0,
            "x0": 0.0,
            "y0": 0.0,
            "x1": 10.0,
            "y1": 10.0,
        }
    )
    matched = pd.DataFrame(
        {"image_id": images, "category_id": 1, "score": np.linspace(1.0, 0.05, 20), "tp": 1.0}
    )
    table, draws = operating_point_by_slice(
        matched, gts, [1], {1: "a"}, {"all": images}, {1: 0.525}, n_boot=10
    )
    rec = table.set_index("metric").loc["recall_a"]
    assert abs(rec["value"] - 0.5) < 1e-9 and rec["n_gt"] == 20 and rec["threshold"] == 0.525
    same = paired_difference(draws, draws)
    assert (same["difference"] == 0).all()
    preds = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "category_id": [1, 2, 1],
            "score": [0.9, 0.9, 0.2],
            "x0": [0.0, 0.0, 0.0],
            "y0": [0.0, 0.0, 0.0],
            "x1": [10.0, 10.0, 10.0],
            "y1": [10.0, 10.0, 10.0],
        }
    )
    counts = predicted_as(preds, gts, 1, np.array([1, 2, 3, 4]), {1: "a", 2: "b"}, {1: 0.5, 2: 0.5})
    assert counts == {"a": 1, "b": 1, "background": 2}
    low = predicted_as(preds, gts, 1, np.array([1, 2, 3, 4]), {1: "a", 2: "b"}, {1: 0.1, 2: 0.1})
    assert low == {"a": 2, "b": 1, "background": 1}
