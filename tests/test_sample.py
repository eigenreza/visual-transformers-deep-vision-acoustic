from __future__ import annotations

import pandas as pd

from echolabel.sample import select_annotation_keys


def test_selection_covers_years_and_labels() -> None:
    rows = []
    for year in ("2018", "2024"):
        for i in range(30):
            rows.append({"key": f"{year}/S/Left/{i}", "status": "paired", "year": year})
    rows.append({"key": "2020/S/Left/x", "status": "xml_only", "year": "2020"})
    pairs = pd.DataFrame(rows)
    key_labels = {r["key"]: {"krill"} for r in rows}
    key_labels["2018/S/Left/0"] = {"redfish"}
    key_labels["2024/S/Left/1"] = {"redfish", "krill"}
    chosen = select_annotation_keys(pairs, key_labels, n_total=10, per_year_min=2, per_label_min=2, seed=1)
    assert len(chosen) == 10
    assert all(k in key_labels for k in chosen)
    years = {k.split("/")[0] for k in chosen}
    assert years == {"2018", "2024"}
    assert sum("redfish" in key_labels[k] for k in chosen) >= 2
    assert chosen == sorted(chosen)
    again = select_annotation_keys(pairs, key_labels, n_total=10, per_year_min=2, per_label_min=2, seed=1)
    assert again == chosen
