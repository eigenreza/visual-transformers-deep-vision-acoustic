"""Result figures of the acoustic models from the evaluation tables.

The command reads the tables written by the evaluation stage and draws, for
every group of configurations named in the configuration, the metric per
class with its interval and the paired difference against the reference with
its interval; for the main group it also draws the fold values and the
coverage of the predictive intervals. Every figure is written in each of the
configured formats, a vector format first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import write_manifest
from .viz.results import difference_plot, fold_spread_plot, interval_plot

log = logging.getLogger(__name__)

LOWER_IS_BETTER = {"mae": True, "crps": True, "log_score": True, "coverage90": False, "presence_ap": False}


def run_plot_results(cfg: dict[str, Any], out_dir: Path, force: bool = False) -> Path:
    """Draw the result figures into ``out_dir`` from the evaluation tables next to it."""
    fcfg = cfg.get("figures", {})
    tables = Path(fcfg["tables_dir"]) if fcfg.get("tables_dir") else out_dir.parent / "acoustic_summary"
    aggregated = pd.read_csv(tables / "metrics_aggregated.csv")
    by_run = pd.read_csv(tables / "metrics_by_run.csv")
    diff_path = tables / "differences.csv"
    differences = pd.read_csv(diff_path) if diff_path.is_file() else pd.DataFrame()
    protocol = str(fcfg.get("protocol", "loto_unflagged"))
    groups: dict[str, list[str]] = dict(fcfg.get("groups", {}))
    classes = list(fcfg.get("classes", sorted(aggregated["class_name"].unique())))
    diff_classes = list(fcfg.get("difference_classes", ["macro_label_classes", "macro_without_krill"]))
    metrics = list(fcfg.get("metrics", ["mae", "crps"]))
    formats = list(fcfg.get("formats", ["pdf", "png"]))
    main = str(fcfg.get("main_group", next(iter(groups), "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregated[aggregated["protocol"] == protocol]
    runs = by_run[by_run["protocol"] == protocol]
    diffs = differences[differences["protocol"] == protocol] if not differences.empty else differences
    outputs: list[Path] = []
    for group, experiments in groups.items():
        present = [e for e in experiments if e in set(agg["experiment"])]
        if not present:
            log.warning("group %s has no evaluated configuration", group)
            continue
        table = agg[agg["experiment"].isin(present)]
        for metric in metrics:
            for fmt in formats:
                outputs.append(
                    interval_plot(
                        table,
                        metric,
                        [c for c in classes if c in set(table["class_name"])],
                        present,
                        out_dir / f"{group}_{metric}.{fmt}",
                        lower=LOWER_IS_BETTER.get(metric, True),
                        title=f"{group}: {metric} per class with block-bootstrap intervals",
                    )
                )
            if not diffs.empty:
                sub = diffs[diffs["experiment"].isin(present) & diffs["class_name"].isin(diff_classes)]
                if not sub.empty and (sub["metric"] == metric).any():
                    for fmt in formats:
                        outputs.append(
                            difference_plot(
                                sub,
                                metric,
                                [c for c in diff_classes if c in set(sub["class_name"])],
                                out_dir / f"{group}_{metric}_difference.{fmt}",
                                title=f"{group}: {metric} minus the reference, paired, with intervals",
                            )
                        )
        if group == main:
            for fmt in formats:
                outputs.append(
                    interval_plot(
                        table,
                        "coverage90",
                        [c for c in classes if c in set(table["class_name"])],
                        present,
                        out_dir / f"{group}_coverage90.{fmt}",
                        lower=False,
                        title=f"{group}: coverage of the 90 percent conformal intervals",
                    )
                )
                for class_name in fcfg.get("fold_classes", classes[:2]):
                    if class_name in set(runs["class_name"]):
                        outputs.append(
                            fold_spread_plot(
                                runs,
                                "mae",
                                class_name,
                                present,
                                out_dir / f"{group}_folds_{class_name}.{fmt}",
                                title=f"{group}: MAE per fold and seed, {class_name}",
                            )
                        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"tables": str(tables), "protocol": protocol, "figures": [p.name for p in outputs]}, fh, indent=2
        )
    write_manifest(out_dir, "plot-results", cfg, {"tables": tables}, [*outputs, out_dir / "summary.json"])
    log.info("wrote %d figures to %s", len(outputs), out_dir)
    return out_dir
