"""GBD-CTU report comparison.

This module merges GNN and baseline scenario reports into a single comparison
table. Inputs are CSV report paths; outputs are combined CSV and JSON tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compare_reports(gnn_report_path: str | Path, baseline_report_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Merge GNN and baseline report tables into a sorted comparison frame."""

    gnn_report = pd.read_csv(gnn_report_path)
    baseline_report = pd.read_csv(baseline_report_path)
    combined = pd.concat([gnn_report, baseline_report], ignore_index=True)
    combined = combined.sort_values(["scenario", "model"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    destination.with_suffix(".json").write_text(json.dumps(combined.to_dict(orient="records"), indent=2), encoding="utf-8")
    return combined
