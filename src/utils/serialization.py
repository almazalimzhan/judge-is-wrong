from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

JSON_FIELDS = {
    "public_test_result",
    "hidden_test_result",
    "ensemble_scores",
    "ensemble_details",
    "ensemble_judgments",
}
BOOL_FIELDS = {"judge_binary", "pass_at_1"}
FLOAT_FIELDS = {"judge_score", "ground_truth_score", "evaluation_gap"}
INT_FIELDS = {"round", "seed", "replicate"}
OPTIONAL_FLOAT_FIELDS = {"ensemble_mean", "ensemble_median", "ensemble_variance"}


def _native(value: Any) -> Any:
    """Arrow/pandas return arrays and null structs; retain portable JSON values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {
            key: (
                {k: _native(v) for k, v in item.items() if v is not None}
                if key == "status_counts" and isinstance(item, dict)
                else _native(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty results")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install [parquet] extra or select CSV output") from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    if path.suffix != ".csv":
        raise ValueError("Results must be .csv or .parquet")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False, allow_nan=False)
                        if isinstance(v, (dict, list, tuple))
                        else v
                    )
                    for k, v in row.items()
                }
            )
    temporary.replace(path)


def read_results(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        rows = [_native(row) for row in pd.read_parquet(path).to_dict("records")]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Results must be .csv or .parquet")
    for row in rows:
        for key, value in list(row.items()):
            if key in BOOL_FIELDS and isinstance(value, str):
                if value.lower() not in {"true", "false"}:
                    raise ValueError(f"Invalid boolean {key}: {value!r}")
                row[key] = value.lower() == "true"
            elif key in JSON_FIELDS and isinstance(value, str):
                row[key] = json.loads(value)
            elif key in FLOAT_FIELDS:
                row[key] = float(value)
            elif key in OPTIONAL_FLOAT_FIELDS:
                row[key] = None if value in (None, "") else float(value)
            elif key == "ensemble_majority_vote":
                row[key] = None if value in (None, "") else str(value).lower() == "true"
            elif key in INT_FIELDS:
                row[key] = int(value)
    return rows


def read_completed_results(path: Path) -> list[dict]:
    """Analysis entry point: incomplete managed runs cannot become findings."""
    if path.stem == "partial":
        raise ValueError("partial.csv is an interrupted run; analyze completed results only")
    rows = read_results(path)
    manifest_path = path.parent / (
        "latest.manifest.json" if path.stem == "latest" else "manifest.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Experiment manifest does not mark this run complete")
        if len(rows) != manifest.get("expected_rows") or len(rows) != manifest.get("actual_rows"):
            raise ValueError("Result row count does not match the complete experiment manifest")
        if {row.get("experiment_id") for row in rows} != {manifest.get("experiment_id")}:
            raise ValueError("Results and manifest identify different experiments")
    return rows
