from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from _bootstrap import ROOT

from src.experiments.runner import run_experiment
from src.metrics.core import compute_metrics
from src.utils.config import load_config
from src.utils.serialization import read_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline smoke test; scripted mock data only")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/experiment.yaml")
    cfg["dataset"] = {"name": "toy", "limit": 3}
    cfg["models"] = {"backend": "mock"}
    cfg["ensemble_size"] = 3
    cfg["rounds"] = 2
    output_dir = args.output_dir or ROOT / "results" / "smoke" / uuid4().hex[:8]
    path = run_experiment(cfg, ROOT, output_dir=output_dir, update_latest=False)
    rows = read_results(path)
    assert len(rows) == 3 * len(cfg["conditions"]) * 3
    assert all(row["run_kind"] == "mock" for row in rows)
    assert {row["round"] for row in rows} == {0, 1, 2}
    assert any(row["ground_truth_score"] < 1 for row in rows)
    print("SMOKE PASSED — synthetic software validation; no research findings.")
    print(f"Saved {len(rows)} rows: {path}")
    example = [
        r for r in rows if r["task_id"] == rows[0]["task_id"] and r["condition"] == "baseline"
    ]
    print("Example mock trajectory:")
    for row in example:
        print(
            f"  round={row['round']} J={row['judge_score']:.3f} "
            f"G={row['ground_truth_score']:.3f} E={row['evaluation_gap']:.3f}"
        )
    print(json.dumps(compute_metrics([r for r in rows if r["round"] == 2]), indent=2))


if __name__ == "__main__":
    main()
