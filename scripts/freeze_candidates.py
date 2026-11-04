from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT

from src.data.base import load_tasks
from src.experiments.frozen import freeze_candidates
from src.utils.serialization import read_completed_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze every shared initial candidate without outcome-based filtering"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        dest="round_index",
        help="Only prespecified shared initial round 0 is supported",
    )
    args = parser.parse_args()
    rows = read_completed_results(args.results)
    manifest_path = args.results.parent / (
        "latest.manifest.json" if args.results.stem == "latest" else "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dataset = manifest.get("config", {}).get("dataset")
    tasks = None
    if dataset:
        path = ROOT / dataset["path"] if dataset.get("path") else None
        tasks = load_tasks(
            dataset["name"],
            path,
            limit=int(dataset["limit"]),
            seed=int(manifest["seed"]),
            exclude_task_ids=dataset.get("exclude_task_ids", ()),
        )
    path = freeze_candidates(
        rows, args.output, args.round_index, tasks=tasks, source_manifest=manifest
    )
    print(f"Saved frozen initial-candidate corpus: {path}")


if __name__ == "__main__":
    main()
