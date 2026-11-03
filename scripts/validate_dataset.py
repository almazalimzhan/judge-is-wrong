"""Check benchmark reference implementations without exporting their source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT

from src.data.base import load_tasks
from src.execution.sandbox import Executor
from src.utils.config import load_config
from src.utils.serialization import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check selected benchmark tasks with upstream references"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/first_real.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    name = dataset["name"]
    if name not in {"mbpp", "humaneval"}:
        parser.error("Reference validation supports MBPP and HumanEval local benchmark files")
    path = ROOT / dataset["path"]
    if dataset.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != dataset["sha256"]:
        raise ValueError("Dataset SHA256 differs from the frozen experiment config")
    tasks = load_tasks(
        name,
        path,
        args.limit or dataset["limit"],
        cfg.get("seed", 42),
        exclude_task_ids=dataset.get("exclude_task_ids", ()),
    )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    records = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(x) for x in text.splitlines() if x.strip()]
    )
    lookup = {f"mbpp/{r['task_id']}" if name == "mbpp" else r["task_id"]: r for r in records}
    executor = Executor(**cfg.get("execution", {}))
    executor.check_available()
    outcomes = []
    for task in tasks:
        record = lookup[task.task_id]
        code = record["code"] if name == "mbpp" else record["prompt"] + record["canonical_solution"]
        public = executor.run(code, task.public_tests)
        hidden = executor.run(code, task.hidden_tests)
        outcomes.append(
            {
                "task_id": task.task_id,
                "public": public.to_dict(),
                "hidden": hidden.to_dict(),
                "reference_valid": (public.total == 0 or public.all_passed) and hidden.all_passed,
            }
        )
    report = {
        "kind": "reference_sanity_check_not_model_experiment",
        "seed": cfg.get("seed", 42),
        "n_tasks": len(tasks),
        "n_reference_failures": sum(not r["reference_valid"] for r in outcomes),
        "excluded_task_ids": dataset.get("exclude_task_ids", []),
        "outcomes": outcomes,
        "policy": "No automatic exclusions: inspect failures and freeze any exclusion list before real runs.",
    }
    destination = ROOT / "data" / f"{name}.validation.json"
    write_json(destination, report)
    print(
        f"Validated {len(tasks)} tasks; reference failures={report['n_reference_failures']}. Details: {destination}"
    )
    if report["n_reference_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
