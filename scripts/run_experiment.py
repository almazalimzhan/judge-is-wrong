from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ROOT

from src.experiments.runner import run_experiment
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired judge-feedback trajectories")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--limit", type=int, help="Override task count for a pilot")
    parser.add_argument("--rounds", type=int, help="Override number of revisions")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = run_experiment(cfg, ROOT, output_dir=args.output_dir)
    print(f"Saved complete experiment: {output}")


if __name__ == "__main__":
    main()
