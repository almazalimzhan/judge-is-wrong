from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from src.analysis.pipeline import analyze
from src.utils.serialization import read_completed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-paired bootstrap analysis from saved rows")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    analyze(
        read_completed_results(args.results),
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        make_figures=not args.no_figures,
    )
    print(
        f"Analysis saved to {args.output_dir}; summary.csv contains 95% task-bootstrap intervals."
    )


if __name__ == "__main__":
    main()
