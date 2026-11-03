from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from src.analysis.failures import generate_failure_cases
from src.analysis.pipeline import analyze
from src.analysis.report import render_report
from src.utils.serialization import read_completed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research readout from saved results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    rows = read_completed_results(args.results)
    analyze(
        rows,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        make_figures=not args.no_figures,
    )
    generate_failure_cases(rows, args.output_dir / "failure_cases")
    report = render_report(rows, args.output_dir, args.output_dir / "run_report.md")
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()
