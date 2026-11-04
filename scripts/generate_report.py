from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from src.analysis.failures import generate_failure_cases
from src.analysis.pipeline import analyze
from src.analysis.report import render_report
from src.utils.serialization import read_completed_results, result_experiment_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research readout from saved results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    rows = read_completed_results(args.results)
    frozen = result_experiment_type(rows) == "frozen_candidates"
    output = args.output_dir or ROOT / "reports" / ("frozen" if frozen else "")
    if frozen:
        from src.analysis.frozen import analyze_frozen, render_frozen_report

        analyzer = analyze_frozen
        renderer = render_frozen_report
    else:
        analyzer = analyze
        renderer = render_report
    analyzer(
        rows,
        output,
        bootstrap_samples=args.bootstrap_samples,
        make_figures=not args.no_figures,
    )
    generate_failure_cases(rows, output / "failure_cases")
    report = renderer(rows, output, output / ("frozen_report.md" if frozen else "run_report.md"))
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()
