from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from src.experiments.frozen import run_frozen_evaluation
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score frozen candidates under four evaluators; never revise or call proposer"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frozen.yaml")
    parser.add_argument("--corpus", "--candidates", dest="corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = run_frozen_evaluation(
        load_config(args.config), ROOT, args.corpus, output_dir=args.output_dir
    )
    print(f"Saved separate frozen-candidate experiment: {destination}")


if __name__ == "__main__":
    main()
