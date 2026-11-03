"""Fetch attributed benchmark files only on an explicit CLI invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT

DATASETS = {
    "mbpp": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
        "filename": "mbpp.jsonl",
        "license": "CC-BY-4.0",
        "attribution": "Austin et al. (2021), Program Synthesis with Large Language Models",
        "paper": "https://arxiv.org/abs/2108.07732",
        "license_source": "https://huggingface.co/datasets/google-research-datasets/mbpp",
    },
    "humaneval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "filename": "HumanEval.jsonl.gz",
        "license": "MIT",
        "attribution": "Chen et al. (2021), Evaluating Large Language Models Trained on Code; Copyright OpenAI",
        "paper": "https://arxiv.org/abs/2107.03374",
        "license_source": "https://raw.githubusercontent.com/openai/human-eval/master/LICENSE",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark with SHA256 provenance")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--sha256", help="Require a previously frozen data digest")
    args = parser.parse_args()
    info = DATASETS[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / info["filename"]
    if destination.exists():
        data = destination.read_bytes()
        print(f"Using existing {destination}")
    else:
        request = urllib.request.Request(
            info["url"], headers={"User-Agent": "judge-is-wrong-research/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("Unexpected benchmark size")
    digest = hashlib.sha256(data).hexdigest()
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise ValueError("Dataset SHA256 mismatch; refusing to save")
    destination.write_bytes(data)
    metadata = {
        **info,
        "sha256": digest,
        "bytes": len(data),
        "downloaded_or_verified_at": datetime.now(timezone.utc).isoformat(),
        "note": "Upstream branch can move; freeze this SHA256 before the research run.",
    }
    destination.with_suffix(destination.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if args.dataset == "humaneval":
        with urllib.request.urlopen(info["license_source"], timeout=30) as response:
            license_text = response.read(100_000).decode()
        (args.output_dir / "HumanEval.LICENSE").write_text(license_text, encoding="utf-8")
    print(f"Saved {destination}\nSHA256: {digest}\nAttribution: {info['attribution']}")


if __name__ == "__main__":
    main()
