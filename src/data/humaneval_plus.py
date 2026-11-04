"""Pinned original-format HumanEval+ harnesses, without synthesizing new tests."""

from dataclasses import replace
from pathlib import Path

from .base import Task
from .humaneval import load_humaneval


def load_humaneval_plus(path: Path) -> list[Task]:
    tasks = load_humaneval(path)
    return [
        replace(
            task,
            metadata={
                "source": "EvalPlus HumanEval+ v0.1.10 original format",
                "license": "Apache-2.0; underlying HumanEval MIT",
                "split": "docstring-public-plus-full-harness-hidden-v1",
                "ground_truth_unit": "complete expanded harness; G is binary",
                "suite_version": "v0.1.10",
            },
        )
        for task in tasks
    ]
