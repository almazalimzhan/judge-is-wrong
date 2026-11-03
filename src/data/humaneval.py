"""HumanEval adapter that preserves the original dependent hidden harness."""

from __future__ import annotations

import ast
import doctest
import gzip
import json
from pathlib import Path

from .base import Task, TestCase


def _public_examples(prompt: str) -> tuple[TestCase, ...]:
    try:
        tree = ast.parse(prompt + "\n    pass\n")
    except SyntaxError:
        return ()
    cases = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node) or ""
        for example in doctest.DocTestParser().get_examples(doc):
            try:
                ast.parse(example.source.strip(), mode="eval")
                expected = ast.literal_eval(example.want.strip())
            except (SyntaxError, ValueError):
                continue  # No evaluation of dataset-supplied arbitrary expressions here.
            cases.append(TestCase(f"assert ({example.source.strip()}) == {expected!r}"))
    return tuple(cases)


def load_humaneval(path: Path) -> list[Task]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    tasks = [
        Task(
            task_id=record["task_id"],
            problem=record["prompt"]
            + "\nReturn a complete Python module with this function and required imports.",
            entry_point=record["entry_point"],
            public_tests=_public_examples(record["prompt"]),
            hidden_tests=(TestCase(record["test"] + f"\ncheck({record['entry_point']})"),),
            metadata={
                "source": "OpenAI HumanEval",
                "license": "MIT",
                "split": "docstring-public-original-harness-hidden-v1",
                "ground_truth_unit": "original complete harness",
            },
        )
        for record in records
    ]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate HumanEval task ids")
    return tasks
