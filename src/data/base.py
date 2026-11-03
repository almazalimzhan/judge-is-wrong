"""Small dataset boundary: public task views never contain hidden tests."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TestCase:
    code: str
    setup: str = ""

    __test__ = False


@dataclass(frozen=True)
class Task:
    task_id: str
    problem: str
    entry_point: str
    public_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_view(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "problem": self.problem,
            "entry_point": self.entry_point,
            "public_tests": [
                {"code": test.code, "setup": test.setup} for test in self.public_tests
            ],
        }


def load_tasks(
    dataset: str,
    path: Path | str | None = None,
    limit: int = 10,
    seed: int = 42,
    exclude_task_ids: tuple[str, ...] | list[str] = (),
) -> list[Task]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if dataset == "toy":
        from .toy import toy_tasks

        tasks = toy_tasks()
    elif dataset == "mbpp":
        from .mbpp import load_mbpp

        if path is None:
            raise ValueError("MBPP requires a local JSONL or JSON path")
        tasks = load_mbpp(Path(path), seed=seed)
    elif dataset == "humaneval":
        from .humaneval import load_humaneval

        if path is None:
            raise ValueError("HumanEval requires a local JSONL[.gz] path")
        tasks = load_humaneval(Path(path))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    # Local RNG, stable input order: no process-global randomness is modified.
    tasks = sorted(tasks, key=lambda task: task.task_id)
    random.Random(seed).shuffle(tasks)
    excluded = set(exclude_task_ids)
    unknown = excluded - {task.task_id for task in tasks}
    if unknown:
        raise ValueError(f"Unknown excluded task IDs: {sorted(unknown)}")
    # Filter after shuffling so exclusions preserve other tasks' relative order.
    tasks = [task for task in tasks if task.task_id not in excluded]
    if limit > len(tasks):
        raise ValueError(
            f"Requested {limit} tasks, but {dataset} contains {len(tasks)} eligible tasks"
        )
    return tasks[:limit]
