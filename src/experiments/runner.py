"""Paired iterative experiments; hidden execution never enters model feedback."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.data.base import load_tasks
from src.evaluators.base import Evaluator
from src.execution.sandbox import Executor
from src.models.base import build_model
from src.proposers.base import Proposer
from src.utils.config import load_config, load_dotenv, redact_config
from src.utils.serialization import write_json, write_results

LOGGER = logging.getLogger(__name__)
CONDITIONS = {
    "baseline",
    "critique",
    "ensemble",
    "adversarial",
    "execution",
    "baseline_no_feedback",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(config)
    model_file = root / str(config.get("models_file", "configs/models.yaml"))
    models = load_config(model_file) if model_file.exists() else {}
    models.update(config.get("models", {}))
    resolved["models"] = models
    conditions = resolved.get("conditions", ["baseline"])
    if (
        not isinstance(conditions, list)
        or not conditions
        or len(set(conditions)) != len(conditions)
        or set(conditions) - CONDITIONS
    ):
        raise ValueError("conditions must be a nonempty list of distinct supported names")
    if int(resolved.get("rounds", 3)) < 0:
        raise ValueError("rounds is the number of revisions and must be nonnegative")
    if resolved.get("output_format", "csv") not in {"csv", "parquet"}:
        raise ValueError("output_format must be csv or parquet")
    resolved["conditions"] = conditions
    return resolved


def run_experiment(
    config: dict[str, Any],
    root: Path,
    *,
    output_dir: Path | None = None,
    experiment_id: str | None = None,
    update_latest: bool = True,
) -> Path:
    root = root.resolve()
    load_dotenv(root / ".env")
    cfg = resolve_config(config, root)
    seed = int(cfg.get("seed", 42))
    rounds = int(cfg.get("rounds", 3))
    dataset = cfg.get("dataset", {"name": "toy", "limit": 10})
    name = str(dataset["name"])
    path = root / dataset["path"] if dataset.get("path") else None
    if path and dataset.get("sha256") and _sha(path) != dataset["sha256"]:
        raise ValueError("Dataset SHA256 differs from the frozen experiment config")
    tasks = load_tasks(
        name,
        path,
        limit=int(dataset.get("limit", 10)),
        seed=seed,
        exclude_task_ids=dataset.get("exclude_task_ids", ()),
    )
    kind = "mock" if cfg["models"].get("backend", "mock") == "mock" else "real"
    if kind == "mock" and name != "toy":
        raise ValueError("Mock backend is a toy fixture; use toy data or configure a real backend")
    identifier = (
        experiment_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{kind}-{uuid4().hex[:8]}"
    )
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("experiment_id must be a simple directory name")
    run_dir = Path(output_dir) if output_dir is not None else root / "results" / identifier
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(
            "Refusing to overwrite an existing experiment; choose another output directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    executor = Executor(**cfg.get("execution", {}))
    # Check before any paid model request. There is no unsafe execution fallback.
    executor.check_available()
    model = build_model(cfg["models"], root / cfg.get("cache_dir", ".cache/model_calls"), seed)
    temperatures = cfg.get("temperatures", {})
    evaluator = Evaluator(
        model,
        ensemble_size=int(cfg.get("ensemble_size", 3)),
        judge_temperature=float(temperatures.get("judge", 0.0)),
        critic_temperature=float(temperatures.get("critic", 0.0)),
        ensemble_temperature=float(temperatures.get("ensemble", 0.7)),
        ensemble_models=cfg.get("ensemble_models"),
    )
    proposer = Proposer(model, temperature=float(temperatures.get("proposer", 0.2)))
    sources = {
        str(p.relative_to(root)): _sha(p)
        for base in ("src", "scripts", "configs")
        for p in sorted((root / base).rglob("*"))
        if p.is_file() and p.suffix in {".py", ".yaml"}
    }
    manifest = {
        "experiment_id": identifier,
        "status": "running",
        "run_kind": kind,
        "started_at": utc_now(),
        "config": redact_config(cfg),
        "config_sha256": hashlib.sha256(
            json.dumps(redact_config(cfg), sort_keys=True).encode()
        ).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "installed_packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        "model": model.metadata(),
        "source_hashes": sources,
        "dataset_sha256": _sha(path) if path else None,
        "task_ids": [task.task_id for task in tasks],
        "execution_backend": executor.backend,
        "round_convention": "0 is initial; T is final after T revisions",
        "expected_rows": len(tasks) * len(cfg["conditions"]) * (rounds + 1),
    }
    write_json(run_dir / "manifest.json", manifest)
    rows: list[dict] = []
    execution_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    try:
        for index, task in enumerate(tasks):
            initial = proposer.generate(task, [], call_id=f"task/{task.task_id}/initial")
            order = list(cfg["conditions"])
            random.Random(f"{seed}/{task.task_id}").shuffle(order)
            for condition in order:
                candidate = initial
                history: list[dict] = []
                trajectory: list[dict] = []
                for round_index in range(rounds + 1):
                    if round_index:
                        candidate = proposer.generate(
                            task,
                            history,
                            call_id=f"task/{task.task_id}/{condition}/revision/{round_index}",
                        )
                    fingerprint = hashlib.sha256(candidate.encode()).hexdigest()
                    execution_key = (task.task_id, fingerprint)
                    if execution_key not in execution_cache:
                        execution_cache[execution_key] = (
                            executor.run(candidate, task.public_tests),
                            executor.run(candidate, task.hidden_tests),
                        )
                    public, hidden = execution_cache[execution_key]
                    decision = evaluator.evaluate(
                        task,
                        candidate,
                        "baseline" if condition == "baseline_no_feedback" else condition,
                        public_result=public,
                        call_id=f"task/{task.task_id}/{condition}/round/{round_index}",
                    )
                    outcome = decision.to_dict()
                    row = {
                        "experiment_id": identifier,
                        "run_kind": kind,
                        "task_id": task.task_id,
                        "condition": condition,
                        "round": round_index,
                        "seed": seed,
                        "replicate": 0,
                        "problem": task.problem,
                        "candidate_code": candidate,
                        **outcome,
                        "public_test_result": public.to_dict(),
                        "hidden_test_result": hidden.to_dict(),
                        "ground_truth_score": hidden.score,
                        "pass_at_1": hidden.all_passed,
                        "evaluation_gap": decision.judge_score - hidden.score,
                        "candidate_sha256": fingerprint,
                        "timestamp": utc_now(),
                    }
                    rows.append(row)
                    trajectory.append(row)
                    if condition == "baseline_no_feedback":
                        history.append({"candidate_code": candidate, "round": round_index})
                    else:
                        history.append(
                            {
                                "candidate_code": candidate,
                                "round": round_index,
                                "judge_score": decision.judge_score,
                                "judge_binary": decision.judge_binary,
                                "judge_reasoning_summary": decision.judge_reasoning_summary,
                                "critic_feedback": decision.critic_feedback,
                            }
                        )
                    # Incremental outputs preserve completed work after an interrupted API call.
                    write_results(run_dir / "partial.csv", rows)
                    safe_task = hashlib.sha256(task.task_id.encode()).hexdigest()[:16]
                    write_json(
                        run_dir / "trajectories" / f"{safe_task}-{condition}.json", trajectory
                    )
                LOGGER.info(
                    json.dumps(
                        {
                            "event": "trajectory_complete",
                            "task": task.task_id,
                            "condition": condition,
                            "task_index": index + 1,
                            "task_count": len(tasks),
                            "run_kind": kind,
                        }
                    )
                )
        suffix = cfg.get("output_format", "csv")
        destination = run_dir / f"results.{suffix}"
        write_results(destination, rows)
        manifest.update(status="complete", completed_at=utc_now(), actual_rows=len(rows))
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "partial.csv").unlink(missing_ok=True)
        if update_latest:
            latest = root / "results" / f"latest.{suffix}"
            temporary = latest.with_suffix(latest.suffix + ".tmp")
            shutil.copyfile(destination, temporary)
            temporary.replace(latest)
            write_json(root / "results" / "latest.manifest.json", manifest)
        return destination
    except BaseException as exc:
        manifest.update(
            status="failed",
            failed_at=utc_now(),
            actual_rows=len(rows),
            error_type=type(exc).__name__,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise
