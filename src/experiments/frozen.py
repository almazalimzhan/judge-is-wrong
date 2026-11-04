"""Evaluate a committed initial-candidate corpus without any proposer calls."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.analysis.pipeline import validate_trajectories
from src.data.base import Task, load_tasks
from src.evaluators.base import Evaluator
from src.execution.sandbox import Executor
from src.experiments.runner import _sha, derive_replicate_seed, resolve_config, utc_now
from src.models.base import build_model
from src.utils.config import load_dotenv, redact_config
from src.utils.serialization import write_json, write_results

FROZEN_CONDITIONS = {"baseline", "adversarial", "ensemble", "execution"}


def public_task_sha256(task: Task) -> str:
    return hashlib.sha256(
        json.dumps(task.public_view(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_id(task_id: str, replicate: int, code_hash: str) -> str:
    return hashlib.sha256(f"frozen-v1|{task_id}|{replicate}|{code_hash}".encode()).hexdigest()


def freeze_candidates(
    rows: list[dict],
    output_path: Path,
    round_index: int = 0,
    *,
    tasks: list[Task] | None = None,
    source_manifest: dict | None = None,
) -> Path:
    """Freeze every shared x0; selecting revisions or filtering on outcomes is unsupported."""
    if round_index != 0:
        raise ValueError(
            "Freeze round 0 only; later trajectory candidates require a separately prespecified corpus"
        )
    data = validate_trajectories(rows)
    output_path = Path(output_path)
    if output_path.suffix != ".jsonl":
        raise ValueError("Frozen corpora use .jsonl")
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen corpus")
    task_lookup = {task.task_id: task for task in tasks or []}
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in data:
        if row["round"] == 0:
            groups[str(row["task_id"]), int(row.get("replicate", 0))].append(row)
    candidates = []
    for (task_id, replicate), group in sorted(groups.items()):
        codes = {row["candidate_code"] for row in group}
        problems = {row["problem"] for row in group}
        if len(codes) != 1 or len(problems) != 1:
            raise ValueError(
                "Initial candidates and problems must be shared across source conditions"
            )
        row = group[0]
        code = next(iter(codes))
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        spec_hashes = {row.get("public_task_sha256") for row in group}
        if len(spec_hashes) != 1:
            raise ValueError("Source public task hashes disagree")
        spec_hash = next(iter(spec_hashes))
        task = task_lookup.get(task_id)
        protocols = {item.get("ground_truth_protocol") for item in group}
        if len(protocols) != 1:
            raise ValueError("Source test protocols disagree")
        protocol = next(iter(protocols))
        if task is not None:
            expected = public_task_sha256(task)
            if task.problem != row["problem"] or (spec_hash is not None and spec_hash != expected):
                raise ValueError("Source problem does not match the bound dataset")
            spec_hash = expected
            expected_protocol = task.metadata.get("split", "unspecified")
            if protocol is not None and protocol != expected_protocol:
                raise ValueError("Source test protocol differs from the bound dataset")
            protocol = expected_protocol
        if not isinstance(spec_hash, str) or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None:
            raise ValueError("Public task binding is unavailable; supply source dataset tasks")
        candidates.append(
            {
                "schema_version": 1,
                "candidate_id": _candidate_id(task_id, replicate, code_hash),
                "task_id": task_id,
                "replicate": replicate,
                "replicate_seed": int(
                    row.get(
                        "replicate_seed", derive_replicate_seed(int(row.get("seed", 42)), replicate)
                    )
                ),
                "candidate_code": code,
                "candidate_sha256": code_hash,
                "public_task_sha256": spec_hash,
                "problem": row["problem"],
                "source_experiment_id": row["experiment_id"],
                "source_round": 0,
                "source_run_kind": row["run_kind"],
                "test_protocol": protocol or "unspecified",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in candidates
    )
    output_path.write_text(content, encoding="utf-8")
    manifest = source_manifest or {}
    write_json(
        output_path.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "corpus_sha256": _sha(output_path),
            "n_candidates": len(candidates),
            "n_tasks": len({row["task_id"] for row in candidates}),
            "replicates": sorted({row["replicate"] for row in candidates}),
            "source_experiment_id": data[0]["experiment_id"],
            "source_dataset_sha256": manifest.get("dataset_sha256"),
            "source_dataset": redact_config(manifest.get("config", {}).get("dataset", {})),
            "selection_protocol": "All source task-by-replicate shared initial candidates, round 0; no filtering by scores or hidden outcomes",
            "frozen_at": utc_now(),
        },
    )
    return output_path


def load_corpus(path: Path) -> tuple[list[dict], dict]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if metadata.get("corpus_sha256") != _sha(path):
        raise ValueError("Frozen corpus digest differs from its committed manifest")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records or metadata.get("n_candidates") != len(records):
        raise ValueError("Frozen corpus is empty or incomplete")
    seen = set()
    task_replicates: dict[str, set[int]] = defaultdict(set)
    for record in records:
        if record.get("schema_version") != 1 or record.get("source_round") != 0:
            raise ValueError("Only versioned, prespecified initial-candidate corpora are supported")
        if (
            not isinstance(record.get("public_task_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["public_task_sha256"]) is None
        ):
            raise ValueError("Frozen corpus lacks an explicit public-task binding")
        task_id, replicate = str(record["task_id"]), record["replicate"]
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("Invalid corpus replicate identifier")
        key = task_id, replicate
        if key in seen:
            raise ValueError("Frozen corpus has duplicate task-by-replicate candidates")
        seen.add(key)
        task_replicates[task_id].add(replicate)
        fingerprint = hashlib.sha256(record["candidate_code"].encode()).hexdigest()
        if record.get("candidate_sha256") != fingerprint or record.get(
            "candidate_id"
        ) != _candidate_id(task_id, replicate, fingerprint):
            raise ValueError("Frozen candidate hash or identity mismatch")
    sets = list(task_replicates.values())
    if any(values != sets[0] for values in sets):
        raise ValueError("Every frozen task must contain the same replicate set")
    if metadata.get("n_tasks") != len(task_replicates) or metadata.get("replicates") != sorted(
        sets[0]
    ):
        raise ValueError("Frozen corpus population does not match its manifest")
    return records, metadata


def run_frozen_evaluation(
    config: dict[str, Any],
    root: Path,
    corpus_path: Path,
    *,
    output_dir: Path | None = None,
    experiment_id: str | None = None,
    update_latest: bool = True,
) -> Path:
    root = root.resolve()
    load_dotenv(root / config.get("env_file", ".env"))
    cfg = resolve_config(config, root)
    if config.get("rounds", 0) != 0:
        raise ValueError("Frozen evaluation prohibits proposer revisions; rounds must be 0")
    cfg["rounds"] = 0
    cfg["experiment_type"] = "frozen_candidates"
    if set(cfg["conditions"]) != FROZEN_CONDITIONS:
        raise ValueError(
            "Frozen evaluation requires baseline, adversarial, ensemble, and execution"
        )
    candidates, corpus_metadata = load_corpus(corpus_path)
    if "replicates" in config and config["replicates"] != len(corpus_metadata["replicates"]):
        raise ValueError("Configured replicates disagree with the frozen corpus")
    cfg["replicates"] = len(corpus_metadata["replicates"])
    seed = int(cfg.get("seed", 42))
    dataset = cfg["dataset"]
    path = root / dataset["path"] if dataset.get("path") else None
    dataset_digest = _sha(path) if path else None
    if dataset.get("sha256") and dataset_digest != dataset["sha256"]:
        raise ValueError("Dataset SHA256 differs from the frozen experiment config")
    if corpus_metadata.get("source_dataset_sha256") not in {None, dataset_digest}:
        raise ValueError("Frozen candidates were generated from a different dataset file")
    tasks = load_tasks(
        dataset["name"],
        path,
        limit=int(dataset["limit"]),
        seed=seed,
        exclude_task_ids=dataset.get("exclude_task_ids", ()),
    )
    lookup = {task.task_id: task for task in tasks}
    if set(lookup) != {record["task_id"] for record in candidates}:
        raise ValueError("Frozen corpus and configured task sample must match exactly")
    for record in candidates:
        task = lookup[record["task_id"]]
        if record["problem"] != task.problem or record["public_task_sha256"] != public_task_sha256(
            task
        ):
            raise ValueError(
                "Frozen candidate public-task specification does not match current data"
            )
        if record.get("test_protocol") != task.metadata.get("split", "unspecified"):
            raise ValueError("Frozen candidates were generated under a different test protocol")
    kind = "mock" if cfg["models"].get("backend", "mock") == "mock" else "real"
    if kind == "mock" and dataset["name"] != "toy":
        raise ValueError("Mock evaluators support toy data only")
    identifier = experiment_id or f"frozen-{uuid4().hex[:12]}-{kind}"
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("experiment_id must be a simple directory name")
    run_dir = Path(output_dir) if output_dir is not None else root / "results" / identifier
    if (run_dir / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen evaluation")
    run_dir.mkdir(parents=True, exist_ok=True)
    executor = Executor(**cfg.get("execution", {}))
    executor.check_available()
    temperatures = cfg.get("temperatures", {})
    # The shared model abstraction defines all roles, but this runner never uses proposer.
    model_config = {"proposer_model": "unused-frozen-no-proposer-calls", **cfg["models"]}
    models = {
        replicate: build_model(
            model_config,
            root / cfg.get("cache_dir", ".cache/model_calls"),
            derive_replicate_seed(seed, replicate),
        )
        for replicate in corpus_metadata["replicates"]
    }
    evaluators = {
        replicate: Evaluator(
            model,
            ensemble_size=int(cfg.get("ensemble_size", 3)),
            judge_temperature=float(temperatures.get("judge", 0)),
            critic_temperature=float(temperatures.get("critic", 0)),
            ensemble_temperature=float(temperatures.get("ensemble", 0.7)),
            ensemble_models=cfg.get("ensemble_models"),
        )
        for replicate, model in models.items()
    }
    manifest = {
        "experiment_id": identifier,
        "experiment_type": "frozen_candidates",
        "run_kind": kind,
        "status": "running",
        "started_at": utc_now(),
        "config": redact_config(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "models": {str(rep): model.metadata() for rep, model in models.items()},
        "execution_backend": executor.backend,
        "dataset_sha256": dataset_digest,
        "corpus": corpus_metadata,
        "corpus_sha256": corpus_metadata["corpus_sha256"],
        "source_hashes": {
            str(p.relative_to(root)): _sha(p)
            for base in ("src", "scripts", "configs")
            for p in sorted((root / base).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".yaml"}
        },
        "task_ids": sorted(lookup),
        "replicates": corpus_metadata["replicates"],
        "expected_rows": len(candidates) * len(FROZEN_CONDITIONS),
        "proposer_calls": 0,
        "revisions": 0,
    }
    write_json(run_dir / "manifest.json", manifest)
    rows = []
    try:
        for record in candidates:
            task = lookup[record["task_id"]]
            code, replicate = record["candidate_code"], record["replicate"]
            public = executor.run(code, task.public_tests)
            hidden = executor.run(code, task.hidden_tests)
            conditions = list(cfg["conditions"])
            random.Random(f"frozen-v1/{seed}/{record['candidate_id']}").shuffle(conditions)
            for condition in conditions:
                decision = evaluators[replicate].evaluate(
                    task,
                    code,
                    condition,
                    public_result=public,
                    call_id=f"frozen/{corpus_metadata['corpus_sha256']}/{record['candidate_id']}/replicate/{replicate}/{condition}",
                )
                rows.append(
                    {
                        "experiment_id": identifier,
                        "experiment_type": "frozen_candidates",
                        "run_kind": kind,
                        "task_id": task.task_id,
                        "replicate": replicate,
                        "replicate_seed": record["replicate_seed"],
                        "seed": seed,
                        "evaluator_seed": derive_replicate_seed(seed, replicate),
                        "condition": condition,
                        "round": 0,
                        "problem": task.problem,
                        "candidate_id": record["candidate_id"],
                        "candidate_code": code,
                        "candidate_sha256": record["candidate_sha256"],
                        "public_task_sha256": record["public_task_sha256"],
                        "corpus_sha256": corpus_metadata["corpus_sha256"],
                        "source_experiment_id": record["source_experiment_id"],
                        "source_round": 0,
                        **decision.to_dict(),
                        "public_test_result": public.to_dict(),
                        "hidden_test_result": hidden.to_dict(),
                        "ground_truth_score": hidden.score,
                        "pass_at_1": hidden.all_passed,
                        "ground_truth_valid": hidden.oracle_valid,
                        "ground_truth_protocol": task.metadata.get("split", "unspecified"),
                        "evaluation_gap": decision.judge_score - hidden.score,
                        "timestamp": utc_now(),
                    }
                )
                write_results(run_dir / "partial.csv", rows)
        suffix = cfg.get("output_format", "csv")
        destination = run_dir / f"results.{suffix}"
        write_results(destination, rows)
        manifest.update(status="complete", completed_at=utc_now(), actual_rows=len(rows))
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "partial.csv").unlink(missing_ok=True)
        if update_latest:
            latest = root / "results" / f"latest_frozen.{suffix}"
            temporary = latest.with_suffix(latest.suffix + ".tmp")
            shutil.copyfile(destination, temporary)
            temporary.replace(latest)
            write_json(root / "results/latest_frozen.manifest.json", manifest)
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
