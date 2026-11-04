"""Small synchronous model interface with auditable disk caching."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        role: str,
        temperature: float,
        seed: int,
        call_id: str,
    ) -> ModelResponse: ...

    def endpoint(self, role: str) -> str: ...


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class CachedModel:
    """Cache identity includes the draw namespace, preserving ensemble draws."""

    def __init__(
        self, backend: Backend, cache_dir: Path, models: dict[str, str], seed: int = 0
    ) -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = dict(models)
        self.seed = seed

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        role: str,
        temperature: float = 0.0,
        call_id: str = "",
        model: str | None = None,
    ) -> str:
        if role not in self.models:
            raise ValueError(f"Unknown model role: {role}")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        selected_model = model or self.models[role]
        prompt_hash = _digest(messages)
        # A distinct namespace is a distinct draw even if all prompts are equal.
        draw_seed = int(_digest([self.seed, call_id])[:8], 16) % (2**31 - 1)
        request = {
            "backend": self.backend.name,
            "endpoint": self.backend.endpoint(role),
            "model": selected_model,
            "role": role,
            "temperature": temperature,
            "backend_settings": self._backend_settings(),
            "prompt_hash": prompt_hash,
            "seed": draw_seed,
            "call_id": call_id,
        }
        request_hash = _digest(request)
        path = self.cache_dir / f"{request_hash}.json"
        hit = path.exists()
        if hit:
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                response = self.backend.generate(
                    messages,
                    model=selected_model,
                    role=role,
                    temperature=temperature,
                    seed=draw_seed,
                    call_id=call_id,
                )
            except Exception as exc:
                self._log_call(
                    {
                        **request,
                        "request_hash": request_hash,
                        "timestamp": _timestamp(),
                        "response": None,
                        "usage": {},
                        "cache_hit": False,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    }
                )
                raise
            record = {
                **request,
                "request_hash": request_hash,
                "timestamp": _timestamp(),
                "response": response.content,
                "usage": response.usage,
                "raw_response": response.raw_response,
            }
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            temporary.chmod(0o600)
            temporary.replace(path)
        call_record = {
            **request,
            "request_hash": request_hash,
            "timestamp": _timestamp(),
            "response": record["response"],
            "usage": record.get("usage", {}),
            "cache_hit": hit,
            "status": "complete",
        }
        self._log_call(call_record)
        return str(record["response"])

    def _log_call(self, record: dict[str, Any]) -> None:
        with (self.cache_dir / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _backend_settings(self) -> dict[str, Any]:
        method = getattr(self.backend, "cache_settings", None)
        return method() if method is not None else {}

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "models": self.models,
            "backend_settings": self._backend_settings(),
            "endpoints": {role: self.backend.endpoint(role) for role in self.models},
            "seed": self.seed,
            "cache_dir": str(self.cache_dir),
        }


def build_model(config: dict[str, Any], cache_dir: Path, seed: int = 0) -> CachedModel:
    """Secrets come only from environment variables, never from saved configs."""
    backend_name = str(config.get("backend", "mock"))
    if backend_name == "mock":
        from .mock import MockBackend

        backend: Backend = MockBackend()
    elif backend_name in {"openai_compatible", "openai-compatible"}:
        from .openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            api_base=str(config.get("api_base") or os.environ.get("LLM_API_BASE", "")),
            timeout_seconds=float(config.get("timeout_seconds", 90)),
            max_tokens=int(config.get("max_tokens", 2048)),
            retries=int(config.get("retries", 2)),
            send_seed=bool(config.get("send_seed", False)),
            local_only=bool(config.get("local_only", False)),
            json_roles=bool(config.get("json_roles", False)),
        )
    else:
        raise ValueError(f"Unknown model backend: {backend_name}")
    models: dict[str, str] = {}
    for role in ("proposer", "judge", "critic"):
        model_name = (
            config.get(f"{role}_model")
            if backend_name == "mock" or config.get("local_only", False)
            else os.environ.get(f"{role.upper()}_MODEL") or config.get(f"{role}_model")
        )
        if not model_name:
            if backend_name == "mock":
                model_name = f"mock-{role}-v1"
            else:
                raise ValueError(f"Set {role.upper()}_MODEL or {role}_model in models config")
        models[role] = str(model_name)
    return CachedModel(backend, cache_dir, models, seed)
