"""Dependency-free client for the Chat Completions-compatible HTTP protocol."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import ModelResponse


class ModelAPIError(RuntimeError):
    pass


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(
        self,
        api_base: str,
        timeout_seconds: float = 90,
        max_tokens: int = 2048,
        retries: int = 2,
        send_seed: bool = False,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.retries = retries
        self.send_seed = send_seed
        if timeout_seconds <= 0 or max_tokens <= 0 or retries < 0:
            raise ValueError("Invalid API timeout, max_tokens, or retries")

    def cache_settings(self) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "max_tokens": self.max_tokens,
            "send_seed": self.send_seed,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
        }

    def endpoint(self, role: str) -> str:
        base = os.environ.get(f"{role.upper()}_API_BASE", self.api_base).rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Set LLM_API_BASE (or role-specific API_BASE) to an HTTP(S) base URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API base must not contain credentials, query, or fragment")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        role: str,
        temperature: float,
        seed: int,
        call_id: str,
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.send_seed:
            body["seed"] = seed
        key = os.environ.get(f"{role.upper()}_API_KEY", os.environ.get("LLM_API_KEY", ""))
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(
            self.endpoint(role), data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as handle:
                    # Bound provider responses; avoid accidental unbounded disk growth.
                    raw = handle.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ModelAPIError("Model response exceeds 16 MiB")
                parsed = json.loads(raw)
                content = parsed["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Model content must be a nonempty string")
                return ModelResponse(content, parsed.get("usage") or {}, parsed)
            except HTTPError as exc:
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                # Do not echo response bodies, headers, URLs, or credentials.
                raise ModelAPIError(f"Model HTTP request failed with status {exc.code}") from None
            except (URLError, TimeoutError, OSError):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelAPIError(
                    "Model request failed due to network error or timeout"
                ) from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise ModelAPIError("Malformed Chat Completions-compatible response") from None
        raise AssertionError("Unreachable retry state")
