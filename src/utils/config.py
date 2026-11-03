"""Small configuration boundary; bundled YAML files are also valid JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install the [yaml] extra for non-JSON YAML syntax") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping")
    return value


def load_dotenv(path: Path) -> None:
    """Read literal KEY=value pairs without shell expansion; existing env wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip().replace("_", "").isalnum():
            raise ValueError("Invalid .env line: expected KEY=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (
                "[REDACTED]"
                if any(s in k.lower() for s in ("api_key", "password", "secret", "authorization"))
                else redact_config(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_config(v) for v in value]
    return value
