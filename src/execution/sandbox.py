"""Fail-closed local generated-code execution with operating-system isolation."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.data.base import TestCase


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    total: int
    passed: int
    status_counts: dict[str, int]
    duration_seconds: float

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "status_counts": dict(self.status_counts),
            "duration_seconds": self.duration_seconds,
            "score": self.score,
            "all_passed": self.all_passed,
        }


class Executor:
    def __init__(
        self,
        timeout_seconds: float = 2,
        memory_mb: int = 256,
        backend: str = "auto",
        docker_image: str = "python:3.11-slim",
    ) -> None:
        if timeout_seconds <= 0 or memory_mb < 64:
            raise ValueError("timeout must be positive and memory_mb >= 64")
        if backend == "auto":
            backend = "macos" if sys.platform == "darwin" else "docker"
        if backend not in {"macos", "docker"}:
            raise ValueError("Only isolated macos and docker backends are supported")
        if backend == "macos" and (sys.platform != "darwin" or not shutil.which("sandbox-exec")):
            raise SandboxUnavailable("macos backend requires macOS sandbox-exec")
        if backend == "docker" and not shutil.which("docker"):
            raise SandboxUnavailable(
                "Install/start Docker or run on macOS; unsafe execution is disabled"
            )
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.backend = backend
        self.docker_image = docker_image
        self._checked = False

    def _profile(self, directory: Path) -> str:
        def quoted(path: str | Path) -> str:
            return json.dumps(str(path))

        # The interpreter/stdlib and OS libraries are readable; user directories
        # and repository data are not. Homebrew dylibs are required on macOS.
        roots = {
            Path(sys.base_prefix).resolve(),
            Path(sys.executable).resolve().parent,
            Path("/System"),
            Path("/usr/lib"),
            Path("/usr/share"),
            Path("/Library/Apple/System/Library"),
            Path("/opt/homebrew"),
        }
        readable = " ".join(f"(subpath {quoted(path)})" for path in sorted(roots))
        # realpath()/dyld also need traversal of ancestor directories. Exact
        # ancestor literals permit directory traversal, not arbitrary descendants.
        ancestors = {parent for root in {*roots, directory} for parent in root.parents}
        readable += " " + " ".join(f"(literal {quoted(path)})" for path in sorted(ancestors))
        return f"""(version 1)
(deny default)
(allow process-exec (subpath {quoted(Path(sys.base_prefix).resolve())}) (literal {quoted(Path(sys.executable).resolve())}))
(allow sysctl-read)
(allow file-read* {readable} (subpath {quoted(directory)}))
(allow file-write* (subpath {quoted(directory)}) (literal "/dev/null"))
(allow file-read* (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))
(deny network*)
"""

    def _command(
        self, directory: Path, *, probe: bool = False, container_name: str | None = None
    ) -> list[str]:
        if self.backend == "macos":
            python = str(Path(sys.executable).resolve())
            if probe:
                return [
                    "sandbox-exec",
                    "-p",
                    self._profile(directory),
                    python,
                    "-P",
                    "-s",
                    "-S",
                    "-c",
                    "print('isolated')",
                ]
            return [
                "sandbox-exec",
                "-p",
                self._profile(directory),
                python,
                "-P",
                "-s",
                "-S",
                str(directory / "worker.py"),
            ]
        # Never auto-pull images (a run should not silently use the network).
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--env=PYTHONHASHSEED=0",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=16",
            f"--memory={self.memory_mb}m",
            "--cpus=1",
            "--user=65534:65534",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=16m",
            "--name",
            container_name or f"judge-is-wrong-{secrets.token_hex(12)}",
            "--mount",
            f"type=bind,source={directory},target=/work",
            "--workdir=/work",
            "-i",
            self.docker_image,
            "python",
            "-P",
            "-s",
            "-S",
        ]
        return command + (["-c", "print('isolated')"] if probe else ["/work/worker.py"])

    def check_available(self) -> None:
        if self._checked:
            return
        with tempfile.TemporaryDirectory(prefix="judge-is-wrong-probe-") as name:
            directory = Path(name).resolve()
            directory.chmod(0o777)  # Docker unprivileged user must traverse this directory.
            container_name = f"judge-is-wrong-probe-{secrets.token_hex(12)}"
            try:
                completed = subprocess.run(
                    self._command(directory, probe=True, container_name=container_name),
                    cwd=directory,
                    env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise SandboxUnavailable("OS sandbox could not start") from error
            finally:
                if self.backend == "docker":
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=15,
                        check=False,
                    )
            if completed.returncode != 0 or completed.stdout.strip() != b"isolated":
                raise SandboxUnavailable(
                    f"{self.backend} OS sandbox unavailable. On macOS, nested process "
                    "sandboxes can block sandbox-exec; run the CLI in a normal terminal. "
                    "For Docker, pre-pull the configured image and start its daemon."
                )
        self._checked = True

    def run(self, code: str, tests: Sequence[TestCase]) -> ExecutionResult:
        if not tests:
            return ExecutionResult(0, 0, {}, 0.0)
        self.check_available()
        started = time.monotonic()
        statuses: Counter[str] = Counter()
        worker = Path(__file__).with_name("_worker.py").read_text(encoding="utf-8")
        for case in tests:
            with tempfile.TemporaryDirectory(prefix="judge-is-wrong-") as name:
                directory = Path(name).resolve()
                directory.chmod(0o777)
                (directory / "worker.py").write_text(worker, encoding="utf-8")
                verdict = directory / "verdict.json"
                nonce = secrets.token_hex(32)
                container_name = f"judge-is-wrong-{secrets.token_hex(12)}"
                payload = json.dumps(
                    {
                        "candidate": code,
                        "test": case.code,
                        "setup": case.setup,
                        "timeout_seconds": self.timeout_seconds,
                        "memory_mb": self.memory_mb,
                        "nonce": nonce,
                        "verdict_path": str(verdict)
                        if self.backend == "macos"
                        else "/work/verdict.json",
                    }
                ).encode()
                with subprocess.Popen(
                    self._command(directory, container_name=container_name),
                    cwd=directory,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                    start_new_session=True,
                ) as process:
                    try:
                        startup_allowance = 5 if self.backend == "docker" else 0.5
                        process.communicate(
                            payload, timeout=self.timeout_seconds + startup_allowance
                        )
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        statuses["timeout"] += 1
                        continue
                    finally:
                        if self.backend == "docker":
                            # Killing only the Docker CLI can leave its container
                            # alive. Force removal also kills any candidate children.
                            subprocess.run(
                                ["docker", "rm", "-f", container_name],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=15,
                                check=False,
                            )
                    if process.returncode in {-signal.SIGXCPU, -signal.SIGKILL}:
                        statuses["timeout"] += 1
                        continue
                    try:
                        result = json.loads(verdict.read_text(encoding="utf-8"))
                        status = result["status"]
                        if (
                            result.get("nonce") != nonce
                            or status
                            not in {
                                "passed",
                                "assertion_failure",
                                "memory_limit",
                                "syntax_error",
                                "runtime_error",
                                "timeout",
                            }
                            or process.returncode != 0
                        ):
                            status = "runtime_error"
                    except (OSError, ValueError, KeyError, TypeError):
                        status = "runtime_error"
                    statuses[status] += 1
        return ExecutionResult(
            len(tests), statuses.get("passed", 0), dict(statuses), time.monotonic() - started
        )
