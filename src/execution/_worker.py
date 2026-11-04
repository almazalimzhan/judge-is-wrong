"""Trusted test controller; candidate code executes in another OS-sandboxed process."""

import ast
import builtins
import json
import math
import os
import random
import resource
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _codec import MAX_BYTES, InvalidMessage, UnsupportedValue, decode, dumps, encode, loads
from _policy import PolicyViolation, check_source


class CandidateFailure(Exception):
    pass


class ProtocolViolation(Exception):
    pass


class UnsupportedProtocol(Exception):
    pass


class CandidateTimeout(Exception):
    pass


def main():
    payload = json.load(sys.stdin)
    random.seed(0)
    seconds = math.ceil(payload["timeout_seconds"]) + 5
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    namespace = {"__name__": "__trusted_test__"}
    process = None
    status = "controller_error"
    suspicion = [None]
    try:
        setup = compile(payload["setup"], "<test-setup>", "exec")
        test = compile(payload["test"], "<test>", "exec")
        tree = ast.parse(payload["setup"] + "\n" + payload["test"])
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "id"
            ):
                raise UnsupportedProtocol
            if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
            ):
                operands = [node.left, *node.comparators]
                if not any(
                    isinstance(item, ast.Constant)
                    and (item.value is None or type(item.value) is bool)
                    for item in operands
                ):
                    raise UnsupportedProtocol
        exec(setup, namespace)
        try:
            candidate_tree = check_source(payload["candidate"])
        except SyntaxError:
            status = "syntax_error"
            return
        except PolicyViolation:
            status = "integrity_violation"
            return
        exports = {node.name for node in candidate_tree.body if isinstance(node, ast.FunctionDef)}
        exports -= set(vars(builtins)) | set(namespace) | {"check", "candidate"}
        process = subprocess.Popen(
            payload["candidate_command"],
            cwd=payload["candidate_directory"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
        )

        def violation():
            suspicion[0] = "integrity_violation"
            raise ProtocolViolation

        def receive():
            raw = process.stdout.readline(MAX_BYTES + 1)
            if not raw:
                try:
                    returncode = process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    violation()
                if returncode in {-signal.SIGXCPU, -signal.SIGKILL}:
                    raise CandidateTimeout
                raise CandidateFailure
            if not raw.endswith(b"\n"):
                violation()
            try:
                response = loads(raw)
            except InvalidMessage:
                violation()
            if type(response) is not dict:
                violation()
            return response

        process.stdin.write(
            dumps(
                {
                    "candidate": payload["candidate"],
                    "timeout_seconds": payload["timeout_seconds"],
                    "memory_mb": payload["memory_mb"],
                }
            )
        )
        process.stdin.flush()
        hello = receive()
        if hello != {"status": "ready"}:
            if hello == {"status": "integrity_violation"}:
                violation()
            if hello.get("status") == "candidate_error" and set(hello) == {"status", "kind"}:
                status = "syntax_error" if hello["kind"] == "SyntaxError" else "runtime_error"
                return
            violation()
        errors = {
            name: type("Remote" + name, (getattr(builtins, name), CandidateFailure), {})
            for name in (
                "ValueError",
                "TypeError",
                "IndexError",
                "KeyError",
                "ZeroDivisionError",
                "OverflowError",
                "AssertionError",
                "MemoryError",
                "TimeoutError",
                "StopIteration",
                "NotImplementedError",
                "RecursionError",
                "RuntimeError",
                "NameError",
                "OSError",
                "AttributeError",
            )
        }

        def proxy(name):
            def call(*args, **kwargs):
                try:
                    # A combined graph rejects aliases across args and kwargs.
                    encode((args, kwargs), trusted=True)
                    process.stdin.write(
                        dumps(
                            {
                                "name": name,
                                "args": encode(args, trusted=True),
                                "kwargs": encode(kwargs, trusted=True),
                            }
                        )
                    )
                    process.stdin.flush()
                except UnsupportedValue:
                    suspicion[0] = "unsupported_protocol"
                    raise UnsupportedProtocol from None
                except (BrokenPipeError, OSError):
                    raise CandidateFailure from None
                response = receive()
                if response == {"status": "integrity_violation"}:
                    violation()
                if response.get("status") == "unsupported_protocol" and len(response) == 1:
                    suspicion[0] = "unsupported_protocol"
                    raise UnsupportedProtocol
                if response.get("status") == "candidate_error" and set(response) == {
                    "status",
                    "kind",
                }:
                    raise errors.get(response["kind"], CandidateFailure)()
                if response.get("status") != "returned" or set(response) != {"status", "value"}:
                    violation()
                try:
                    return decode(response["value"])
                except InvalidMessage:
                    violation()

            call.__name__ = name
            return call

        namespace.update({name: proxy(name) for name in exports})
        try:
            exec(test, namespace)
            status = "passed"
        except CandidateFailure as error:
            status = "memory_limit" if isinstance(error, MemoryError) else "runtime_error"
        except AssertionError:
            status = "assertion_failure"
    except (UnsupportedProtocol, UnsupportedValue):
        status = "unsupported_protocol"
    except ProtocolViolation:
        status = "integrity_violation"
    except CandidateTimeout:
        status = "timeout"
    except CandidateFailure:
        status = "runtime_error"
    except BaseException:
        status = "controller_error"
    finally:
        if suspicion[0]:
            status = suspicion[0]
        if process is not None:
            process.kill()
            try:
                process.communicate(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        Path(payload["verdict_path"]).write_text(
            json.dumps({"nonce": payload["nonce"], "status": status})
        )


if __name__ == "__main__":
    main()
