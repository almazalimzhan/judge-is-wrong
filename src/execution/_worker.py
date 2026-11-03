"""Trusted launcher. Invoked only by Executor inside an OS sandbox."""

import json
import math
import os
import random
import resource
import signal
import sys


def main() -> None:
    random.seed(0)
    payload = json.load(sys.stdin)
    trusted_exec = exec
    trusted_write = os.write
    trusted_close = os.close
    trusted_os_open = os.open
    trusted_setitimer = signal.setitimer
    verdicts = {
        status: json.dumps({"nonce": payload["nonce"], "status": status}).encode()
        for status in [
            "passed",
            "assertion_failure",
            "memory_limit",
            "syntax_error",
            "runtime_error",
            "timeout",
        ]
    }
    timeout = int(math.ceil(payload["timeout_seconds"]))
    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    memory = payload["memory_mb"] * 1024 * 1024
    # Darwin RLIMIT_AS is advisory/unavailable on some releases. RLIMIT_DATA is
    # the available best-effort heap limit; the timeout remains mandatory.
    memory_resource = resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS
    try:
        resource.setrlimit(memory_resource, (memory, memory))
    except (OSError, ValueError):
        if sys.platform != "darwin":
            raise
        # macOS can reject both DATA and AS limits. This limitation is explicit
        # in docs/execution.md; Docker is required for a hard memory boundary.

    def wall_timeout(signum, frame):
        raise TimeoutError("wall time")

    signal.signal(signal.SIGALRM, wall_timeout)
    signal.setitimer(signal.ITIMER_REAL, payload["timeout_seconds"])
    namespace = {"__name__": "__candidate__"}
    status = "passed"
    try:
        candidate = compile(payload["candidate"], "<candidate>", "exec")
        setup = compile(payload["setup"], "<test-setup>", "exec")
        test = compile(payload["test"], "<test>", "exec")
        trusted_exec(candidate, namespace)
        trusted_exec(setup, namespace)
        trusted_exec(test, namespace)
    except TimeoutError:
        status = "timeout"
    except AssertionError:
        status = "assertion_failure"
    except MemoryError:
        status = "memory_limit"
    except SyntaxError:
        status = "syntax_error"
    except BaseException:
        status = "runtime_error"
    # No candidate stdout, exceptions, or test expressions cross this boundary.
    # A successful exit without this authenticated completion is NOT a pass.
    trusted_setitimer(signal.ITIMER_REAL, 0)
    descriptor = trusted_os_open(
        payload["verdict_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
    )
    trusted_write(descriptor, verdicts[status])
    trusted_close(descriptor)


if __name__ == "__main__":
    main()
