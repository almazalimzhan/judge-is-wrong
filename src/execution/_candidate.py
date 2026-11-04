"""Untrusted-side function server: no tests, expected values or verdict capability."""

import math
import os
import random
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _codec import MAX_BYTES, UnsupportedValue, decode, dumps, encode, loads
from _policy import PolicyViolation, candidate_builtins, check_source


def main():
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sink = open(os.devnull, "w")
    os.dup2(sink.fileno(), 1)
    sys.stdout = sink
    request = loads(sys.stdin.buffer.readline(MAX_BYTES + 1))
    seconds = math.ceil(request["timeout_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    memory = request["memory_mb"] * 1024 * 1024
    try:
        resource.setrlimit(
            resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS,
            (memory, memory),
        )
    except (OSError, ValueError):
        if sys.platform != "darwin":
            raise
    random.seed(0)
    namespace = {"__name__": "__candidate__", "__builtins__": candidate_builtins()}
    check_source(request["candidate"])
    candidate_code = compile(request["candidate"], "<candidate>", "exec")
    blocked = [False]

    def audit(event, arguments):
        sensitive = event in {
            "sys._getframe",
            "sys._current_frames",
            "sys.settrace",
            "sys.setprofile",
            "code.__new__",
            "function.__new__",
            "compile",
            "open",
        } or event.startswith(("ctypes.", "socket.", "subprocess.", "os.exec", "os.fork"))
        if event == "exec" and arguments[0] is not candidate_code:
            sensitive = True
        if sensitive:
            blocked[0] = True
            raise PolicyViolation("Unsupported runtime capability")

    sys.addaudithook(audit)

    def send(value):
        channel.write(dumps(value))

    def error(value):
        send({"status": "candidate_error", "kind": type(value).__name__})

    try:
        exec(candidate_code, namespace)
    except BaseException as exc:
        if blocked[0]:
            send({"status": "integrity_violation"})
        else:
            error(exc)
        return
    if blocked[0]:
        send({"status": "integrity_violation"})
        return
    send({"status": "ready"})
    for raw in sys.stdin.buffer:
        request = loads(raw)
        try:
            args, kwargs = decode(request["args"]), decode(request["kwargs"])
            before = (encode(args), encode(kwargs))
            function = namespace.get(request["name"])
            if not callable(function):
                raise NameError("Missing candidate function")
            result = function(*args, **kwargs)
            if blocked[0]:
                send({"status": "integrity_violation"})
            elif (encode(args), encode(kwargs)) != before:
                send({"status": "unsupported_protocol"})
            else:
                send({"status": "returned", "value": encode(result)})
        except UnsupportedValue:
            send({"status": "unsupported_protocol"})
        except BaseException as exc:
            if blocked[0]:
                send({"status": "integrity_violation"})
            else:
                error(exc)


if __name__ == "__main__":
    main()
