# Generated-code execution

The executor **fails closed** if an OS isolation backend is unavailable. There is
no `unsafe` execution option. Run `Executor.check_available()` before spending API
calls. Each test runs in a fresh isolated Python process, with a fresh namespace
and temporary directory. Candidate stdout and stderr are discarded. Results
contain only pass counts and coarse failure categories.

## macOS

`backend: auto` chooses macOS `sandbox-exec` on Darwin. The deny-by-default Seatbelt
profile denies network access, host-user-file reads, and writes outside the
per-test temporary directory. It permits interpreter/stdlib and OS library reads,
Homebrew runtime reads when applicable, exact ancestor traversal, and access to
the test's temporary directory. Broad `/opt/homebrew` read access is required for
this development interpreter's dylibs; do not place secrets in that prefix.
Process execution is limited to the interpreter runtime. Process forking is not
allowed by the profile.

The child inherits only a minimal PATH, locale and fixed hash seed. No API keys or
other parent secrets are forwarded. Limits cover wall time, CPU time, open files,
file size, and core dumps. DATA/AS memory limits are attempted on macOS, but this
development machine rejects both. **There is no guaranteed macOS memory cap.**
Use the Docker backend when a hard memory boundary is required.

Python starts with safe-path, no-user-site, and no-site flags (`-P -s -S`). The
sanitized environment fixes `PYTHONHASHSEED=0` (also explicitly inside Docker),
and each worker seeds the standard-library `random` module to zero. Candidate
time access, entropy sources, concurrency or explicit reseeding can still cause
nondeterministic behavior; these controls are not a determinism guarantee.

`sandbox-exec` is an older platform facility and can be unavailable on future
macOS versions. A containing application sandbox can prevent it from applying its
own profile. In that case the CLI exits with `SandboxUnavailable`; run from a
normal terminal or use Docker. This is not permission to disable isolation.

## Docker (Linux or explicit macOS backend)

Install/start Docker, then explicitly pull the configured image before running:

```sh
docker pull python:3.11-slim
```

The executor uses `--pull=never`, `--network=none`, a read-only root filesystem,
an unprivileged user, dropped capabilities, no-new-privileges, PID/CPU/memory
limits, a bounded temporary filesystem, and one per-test writable bind mount.
Containers receive unique names; forced cleanup runs even after a timeout so
killing the Docker CLI does not leave candidate processes running. For archived
experiments use an immutable image digest through `docker_image` rather than
relying on a mutable tag. Docker daemon access itself is privileged: run on a
dedicated machine/VM appropriate for generated-code research.

Only the macOS backend was exercised in the development environment; Docker
support requires validation on the intended host. A missing daemon/image causes
a preflight failure rather than a fallback to ordinary subprocess execution.

## Integrity and remaining limitations

The worker writes a nonce-checked completion after executing the trusted tests.
Successful process exit alone, printed fake verdicts, and a simple monkeypatch
of `builtins.exec` do not produce a passing result. Wall time kills the entire
local process group; Docker cleanup kills container children as well.

**OS host isolation is different from tamper-proof ground truth.** Candidate code
and tests still share one Python interpreter. A deliberately malicious candidate
can inspect stack frames or mutate interpreter state; the random completion
nonce is a reliability guard, not a cryptographic separation from that code.
This runner does not establish oracle integrity against an adversarial Python
program trying to subvert the harness. Review anomalous candidates and use a
stronger function-call RPC/hypervisor harness for that threat model. The proposer
is asked to optimize evaluator feedback, not to attack the test harness.

Neither backend is a proof against OS/container vulnerabilities or resource
exhaustion. macOS memory accounting is explicitly incomplete. Large workloads
can fill the per-test directory across many files despite a per-file size limit.
Tests are sparse proxies for full functional correctness, and independent fresh
test processes may miss bugs involving persistent state across calls except
where the benchmark's individual test code exercises it.

Relevant primary implementation references:
[Python resource limits](https://docs.python.org/3/library/resource.html) and
[Docker run isolation options](https://docs.docker.com/reference/cli/docker/container/run/).
