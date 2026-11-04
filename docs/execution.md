# Generated-code execution

The executor **fails closed** if an OS isolation backend is unavailable. There is
no `unsafe` execution option. Run `Executor.check_available()` before spending API
calls. Each test has a fresh trusted controller and an OS-sandboxed candidate
process, in different directories. Candidate stdout and stderr are discarded. Results
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

## Ground-truth boundary: function-rpc-v1

The trusted controller compiles and executes benchmark tests. It never executes
candidate source. The candidate process receives only candidate source and
function-call arguments; hidden tests, expected values, nonce and verdict path
stay in the controller. Its filesystem policy denies the controller directory.
The controller makes assertions on decoded plain values, then writes the
nonce-checked verdict. Candidate messages cannot directly declare a test passed.
The candidate remains alive across calls within a test, retaining ordinary
function global state. Each next test starts fresh.

The wire format supports exact builtin scalar/container values with bounded
size/depth, using JSON tags, never pickle or arbitrary object deserialization.
NumPy scalars from trusted tests are normalized to builtin values. Custom
objects, generators, argument mutation, cyclic/shared container inputs and
observable object identity are unsupported. These cases produce
`unsupported_protocol` with `oracle_valid=false`; they are not usable
correctness labels. All statistical entry points reject such rows. Numeric
scores in invalid raw diagnostic rows must not be interpreted as correctness.

Because candidate code shares an interpreter with the candidate-side transport,
separation alone cannot protect that transport against unrestricted reflection.
Before execution a conservative AST policy rejects runtime-control imports,
private/reflection attributes, dynamic execution, file I/O, classes and
generators. Imports use facades for a closed list of standard-library modules;
candidate builtins exclude reflective and I/O capabilities. Runtime audit hooks
mark prohibited operations as sticky integrity violations, even if candidate
code catches the exception. Policy rejection does not establish malicious intent;
it can also reject legitimate Python outside the supported subset.

Regression tests cover the original frame/verdict attack, transport encoder
replacement, builtin/namespace modification, runtime controls, unsupported
values/aliases/mutation and malformed trusted setup. OS filesystem/network
denial is also tested directly, independently of the feature policy.

**This is not a proof of secure execution for arbitrary adversarial Python.**
Obfuscated policy bypasses, native/runtime vulnerabilities, attacks on the
transport implementation, OS/container escapes and resource exhaustion remain
unsupported threat models. A hardened adversarial service requires a separately
audited restricted runtime or VM. The trusted benchmark source and installed
test dependencies are assumed trustworthy. Do not run arbitrary uploaded test
code as trusted benchmark code.

The wall watchdog includes a bounded startup allowance (1.5 seconds on macOS,
7 seconds with Docker) in addition to the configured test budget; CPU limits
are applied separately. Whole-suite timings therefore differ from official
benchmark runners. Wall timeout kills the local process group, and Docker
cleanup removes its container. No unsafe same-interpreter fallback exists.

Earlier result files from the same-interpreter harness remain readable for
compatibility but reports explicitly label them `legacy_unverified`. Re-execute
their candidates with this protocol before interpreting correctness labels.

Neither backend is a proof against OS/container vulnerabilities or resource
exhaustion. macOS memory accounting is explicitly incomplete. Large workloads
can fill the per-test directory across many files despite a per-file size limit.
Tests are sparse proxies for full functional correctness, and independent fresh
test processes may miss bugs involving persistent state across calls except
where the benchmark's individual test code exercises it.

Relevant primary implementation references:
[Python resource limits](https://docs.python.org/3/library/resource.html) and
[Docker run isolation options](https://docs.docker.com/reference/cli/docker/container/run/).
