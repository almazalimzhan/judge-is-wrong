# Dataset boundary and test construction

The primary experiment uses MBPP's **official evaluation IDs 11–510**. Adapters
read downloaded local files; the smoke test instead uses ten original synthetic
fixtures, labelled `toy/*`. Synthetic fixtures are software checks, not empirical
evidence about language models.

## MBPP

[Google Research's dataset documentation](https://github.com/google-research/google-research/tree/master/mbpp)
defines the train/prompt/validation/test task groups. This adapter filters to the
official test IDs even when the downloaded file also contains the other groups.
The original JSONL and sanitized JSON array formats are supported. No reference
implementation is returned by the adapter.

MBPP does not provide the public/hidden separation needed here. For each task, the
public test is the index whose SHA-256 digest of
`mbpp-public-v1|<seed>|mbpp/<task_id>|<index>` is lexicographically smallest. The
remaining tests are held out from both proposer and judge. This is deterministic,
disjoint by index, and versioned; with three tests there is one public test and two
hidden tests, so G has the coarse support {0, 0.5, 1}. Repeated or correlated test
contents can still reduce effective coverage. An index split is not a guarantee
of semantic independence.

Public task text contains the task description and function signatures extracted
from the reference AST, with implementation bodies removed. Signatures are a
reference-derived interface aid; helper names can reveal implementation hints.
The entry point is selected using the public test only. Public test setup/imports
are included because they are necessary to run that test. This setup is treated
as public benchmark metadata. We exclude challenge tests rather than silently
mixing different test regimes.

The holdout is *procedurally hidden in this experiment*, not private in the
original public benchmark. A pretrained model may have memorized all tests or
reference solutions. This design cannot rule out benchmark contamination.
MBPP is distributed under CC BY 4.0: credit Austin et al. (2021), Google Research,
retain upstream attribution, and state the above transformation when sharing
derived benchmark content. Dataset files remain ignored by version control.

## HumanEval

[OpenAI HumanEval](https://github.com/openai/human-eval) is an alternative adapter
under its [MIT license](https://github.com/openai/human-eval/blob/master/LICENSE).
The complete original prompt is public. Simple expression/literal-output doctest
examples are additionally converted into public assertions. Unsupported doctest
forms are omitted from execution evidence, while remaining visible in the
original problem statement. Some tasks therefore have no executable public tests;
zero public tests means **unavailable evidence**, never evidence of correctness.

The original `check(candidate)` harness is preserved as one hidden test unit.
Splitting arbitrary dependent loops/setup into individual assertions could alter
semantics. Consequently HumanEval G is binary at harness granularity, and its
partial-correctness scores are not comparable to MBPP's two-test G. Do not pool
these adapters in one analysis. The candidate must be a complete module with
required imports, rather than a function-body completion.

## Leakage controls

`Task.public_view()` is an explicit field allowlist. It excludes all hidden tests,
reference code, and metadata. Execution results expose only counts, status
categories, runtime, and scores; no assertion source, exception message, or
candidate stdout enters judge/proposer feedback. Hidden outcomes are recorded
for offline analysis only. Test fixtures verify this boundary with sentinel data.

Dataset sampling uses a local seeded RNG after sorting task IDs. Archive the raw
dataset digest, split seed, adapter version/source revision, and selected task IDs
with every run. A seed by itself does not reproduce a changed upstream file.

## Reference preflight and the first-run exclusion

Reference solutions are checked against their original tests before any paid
model experiment. On the development macOS/Python 3.14 platform, `mbpp/180`'s
reference solution passes only one of its three original tests. Its other two
assertions compare floating-point results with exact equality. A diagnostic
execution that temporarily replaces those equality checks with
`math.isclose(rel_tol=1e-12, abs_tol=1e-9)` passes all three tests. This supports a
numerical-brittleness explanation; it does not justify changing the benchmark's
scoring rules silently.

The first-real-run configuration explicitly excludes `mbpp/180` after the seeded
shuffle and before taking the first 75 eligible tasks. Other tasks keep their
relative order; the next eligible task replaces the excluded one.
Original benchmark tests remain unchanged. The exclusion is an oracle-integrity
decision made using reference behavior, before observing proposer outcomes.
Record it in the run configuration/manifest and state the reduced eligible task
pool in reports. The diagnostic tolerance is not used for experiment scoring.
Recheck reference behavior on a different Python/OS runtime and report any
platform-specific exclusions rather than assuming that this issue generalizes.

## Optional expanded-suite protocol: HumanEval+

`dataset.name: humaneval_plus` loads the official original-format
[HumanEval+ v0.1.10 release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10).
This is a distinct dataset protocol, not an augmentation fitted to our proposer
failures. The downloaded gzip is SHA256-pinned to
`daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5`.
The public prompt/doctest examples remain public. The entire expanded
`check(candidate)` is hidden and preserved as one dependent test unit, so **G is
binary**. We do not split arbitrary loops or regenerate expected outputs.
The format is explicitly provided for compatibility by the upstream release.

[EvalPlus](https://github.com/evalplus/evalplus) supplies expanded test coverage
and records historical corrections to benchmark contracts/oracles. Its release
repository uses [Apache-2.0 with underlying HumanEval MIT attribution](https://github.com/evalplus/humanevalplus_release/blob/main/LICENSE).
The downloader retains both licenses locally. Cite Liu et al. (2023),
[Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210),
and Chen et al. (2021). Benchmark files and hidden harnesses stay ignored by Git.
NumPy is required by the trusted expanded harness (`[analysis]` or `[all]` extra).

Run `validate_dataset.py --config configs/stronger_suite.yaml` before generation.
It checks upstream references using this repository's execution boundary and
flags unsupported protocol cases separately. Coverage and time limits matter:
an RPC boundary has overhead, so the optional config allows ten seconds per
harness. This is not a reproduction of EvalPlus's official timing/evaluation
implementation; do not present these scores as official leaderboard results.

Sparse MBPP holdouts can falsely accept incorrect programs and produce coarse
G values; even all hidden tests passing does not establish semantic correctness.
An expanded public benchmark remains vulnerable to contamination, correlated
tests, faulty expected values, specification ambiguity and platform differences.
Neither suite is a perfect oracle. Freeze benchmark bytes, split, allowed
protocol, exclusions and reference-preflight policy before a confirmatory run.
Do not create or select new tests from failures in that confirmatory sample.
Any such diagnosis belongs to an explicitly exploratory follow-up sample.
Never pool MBPP fractional G with whole-harness HumanEval+ G without reporting
their distinct estimands. Oracle-invalid records are not ordinary incorrect
solutions and are rejected by the analysis entry points.
