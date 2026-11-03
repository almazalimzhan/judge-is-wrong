# When the Judge Is Wrong

## Abstract

We propose an empirical study of whether iterative revision against an LLM evaluator increases evaluator scores without corresponding gains in test correctness. Code tasks permit an external correctness measure, subject to the incompleteness of tests. The implementation compares single judges, ordinary critique, adversarial critique, ensembles, and public execution evidence. A no-feedback revision control and paired task-level analyses constrain interpretation. Real-model results are pending; mock execution is software validation only.

## 1. Introduction

An evaluator is an imperfect proxy for task quality. Optimizing against its feedback can improve solutions, reveal blind spots, or do both. We ask whether these effects can be distinguished in small, reproducible code-generation experiments. We do not claim that this setting captures general scalable oversight or establishes deceptive intent.

## 2. Related Motivation

The project connects conceptually to weak supervision, evaluator exploitation, reward hacking, scalable oversight, and adversarial evaluation. It is informed by [Gao et al. on reward-model overoptimization](https://arxiv.org/abs/2210.10760), [Zheng et al. on LLM judging](https://arxiv.org/abs/2306.05685), [Burns et al. on weak-to-strong generalization](https://arxiv.org/abs/2312.09390), and [Irving et al. on debate](https://arxiv.org/abs/1805.00899). Our setup is an inference-time revision experiment with executable tests, not a replication of those papers' training procedures.

## 3. Experimental Setup

Primary data are the official MBPP test tasks, deterministically sampled. One public test per task is exposed and remaining tests are withheld; this changes the original benchmark protocol. G is hidden-test fraction passed and Y indicates all hidden tests passed. J predicts Y; E=J−G is a diagnostic gap with different semantics from a calibration error. The same initial solution seeds all condition trajectories; three revisions follow. Only evaluator feedback is returned to the proposer. Hidden outcomes are recorded solely for offline analysis. A revision control receives no evaluator scores or summaries.

## 4. Evaluator Conditions

Single judge, ordinary critique then judge, adversarial critique then judge, independently requested judge ensemble, and judge with public execution evidence. Ensemble mean scores and majority approvals are primary aggregate outputs. Requests can have correlated errors. Intervention compute is not matched; final-candidate comparisons include changes to candidate distribution.

## 5. Metrics

Report final-round J, G, E, conditional false approvals/rejections, classification metrics, Brier and 10-bin ECE against Y, and initial-to-final changes. Resample task clusters with all conditions and rounds retained. Report paired differences and effect sizes with percentile 95% bootstrap intervals. Missing denominators remain missing. High-gap selection is descriptive, not a test of a causal mechanism. README and saved configs state the hypotheses and analysis choices.

## 6. Results

**Pending. No real-model experiment has been run.** Mock trajectories and any corresponding figures are deterministic software fixtures and provide no evidence for H1–H4. Use `scripts/generate_report.py` after an actual run to create a separate source-backed run report. Preserve this design document as the preregistration-style record, with a version hash.

## 7. Failure Analysis

Select J≥0.8 and G≤0.5; export candidate, public task, concise evaluation summaries and sanitized failure counts. Heuristic labels are provisional and need human confirmation. Never publish hidden-test source or reference solutions in case reports.

## 8. Limitations

Code tasks are a proxy; tests are incomplete and MBPP hidden scores are coarse. Public benchmark contamination may affect both roles. Mock imperfections are synthetic. Model families may share biases. LLM judging differs from human oversight. Small samples, multiple comparisons and provider changes limit generalization. Sandboxed subprocesses do not make same-interpreter grading secure against malicious introspection. Stronger evaluator comparisons need independent capability measurement. Resources are not compute matched.

## 9. Discussion

A positive gap increase would support an evaluation mismatch in this protocol. It would not imply deception, general evaluator unreliability, or a solution to scalable oversight. A null result could reflect ceiling effects, task difficulty, limited optimization, or effective oversight. Compare revision controls and fixed initial candidates before attributing causal effects to feedback.

## 10. Future Work

Cross-evaluate a frozen candidate corpus; repeat seeds; compare independently measured evaluator strength; use cross-family judges, compute-matched conditions, larger held-out suites, blinded human annotations, and contamination-resistant fresh tasks.
