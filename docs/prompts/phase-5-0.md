Plan Phase 5 (regression gate). Wait for my OK.

Baseline:
- evals/baseline.json built from 20260928-noise-draft (6 repeats), marked "provisional" until I classify
  the facts_recall failures. It records provenance: run name, frozen config sha, judge key, dataset
  hashes, as_of, per-item rates, overall and per-category rates, and the noise numbers from stats noise.
- I set the gate parameters myself. Put them in baseline.json as fields I fill in: candidate_repeats,
  overall_margin_pts. Propose values from the Phase 4 numbers but do not write them.
- Updating baseline.json needs my explicit approval; the gate never rewrites it.

Gate rules, in this order, all reported even when an earlier one fails:
1. Safety: any safety failure by failure type in any candidate repeat -> fail.
2. Overall: paired mean delta (candidate item-level rate - baseline item-level rate) below -margin -> fail.
   Also show the paired bootstrap 95% interval, as information.
3. Stable-pass regression: an item that passed in all baseline repeats and failed in all candidate
   repeats -> fail, with the item named.
4. Per category and all pass->fail movers: reported, never gating.

Interfaces: `stats gate <baseline> <candidate>` with exit code 0/1 and a readable report, plus
tests/eval/test_regression_gate.py that reads the candidate path and fails with the same report.
Refuse candidates that are filtered, have a different frozen config, as_of, judge key or grading facts.

Validation (proves the gate can fail), with a prediction in DECISIONS.md committed before each run:
- no-change candidate: must pass
- degraded prompt (e.g. citation instruction removed): must fail on overall or stable-pass
- safety break (access filter applied after ranking): must fail on safety
The two broken candidates must never reach main: use a throwaway branch or an experiment-only flag,
and say which in the plan. One night script runs all three candidates (3 repeats each) with --resume.

Include a time estimate from the Phase 4 p50/p95 and unit tests for every gate rule with synthetic
results files (no model), each with a mutation check.