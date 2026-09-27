# Status

## Current
Phase 4 (noise) tooling built 2026-09-27 (DECISIONS 36-38): frozen config (`freeze`, `--frozen`), prompt-eval vs
reply timing, `rejudge` judge samples, `stats noise`, paired-bootstrap `compare`, Langfuse-failure bookkeeping +
`publish`, idempotent `--resume`, `scripts/night_run.ps1`. 241 tests.
Dry run `20260927-dry` (ver-01, fact-01; R = 2, K = 1; frozen): killed mid-generate and mid-evaluate, re-run
with the same command -> 4 answers, 4 grades, 2 sample grades, all unique, 0 Langfuse failures.
`evals/frozen/phase4-draft.json` written by `freeze`, NOT committed yet (Marko reviews and commits).

## Open items
- Judge controls (`pytest -m eval tests/eval/test_judge_controls.py`, 6 tests): v5 6/6 in one pytest round and
  32/32 samples before the cap; v4 was lenient on "superseded 4.2M presented as current" in 4 of 26 samples.
  2 rounds only -- keep running the controls on every judge change (DECISIONS 33-35). `forbidden_absent` still
  catches a wrong version independently of the judge.
- Judge cost: v5 did NOT speed up the judge (smoke p50 57.7 s vs v4 47.8 s; tokens p50 99 vs 79); one reply hit
  the 256 cap (ver-02, retried). Proposed, not done: measure prompt-eval vs reply time
  (`prompt_eval_duration`/`eval_duration`) before any further cost change (DECISIONS 35).
- Dev run `20260927-dev-versioning` (filter check, 4 items, 1 repeat): versioning 1/4 vs 3/4 in the smoke run on
  the same questions -- regenerated answers, not a judge change. ver-01 answered the superseded version
  (`forbidden_absent` and the v5 judge both failed it). More evidence for measuring with repeats in Phase 4.
- Proposed (Marko to decide): a code check for version items in the judge's place (DECISIONS 34 outcome).
- Langfuse: judge scores sent before 19400a5 use the old id (trace + name); later evaluates add scores under the
  per-judge-version id next to them (DECISIONS 31).
- `results/20260925-smoke-draft.judge-v2.bak.json`: local backup taken before the v3 re-run; its evaluation is now
  merged into the results file. Safe to delete.
- Smoke-run model failures: two-part questions answered only in part (fact-01, fact-05, restr-03b) -- all
  generation failures, the expected doc was retrieved each time; ver-04 returned an empty answer with no citations
  (became an `uncited` refusal).
- Contract change to measure AFTER Phase 4, do not change now: ver-04's empty answer with no citations is turned
  into an `uncited` refusal (DECISIONS #4), so it is graded as a wrong refusal, not as an invalid output. Decide
  whether an empty `answer` on a non-refused reply should be a contract violation (`invalid_output`, counted and
  retried once) instead. Measure how often it happens with repeats first; changing it now would move the numbers
  Phase 4 is supposed to baseline.
- Finding: lead asking "What is the approved budget for Project Orion?" answered 5 million
  (orion-budget-v2, effective 2027-01-01) instead of the in-force 4.2 million in both real runs so far
  (2026-09-24, 2026-09-25; n=2). The model ignores the "not after today" rule. Measure with repeats in
  Phase 3/4 before changing the prompt.
- Generation takes ~15-29 s per answer on qwen3:4b (prompt ~550 tokens). Relevant for repeat counts in Phase 4.
- Langfuse org is v4-only: legacy `/api/public/traces` returns 410 (sunset for all 2026-11-16). Phase 3/4 code
  that reads results must use `api.observations.get_many` (`/api/public/v2/observations`) or v2 metrics.
- `uv run pytest` broke after the folder rename (stale venv launchers). Fixed with `uv sync --reinstall`;
  close VS Code's Python language server first, it locks `.pyd` files.

- Dry-run timing differs a lot from the smoke run: generation p50 4.2 s (smoke 33.1 s), prompt eval p50 0.2 s
  once the model is warm; judge p50 42.9 s (smoke 57.7 s). 4 answers only. If it holds, R = 6 + K = 3 takes
  ~2.5 h, not 4.7-5.9 h, and R could go up. Marko decides R.
- Dry run already showed ver-01 (Orion) answering the not-in-force 5 million in 2 of 2 repeats, and one judge
  verdict flip (fact-01 r1: 4 -> 3) in 1 re-sample.
- test-reviewer pass on the Phase 4 tests: before the phase is marked done.

## Next
Marko: review + commit `evals/frozen/phase4-draft.json`, confirm R/K, run the night-run checklist
(DECISIONS 38), start `scripts/night_run.ps1`. Then interpret `stats noise` and propose the Phase 5 margin.
Marko writes evals/dataset.jsonl (with expected_docs) in parallel.

