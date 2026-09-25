# Status

## Current
Phase 3 (dataset, two-pass runner, evaluators, report) implemented 2026-09-25, NOT committed yet.
Plan: `_planning/plans/2026-09-25-phase-3-evals.md`. Smoke run `20260925-smoke-draft` (draft dataset, 1 repeat):
80.0% item-level (n=30), versioning 25%, safety categories 100%/83.3%.

## Open items
- Judge does not know the run's `as_of` or the "latest effective version" rule: in the smoke run it failed correct
  answers ver-02 and ver-03 as "context has conflicting versions". Proposed fix (Marko to decide, Phase 6 territory):
  give the judge the as_of date and the version rule. Re-judge only needs `evaluate`.
- Smoke-run model failures: two-part questions answered only in part (fact-01, fact-05, restr-03b); ver-04 returned
  an empty answer with no citations (became an `uncited` refusal).
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

## Next
Marko reviews Phase 3 + draft dataset, then commit. Marko writes evals/dataset.jsonl. Then Phase 4 - noise.

