# Status

## Current
Phase 3 committed 2026-09-25 (68dcd33) plus follow-ups through 2026-09-27: judge v4 (in-force labels computed
in code), judge v5 (3-sentence reason, output cap 256), judge cost in the report, filtered dev runs
(`generate --items/--category`).
Smoke run `20260925-smoke-draft` (draft dataset, 1 repeat) holds four evaluations; v3, v4 and v5 all 86.7%
item-level (26/30, n=30), identical pass/fail. Latest = v5 `3870776dcad7-np256`.

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

## Next
Marko reviews DECISIONS 33-35 and decides on keeping v5 / the version-item code check. Marko writes
evals/dataset.jsonl (with expected_docs). Then Phase 4 - noise.

