# Status

## Current
Phase 3 committed 2026-09-25 (68dcd33) plus follow-ups (judge as_of fix, expected_docs, retrieval_recall,
judge v3, per-judge-version evaluations, resumable evaluate).
Smoke run `20260925-smoke-draft` (draft dataset, 1 repeat) holds two evaluations: judge v2 `797638a132e2` 83.3%
and judge v3 `ad47ba4fb182` 86.7% item-level (n=30); v3: versioning 75%, safety categories 100%/83.3%.

## Open items
- Judge cannot tell which Orion version is in force AFTER 2027-01-01: the after-v2 eval controls fail
  (`pytest -m eval tests/eval/test_judge_controls.py`: 2 failed, 2 passed). It passes "4.2M is current" on
  2027-03-01 (8/8 samples on v3, 5/5 on v2) and sometimes fails "5M" (reasons misorder the dates).
  `forbidden_absent` still catches a wrong version in the dataset run. Proposed (Marko to decide): compute the
  version in force in code and label docs in the judge prompt; own prediction first (DECISIONS 30).
- Judge v3 reasons are long ("But wait, ..." loops); the 30-answer evaluate took ~50 min on CPU. Watch the
  timeout count in the report; raise `OLLAMA_READ_TIMEOUT_S` and `evaluate --resume` if needed.
- Langfuse: judge scores sent before 19400a5 use the old id (trace + name) and hold v3 values; the next evaluate
  adds v3 scores under the new id next to them (DECISIONS 31).
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
Marko decides on the judge's in-force labelling (after-v2 controls). Marko writes evals/dataset.jsonl
(with expected_docs). Then Phase 4 - noise.

