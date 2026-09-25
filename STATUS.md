# Status

## Current
Phase 3 committed 2026-09-25 (68dcd33) plus follow-ups (judge as_of fix, expected_docs, retrieval_recall).
Smoke run `20260925-smoke-draft` (draft dataset, 1 repeat), re-evaluated: 83.3% item-level (n=30),
versioning 50%, safety categories 100%/83.3%.

## Open items
- Judge still fails ver-02 (Orion as_of 2027-03-01, correct 5M answer): score 2 in 5/5 samples. Its reason says 5M
  is in force, then scores 2 -- `score` precedes `reason` in the JSON, so it commits before reasoning. Proposed
  (Marko to decide): reason before score + a control for the after-v2 side, with a written prediction first.
- Judge has no control for "as_of after v2, new figure correct"; existing controls cover only before-v2.
- A judge call hit the 120 s Ollama read timeout once (manual re-sample); `evaluate` does not catch it, so a
  timeout would abort the pass. Local results are saved only at the end of pass 2.
- `stats report` header shows the generation-time dataset sha256, not the one the evaluation used
  (`evaluation.dataset_sha256`); after a facts/docs edit they differ.
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
Marko decides on the judge reason-before-score change. Marko writes evals/dataset.jsonl (with expected_docs).
Then Phase 4 - noise.

