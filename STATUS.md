# Status

## Current
Phase 2 (Langfuse tracing with restricted masking) committed 2026-09-25.
Plan: `_planning/plans/2026-09-25-phase-2-tracing.md`. Phase 3 (dataset + evaluators) next.

## Open items
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
Phase 3 - dataset + evaluators (plan awaiting Marko's OK).

Deferred test-reviewer findings (medium, Phase 2): positive control does not separate
masking-by-user from masking-by-retrieval (lead + public-only retrieval); pin sample_rate/tracing_enabled/sampler in
the capture fixture; no test for the fail-closed default; no test for restricted call failing both attempts.
