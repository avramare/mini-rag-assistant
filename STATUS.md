# Status

## Current
Phase 1 committed, review round 2 applied (2026-09-24). Phase 2 not started.

## Open items
- Marko: fix `.env`. Values contain the key twice (`GEN_MODEL=GEN_MODEL=qwen3:4b`); check_env fails on it.
  Also add `NUM_CTX` / `MAX_PROMPT_CTX_SHARE` (see `.env.example`).
- Finding from first real run (n=1): lead asking "What is the approved budget for Project Orion?" got
  5 million (orion-budget-v2, effective 2027-01-01) instead of the in-force 4.2 million. The model ignores the
  "not after today" rule. Measure it in Phase 3 with repeats before changing the prompt.
- Generation takes ~15–27 s per answer on qwen3:4b (prompt ~550 tokens). Relevant for repeat counts in Phase 4.
- `uv run pytest` broke after the folder rename (stale venv launchers). Fixed with `uv sync --reinstall`;
  close VS Code's Python language server first, it locks `.pyd` files.

## Next
Phase 2 – Langfuse tracing.
