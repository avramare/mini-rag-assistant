# Status

## Current
Phase 0 + Phase 1 implemented (2026-09-24). Waiting for Marko's review.

## Open items
- Marko: review draft corpus in `data/docs/` (15 public incl. injection doc `supplier-newsletter-q3`, 5 restricted) before writing `evals/dataset.jsonl`.
- `.env` lives at repo root (`D:\ai-evals\.env`); `Settings` reads `mini-rag-starter/.env`. Move or copy it, then fill model names.
- test-reviewer findings not yet applied: decouple citation test from access filter, trim retrieval parametrize, add refusal-path / retry-context tests.
- Decision for Marko: should a non-refused answer with zero citations be rejected or only flagged for evals?

## Next
Phase 2 – Langfuse tracing.
