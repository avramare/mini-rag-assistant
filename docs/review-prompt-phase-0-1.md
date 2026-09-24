Marko's review on Phase 0 and Phase 1 implementation:
- Corrected repo root folder name to `mini-rag-assistant` with correct `.env` file, also skills and agents from `.claude/` should work correct. 
- Reviewed draft corpus in `data/docs/`

Decisions on the review:
- Apply all 4 test-reviewer findings.
- Missing test 1: an answer with no citations and refused=false is rejected by the app and turned into a refusal
  with reason "uncited". Record the reason so evals can count it later.
- Add missing tests 2 (refusal path) and 3 (retry keeps system prompt and filtered context).

Also in OllamaClient: always set num_ctx explicitly (from .env, default 4096), disable thinking for models that
support it, use Ollama structured outputs with the Answer JSON schema, and return prompt_eval_count so we can
detect prompts close to the context limit. Add a unit test that fails when a prompt exceeds a configured share of num_ctx.

After that: run check_env, run the fast suite, run the test-reviewer agent (it should load now), and then ask
3 questions end to end with the real model: one public, one restricted as analyst, one restricted as lead.
Show me answers, citations and timings. Then give me a commit message for Phase 1. Do not start Phase 2.