Plan approved with these changes:

1. Effective dates: "latest effective date" must mean latest date that is not in the future.
   - Inject the current date (a clock/today provider in Settings or Assistant), put "Today is YYYY-MM-DD"
     in the prompt, and fix the date in tests.
   - holiday and remote-work pairs: both versions in the past (old 2025-01-01, new 2026-06-01).
   - orion pair: keep v2 in the future on purpose, as a "announced but not yet effective" trap.
     Adjust its text so this is consistent.
   - Update the expected answers in step 6 accordingly and add one question for the orion trap.
   - Unit test: with a fixed date before and after v2's effective date, the prompt says today's date correctly.

2. Context budget:
   - The unit test uses explicit num_ctx values, not the real .env. Keep the num_ctx=256 positive control.
   - Move the check of the real configured value into check_env.py.
   - Add a runtime flag on AnswerResult: truncation_risk = prompt_tokens > max_prompt_ctx_share * num_ctx,
     using the real prompt_eval_count. Unit test it with FakeLLM returning token counts.

3. I changed .claude/settings.json to deny only ./.env. Add NUM_CTX and MAX_PROMPT_CTX_SHARE to .env.example yourself.

Everything else as planned.