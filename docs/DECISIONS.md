# Decisions

One line per decision, with the reason. Newest at the bottom.

1. **Filter by access before ranking, not after top-k.** Otherwise restricted docs can take the top-k slots and a cleared-for-less user gets fewer (or zero) results, and a bug in the post-filter leaks.
2. **Invalid model output gets exactly one retry, recorded on the result.** Evals must be able to count format failures; unlimited retries would hide them.
3. **Citations outside the retrieved set are an error (`invalid_citation`).** A citation the user was never shown is either hallucinated or a leak.
4. **A non-refused answer with no citations is turned into a refusal (`refusal_reason="uncited"`).** An ungrounded answer must not reach the user, but the reason is kept so evals can count it apart from model refusals.
5. **Old and new versions of a policy coexist as separate docs with unique ids and an `effective` date.** Tests whether the model picks the current version; ids stay unique because citations and access control key on them.
6. **"Current" means latest effective date not after today; today is injected, not read from the clock.** A future-dated doc (Orion v2, 2027-01-01) is an "announced, not in force" trap; tests pin the date so they stay deterministic.
7. **`num_ctx` is sent explicitly on every Ollama call (default 4096, from `.env`).** Ollama silently truncates the start of an over-long prompt, dropping system rules and context without an error.
8. **Record `prompt_eval_count` and flag `truncation_risk` above `MAX_PROMPT_CTX_SHARE` (0.75) of `num_ctx`.** The real token count is the only runtime signal that a prompt is close to being truncated.
9. **Static worst-case prompt estimate uses chars/3.** English is ~4 chars/token, so /3 overestimates on purpose; a check that is too generous is worse than one that is slightly strict.
10. **Thinking is disabled only for models that report the `thinking` capability.** Thinking tokens cost time and are not part of the answer contract; sending `think` to other models is an error.
11. **Structured outputs with the `Answer` JSON schema (`additionalProperties: false`).** Constrained decoding cuts format failures at the source; validation with pydantic still runs as the contract check.
12. **Unit tests use explicit config values, never the local `.env`; the real configured values are checked by `scripts/check_env.py`.** Tests must give the same result on every machine.
13. **Orion "current budget" prompt is left unchanged after the first real run answered 5M instead of 4.2M.** One run is anecdote; measure the failure rate with repeats in the eval phase before changing the prompt.
