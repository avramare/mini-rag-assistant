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
14. **Supersession is explicit (`supersedes: <id>`) and checked on load: same or stricter access, strictly later effective date.** A public v2 of a restricted doc would leak it; an unordered pair has no "current" version. Explicit field over a `-v2` naming convention.
15. **Retrieval index cached on disk (`.cache/`), keyed by sha256 of embedding model + exact doc texts.** Re-embedding took ~6 s per CLI call; any doc edit or model change yields a new key, so a stale index is never reused. The cache holds vectors of restricted docs too, so it stays out of git.
16. **Access-filter tests carry `@pytest.mark.security` but stay in `tests/unit/`.** `pytest -m security` selects them without moving files; they use FakeLLM and real retrieval as the security rules require.
17. **Tests that pass for the wrong reason get a mutation check before they count.** A green test only has signal if some plausible bug turns it red. Four found in review rounds 2-3, each fixed and then proven by breaking the code on purpose:
    - *Vacuous `all()`*: `all(public for h in hits)` passes on `[]`, which is exactly what filter-after-top-k returns at k=1. Fix: assert `len(hits) == 1` too, plus the precondition that the restricted doc ranks first for the lead. Mutation "filter after top-k" -> `test_top1_for_restricted_best_match_is_one_public_doc` fails.
    - *Control that can't see its bug*: the `num_ctx=256` budget control passed on fixed prompt parts alone, even if docs were never counted. Fix: add a 30k-char restricted doc and assert the estimate grows by it. Mutation "estimate ignores docs" -> `test_estimate_counts_restricted_docs` fails.
    - *Assertion already enforced upstream*: "v2 is dated later" could never fail because the loader raises first; the real gap was a `-v2` doc without `supersedes`, which skips the check. Fix: assert every `-vN` doc declares `supersedes`. Mutation "drop `supersedes` from holiday-policy-v2" -> `test_every_versioned_corpus_doc_declares_what_it_supersedes` fails.
    - *Negative test that passes on empty input*: "`_TEMPLATE` is skipped" passed if the loader returned nothing; "rejected" tests matched only the file name, so any error passed. Fix: assert the fixture ids still load; match the rejection reason. Mutations "loader returns []" and "unknown target reported as a date problem" -> `test_loader_skips_underscore_files` and `...cannot_be_ordered_is_rejected[unknown-target]` fail.

    Also proven by mutation: dropping the model from the cache key, and not sending `num_ctx` to Ollama, each fail one targeted test. A first draft of the last mutation kept the text "unknown id" and survived -- the mutation, not the test, was wrong; mutations must remove the behaviour, not reword it.
