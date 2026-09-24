Commit Phase 1 now with your message. Add _planning/ to .gitignore, rename the review prompt file to
docs/review-prompt-phase-0-1.md, and start docs/DECISIONS.md with the decisions so far (one line each + why).

Then, as a separate commit, apply review round 2:
- Finding 1: the budget test uses the real data/docs corpus with an explicit num_ctx=4096 (not .env).
- Finding 2: keep the k=1 case. Assert exactly one result and that it is public (filter-after-top-k would
  return an empty list, which all() passes vacuously). For k=10 assert the exact set of documents.
- Findings 3-7 as proposed; for 3 use the Orion question as analyst and assert the secret is not in the retry prompt.
- Add the restricted-citation test. Mark it and the existing access-filter tests with @pytest.mark.security.
- Add the corpus invariant: a doc that supersedes another has the same or stricter access level and a later date.
- Cache the retrieval index on disk keyed by a hash of the doc contents and the embedding model name.
  Test that changing a doc or the model invalidates the cache.

Leave the Orion prompt as it is. Run the fast suite, `pytest -m security` and the test-reviewer, then show me the results.