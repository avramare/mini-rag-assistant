Before committing Phase 2:
- Add run config to trace metadata: k, num_ctx, gen and embed model, a hash of SYSTEM_PROMPT, and the corpus hash
  (reuse the one from the index cache key). Add an optional dataset_item_id field that Phase 3 will fill.
  Extend the masking tests so config metadata is present on both masked and unmasked traces.
- Add the observations API finding to docs/PLAN.md under Phase 3.
- Run test-reviewer on the files changed in Phase 2 only. Show findings; apply high severity only.
- Commit as "feat: phase 2 tracing with restricted-context masking".

Then plan Phase 3 and wait for my OK. Constraints:
- run_experiment works in two passes: first generate and save all answers to results/, then run evaluators
  and the judge on the saved answers. Only one large model is in memory at a time, and the judge can be
  re-run without regenerating.
- Build and test everything against evals/dataset.example.jsonl; I will add the real dataset.
- Results must keep the full run config so runs can be compared.
- Include a way to report per-category pass rates, so the versioning category (Orion trap) is visible on its own.