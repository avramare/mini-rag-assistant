Decisions on 3.3:
1) Keep judge v5 and freeze it for Phase 4. Record in DECISIONS that v5 is safer (cap, truncation
   detection), not faster, and that the 3-sentence rule made replies longer than v4.
2) Version correctness stays with forbidden_absent. No more judge work on versioning before Phase 6.
3) Prompt-vs-reply timing becomes part of Phase 4 reporting (prompt_eval_duration vs eval_duration, p50/p95).

Now plan Phase 4 (noise) on the draft dataset. Constraints:
- Frozen config: judge v5, gen model digest, dataset, as_of, k, num_ctx. Refuse to run if anything differs.
- One run with R repeats (propose R from measured p50/p95 so generation + judging fit in ~6 hours).
  Each repeat is a full pass and already its own Langfuse run.
- Generation noise: overall pass rate per repeat (spread = run-to-run noise), item-level pass rate
  across repeats, Wilson intervals on item-level rates (n = items), list of unstable items
  (pass rate strictly between 0 and 1), per category with versioning and the Orion item called out.
- Judge noise, separately: re-judge one repeat's saved answers K times with the same judge version,
  stored as samples (not overwriting); report per-answer verdict flip rate.
- compare: paired analysis on the same items (per-item deltas, paired bootstrap interval of the mean
  difference), in addition to the existing checks.
- Propose how the noise margin for Phase 5 is derived from these numbers. Do not write baseline.json
  or thresholds; I decide them.
- Night-run checklist in the plan: check_env, power/sleep, nothing else on the CPU, resume on crash.

Wait for my OK on the plan.