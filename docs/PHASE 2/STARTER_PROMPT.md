First run scripts/check_env.py now and show me its actual output. Add a rule to CLAUDE.md:
status about the environment (.env, models, services) must come from a check run in the same turn, never from earlier reports.

Then plan Phase 2 (tracing) with these constraints:
- Check the current Langfuse Python SDK docs before planning, including its masking option.
- If any retrieved doc in a call is restricted, mask prompt and answer text in the trace. Keep doc ids,
  access levels, token counts, durations, truncation_risk, refusal_reason and model name.
- Security test: lead asks about Orion with tracing enabled through a fake exporter or the masking function;
  assert that neither the codename nor the budget figure appears in anything sent out.
- Tracing stays a no-op without Langfuse keys, and the fast suite must not need network.
- Add a DECISIONS.md entry for the masking trade-off (safer vs. harder to debug restricted cases in the UI).

Wait for my OK on the plan.