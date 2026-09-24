Round 3 decisions:
- Finding 1: merge as you proposed (k=1 test takes over the ranking check, delete the other).
- Finding 2: keep the restricted-citation test separate.
- Findings 3-7: apply.
- Finding 8: set FIXED_TODAY to 2026-07-15 (not today, and before Orion v2 is effective).
  Tests that need a date after it set their own.
- Add missing tests 1 (OllamaClient with httpx.MockTransport) and 2 (access change with a cached index).
- Add a DECISIONS.md entry: "tests that pass for the wrong reason", with the 4 examples from rounds 2-3
  and the mutation checks that proved the fixes.

Also update .claude/agents/test-reviewer.md: at most 5 findings, each with severity (high/medium/low) and the
concrete bug the test would miss; report low only if asked; review only new or changed code.

This is the last review round for Phase 1. Commit as "test: review round 3", then plan Phase 2 and wait for my OK.