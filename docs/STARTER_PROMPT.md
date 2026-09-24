Read CLAUDE.md and docs/PLAN.md. We are starting Phase 0 and Phase 1.

Before writing code, give me a short plan: files you will create, dependencies, and the list of unit tests
with the risk each one protects against. Wait for my OK.

After implementation, run `uv run pytest` and `uv run ruff check .`, then use the test-reviewer subagent
on tests/ and show me its findings. Explain any Python or pytest feature I might not know
(fixtures, conftest, parametrize, markers) in 1-2 sentences each.
