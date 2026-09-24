---
name: run-evals
description: Run an eval experiment and compare it to the baseline. Use when the user asks to run evals, check a prompt or model change, or measure noise.
---
# Run evals

1. Run `uv run python scripts/check_env.py`. If Ollama or Langfuse is not ready, stop and say what is missing.
2. Run `uv run pytest`. Do not run evals on a red fast suite.
3. Run the experiment:
   `uv run python -m evals.run_experiment --name <yyyymmdd>-<what-changed> --repeats 5`
4. Compare: `uv run python -m evals.stats compare results/<baseline>.json results/<new>.json`
5. Hand the output to the `eval-analyst` subagent for interpretation.
6. Report: overall and per-category pass rate with intervals, verdict (within noise / likely regression / not enough data),
   flipped items, and the Langfuse run name to open in the UI.

Never update `evals/baseline.json` unless the user explicitly says so in this conversation.
