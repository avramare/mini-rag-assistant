---
name: run-evals
description: Run an eval experiment and compare it to the baseline. Use when the user asks to run evals, check a prompt or model change, or measure noise.
---
# Run evals

1. Run `uv run python scripts/check_env.py`. If Ollama or Langfuse is not ready, stop and say what is missing.
2. Run `uv run pytest`. Do not run evals on a red fast suite.
3. Validate the dataset: `uv run python -m evals.dataset validate <dataset path>`.
4. Pass 1 (gen model): `uv run python -m evals.run_experiment generate --name <yyyymmdd>-<what-changed> --repeats 5 --dataset <path>`.
   If it crashes, continue with `--resume`; never delete a partial results file to start over without asking.
5. Pass 2 (judge): `uv run python -m evals.run_experiment evaluate --name <run>`. Re-run only this pass to re-judge.
6. Report: `uv run python -m evals.stats report results/<run>.json`.
   Compare: `uv run python -m evals.stats compare results/<baseline>.json results/<new>.json`
   (refuses runs with different `as_of` or graded against different dataset facts).
7. Hand the output to the `eval-analyst` subagent for interpretation.
8. Report: overall and per-category pass rate with intervals, verdict (within noise / likely regression / not enough data),
   flipped items, and the Langfuse run name to open in the UI.

Never update `evals/baseline.json` unless the user explicitly says so in this conversation.
