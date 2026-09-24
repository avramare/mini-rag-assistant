# mini-rag-assistant

Interview-prep project for a **Senior QA and AI Evals Engineer** role. The app itself is deliberately small.
The point of the repo is the **quality system around it**: deterministic tests, an eval pipeline in Langfuse,
statistics that separate signal from noise, a regression gate, and authorization tests.

The owner (Marko) is an SDET (Playwright/TypeScript background) learning Python/pytest and LLM evals here.
Explain non-obvious Python and statistics choices briefly in PR descriptions or comments — he wants to understand, not just receive code.

## What the app does
A question-answering assistant over ~20 short Markdown documents in `data/docs/`.
Each document has an access level. Users have clearances. The assistant must answer only from documents
the user is allowed to see, cite them, and refuse when the context does not contain the answer.

## Stack
- Python 3.12, `uv` for env and deps, `pytest`, `ruff`, `pydantic`
- Ollama (local) for generation, embeddings and the LLM judge. Model names come from `.env`, never hard-coded.
- Langfuse (Python SDK) for tracing, datasets, experiments and scores. **Check the current Langfuse SDK docs
  before writing integration code** — the API has changed between major versions.
  Our Langfuse Cloud org is v4-only: the legacy `/api/public/traces` API returns 410. Read data with
  `api.observations.get_many(...)` (`GET /api/public/v2/observations`) or `/api/public/v2/metrics`.
- OS: Windows (PowerShell). Use `pathlib`, no bash-only commands in scripts or docs.

## Architecture
```
src/mini_rag/
  config.py        # settings from .env (pydantic-settings)
  documents.py     # load data/docs/*.md (skip files starting with "_") with YAML frontmatter: id, title, access
  users.py         # users and clearances from data/users.yaml
  retrieval.py     # embeddings + cosine top-k (numpy, no vector DB); ACCESS FILTER BEFORE RANKING
  llm.py           # LLMClient Protocol; OllamaClient; FakeLLM for deterministic tests
  assistant.py     # answer(question, user) -> Answer
  tracing.py       # Langfuse wiring, safe no-op when keys are missing
evals/
  dataset.jsonl    # written by Marko (see dataset.example.jsonl for the schema)
  upload_dataset.py
  evaluators.py    # code evaluators + LLM judge
  run_experiment.py# runs N repeats per item, writes results/<run>.json and pushes to Langfuse
  stats.py         # Wilson interval, run comparison, per-item flips
  baseline.json    # committed; only updated with Marko's explicit approval
tests/
  unit/            # deterministic, FakeLLM, no network, must run in seconds
  security/        # authz boundaries and prompt injection (FakeLLM + real retrieval)
  eval/            # marker `eval`: regression gate, needs Ollama + Langfuse
```

### Output contract
The model must return JSON matching:
```json
{"answer": "string", "citations": ["doc_id"], "refused": false}
```
Validate with pydantic. Invalid JSON is a failure that gets counted, not silently retried forever (max 1 retry, and the retry is recorded).

## Testing rules
- `pytest` with no args runs only fast deterministic tests: `addopts = -m "not eval"` in `pyproject.toml`.
- Markers: `security`, `eval`. Register them in `pyproject.toml`.
- Every test name says what risk it protects against. No assert-less tests, no tests that only exercise a mock.
- Do not mock what you are testing. Mock the LLM; do not mock retrieval in security tests.
- No `time.sleep` in tests.

## Hard rules for Claude
- **Do not write `evals/dataset.jsonl` content or `evals/human_labels.csv`.** Marko writes questions, expected facts and human labels. You may validate their format.
- **Do not change thresholds or `baseline.json` to make a gate pass.** Report the numbers and ask.
- Do not weaken or delete a failing test without saying so explicitly and explaining why.
- Never read or print `.env`. Never log secrets. Never put real personal data in fixtures.
- Keep the app small. If a feature is not needed by a test or eval, do not add it.
- Status of the environment (`.env`, models, services) must come from `uv run python scripts/check_env.py`
  run in the same turn, never from earlier reports, STATUS.md or memory.

## Commands
```powershell
uv sync
uv run pytest                    # fast suite
uv run pytest -m security
uv run pytest -m eval            # needs Ollama + Langfuse
uv run python -m evals.run_experiment --name <run-name> --repeats 5
uv run python -m evals.stats compare results/<baseline>.json results/<candidate>.json
uv run ruff check .
```

See `docs/PLAN.md` for the phased plan and acceptance criteria.
