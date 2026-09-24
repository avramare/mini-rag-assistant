# Changelog

## 2026-09-24 – Phase 0 + 1
- uv project (Python 3.12), pytest + ruff config, markers `security`/`eval`, `eval` excluded by default.
- `scripts/check_env.py`: .env, model names, Ollama reachability, pulled models, Langfuse keys.
- Core: document loader, users, access-filtered cosine retrieval, LLM/Embedder protocols, OllamaClient, FakeLLM, FakeEmbedder, assistant with JSON contract and one recorded retry.
- 20 draft corpus docs (fictional "Kestrel Logistics").
- 44 deterministic unit tests, < 1 s, no network.
