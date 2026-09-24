# Changelog

## 2026-09-24 – Phase 0 + 1
- uv project (Python 3.12), pytest + ruff config, markers `security`/`eval`, `eval` excluded by default.
- `scripts/check_env.py`: .env, model names, Ollama reachability, pulled models, Langfuse keys.
- Core: document loader, users, access-filtered cosine retrieval, LLM/Embedder protocols, OllamaClient, FakeLLM, FakeEmbedder, assistant with JSON contract and one recorded retry.
- 20 draft corpus docs (fictional "Kestrel Logistics").
- 44 deterministic unit tests, < 1 s, no network.

### Review fixes (Marko's Phase 0/1 review)
- Uncited non-refused answers become refusals with `refusal_reason="uncited"`; model refusals get `"model"`.
- Versioned docs: optional `effective` date, shown in context headers; injected clock puts "Today is …" in the prompt;
  rule: use latest effective date not after today. Added v2 of holiday, remote-work (in force) and Orion budget (future trap).
- OllamaClient: explicit `num_ctx`, `think: false` when the model supports thinking, structured outputs with the
  Answer JSON schema, returns prompt/completion tokens and duration. `AnswerResult.truncation_risk` flags prompts
  above `MAX_PROMPT_CTX_SHARE` of `NUM_CTX`.
- check_env: corpus loads and worst-case prompt fits the context budget.
- Tests: applied 4 test-reviewer findings, added refusal-path, retry-context, uncited, date and truncation tests. 51 tests.

### Review round 2
- Retrieval index cached on disk, keyed by embedding model + doc texts (cold ~6 s -> cached ~8 ms).
- `supersedes` frontmatter; loader rejects a newer version that is less strict or not dated later.
- Tests: budget check on the real corpus, exact-result clearance tests, restricted-citation test,
  retry leak check, truncation across retry, cache invalidation; `security` marker on access tests. 67 tests.
