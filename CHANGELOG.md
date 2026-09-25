# Changelog

## 2026-09-25 - Phase 3: dataset, two-pass runner, evaluators
- `evals/dataset.py`: JSONL with header (`name`, fixed `as_of`, per-item override); facts as string or list of
  variants; validation (category enum incl. `versioning`, users, refusal/facts, answer-in-question, duplicates);
  canonical fact variant must appear verbatim in the corpus. `generation_key` vs file `sha256`.
- `evals/dataset.draft.jsonl`: 30 generated items (DECISIONS 27); example file gets header + versioning placeholder.
- `run_experiment generate|evaluate`: pass 1 saves answers atomically after each one (`--resume`), links masked
  traces to one Langfuse dataset run per repeat, unloads the gen model; pass 2 runs code evaluators + LLM judge on
  saved answers (re-runnable), refuses changed questions/corpus, pushes scores with deterministic ids.
- Evaluators: schema_valid, refusal_correct, facts_recall (1.0), forbidden_absent, citations_valid (independent of
  app flags), judge_faithfulness (1-5, pass >= 4, one retry, judge_error counted).
- `stats report` (item-level rates overall + per category, evaluator fails, refusal reasons, errors, retries,
  truncation, p50/p95 duration, failing items); `stats compare` refuses different `as_of` or facts.
- Run config: `today`, Ollama model digests, git commit; `AnswerResult.trace_id`; OllamaClient `model_digests`,
  `loaded_models`, `unload`.
- Security: dataset upload sends no facts; judge reasons masked for restricted context (real client, captured HTTP).
- Phase 2 review findings closed (separate commit). 8 Phase 3 mutations killed. 129 tests, ~3 s.

## 2026-09-25 - Phase 2: tracing
- `tracing.py`: one Langfuse trace per `answer()` (root `answer` span, `retrieval` retriever span, one
  `generation` per attempt with model and token usage). `NoopTracer` unless both Langfuse keys are set.
- Restricted masking: if any retrieved doc is restricted, question, prompt and outputs become
  `[masked: restricted context, N chars]`; ids, access levels, tokens, durations and flags stay (DECISIONS 18).
- CLI traces and flushes; `check_env` adds a Langfuse auth check.
- Run config on every trace (never masked): gen/embed model, k, num_ctx, SYSTEM_PROMPT sha256, corpus sha256
  (= retrieval index cache key); optional `dataset_item_id` for Phase 3.
- Security tests run a real Langfuse client into an in-memory OTel exporter and assert neither codename nor
  budget figure is exported, with a public-only positive control and a network-request guard. Five masking/config
  mutations each killed. 82 tests, < 1 s.
- CLAUDE.md: environment status must come from `check_env.py` run in the same turn.

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

### Review round 3 (last for Phase 1)
- OllamaClient contract tests via `httpx.MockTransport` (num_ctx, schema, think, usage mapping).
- Security test: doc switched to restricted is hidden even with a warm index cache.
- Fixed four tests that passed for the wrong reason (see DECISIONS 17); `FIXED_TODAY` = 2026-07-15.
- test-reviewer agent: max 5 findings with severity, only new/changed code. 73 tests.
