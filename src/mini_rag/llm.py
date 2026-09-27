"""LLM and embedding clients.

`LLMClient` and `Embedder` are `typing.Protocol`s: any object with matching methods satisfies them
(structural typing, like a TypeScript interface), so tests pass fakes without inheritance.
"""

import hashlib
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np


class LLMTimeoutError(TimeoutError):
    """The model did not reply in time. Backend-neutral, so callers (the judge) can catch it
    without knowing about httpx, and FakeLLM can raise it in tests."""


@dataclass(frozen=True)
class Generation:
    """One model reply plus the usage numbers we need for evals. `None` = backend did not say."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None
    # The reply hit the output-token cap (`num_predict`) and was cut off, so it is incomplete.
    truncated: bool = False


class LLMClient(Protocol):
    model: str

    # `schema`: JSON Schema the reply must follow. Passed in (not imported) so this module does not
    # depend on assistant.py, and the judge can later pass its own schema.
    def generate(self, system: str, prompt: str, schema: dict | None = None) -> Generation: ...


class Embedder(Protocol):
    embed_model: str  # part of the index cache key: another model means other vectors

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaClient:
    """Talks to the Ollama REST API. Model names come from settings, never hard-coded.

    `num_ctx` is sent on every call. Ollama does not error on a prompt longer than the context
    window: it silently drops the start of the prompt (our system rules and documents). So we
    return `prompt_eval_count` and let the caller compare it with `num_ctx`.
    """

    def __init__(self, host: str, model: str, embed_model: str, num_ctx: int,
                 read_timeout: float = 120.0, num_predict: int | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.embed_model = embed_model
        self.num_ctx = num_ctx
        self.read_timeout = read_timeout
        # Cap on output tokens per reply; None = Ollama's default (no cap we chose). A capped
        # reply ends mid-text, so callers must treat `Generation.truncated` as a failed attempt.
        self.num_predict = num_predict
        # Only the read timeout (waiting for the reply) is long; connecting to a local server is
        # fast or broken. `transport` lets unit tests plug in httpx.MockTransport: no network.
        self._http = httpx.Client(base_url=host, timeout=httpx.Timeout(10.0, read=read_timeout),
                                  transport=transport)
        self._supports_thinking: bool | None = None

    def supports_thinking(self) -> bool:
        """Ask Ollama once whether the model can think. Sending `think` to a model without that
        capability is an error, so we only send it when the model lists it."""
        if self._supports_thinking is None:
            resp = self._http.post("/api/show", json={"model": self.model})
            resp.raise_for_status()
            self._supports_thinking = "thinking" in resp.json().get("capabilities", [])
        return self._supports_thinking

    def generate(self, system: str, prompt: str, schema: dict | None = None) -> Generation:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Structured outputs: with a JSON Schema, Ollama constrains decoding to that shape.
            "format": schema or "json",
            "options": {"num_ctx": self.num_ctx,
                        **({"num_predict": self.num_predict} if self.num_predict else {})},
            "stream": False,
        }
        if self.supports_thinking():
            body["think"] = False  # thinking tokens cost time and are not part of the contract
        try:
            resp = self._http.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.model} did not reply within {self.read_timeout:g}s") from exc
        resp.raise_for_status()
        data = resp.json()
        total_ns = data.get("total_duration")
        return Generation(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            duration_ms=total_ns / 1e6 if total_ns is not None else None,
            truncated=data.get("done_reason") == "length",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
        resp.raise_for_status()
        return np.asarray(resp.json()["embeddings"], dtype=np.float64)

    def model_digests(self, models: list[str]) -> dict[str, str | None]:
        """Content digest of each pulled model. A tag like "qwen3:4b" can point to a new build
        after `ollama pull`; the digest identifies the exact weights a run used."""
        resp = self._http.get("/api/tags")
        resp.raise_for_status()
        pulled = {m["name"]: m.get("digest") for m in resp.json().get("models", [])}
        # `ollama list` shows "llama3.2:latest" for a model configured as "llama3.2"
        return {m: pulled.get(m) or pulled.get(f"{m}:latest") for m in models}

    def loaded_models(self) -> list[str]:
        """Models currently in memory (`ollama ps`)."""
        resp = self._http.get("/api/ps")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def unload(self, model: str) -> None:
        """Free the model's memory now instead of after Ollama's idle timeout."""
        resp = self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
        resp.raise_for_status()


@dataclass
class FakeLLM:
    """Returns scripted responses in order and records every call. Deterministic, no network.

    A response is a plain string, a `Generation` when a test needs token counts, or an exception
    to raise (e.g. `LLMTimeoutError`) to simulate a failing backend.
    """

    responses: Iterable[str | Generation | Exception]
    model: str = "fake-llm"
    calls: list[tuple[str, str]] = field(default_factory=list)
    schemas: list[dict | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue = deque(self.responses)

    def generate(self, system: str, prompt: str, schema: dict | None = None) -> Generation:
        self.calls.append((system, prompt))
        self.schemas.append(schema)
        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted responses")
        response = self._queue.popleft()
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, Generation) else Generation(response)


class FakeEmbedder:
    """Hashed bag-of-words vectors. Same text -> same vector on every machine and run.

    Uses md5 (not Python's `hash()`, which is randomized per process) to pick a bucket per token.
    """

    def __init__(self, dim: int = 256, embed_model: str = "fake-embedder") -> None:
        self.dim = dim
        self.embed_model = embed_model

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return out
