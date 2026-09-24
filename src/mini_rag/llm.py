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


@dataclass(frozen=True)
class Generation:
    """One model reply plus the usage numbers we need for evals. `None` = backend did not say."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None


class LLMClient(Protocol):
    model: str

    # `schema`: JSON Schema the reply must follow. Passed in (not imported) so this module does not
    # depend on assistant.py, and the judge can later pass its own schema.
    def generate(self, system: str, prompt: str, schema: dict | None = None) -> Generation: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaClient:
    """Talks to the Ollama REST API. Model names come from settings, never hard-coded.

    `num_ctx` is sent on every call. Ollama does not error on a prompt longer than the context
    window: it silently drops the start of the prompt (our system rules and documents). So we
    return `prompt_eval_count` and let the caller compare it with `num_ctx`.
    """

    def __init__(self, host: str, model: str, embed_model: str, num_ctx: int,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.embed_model = embed_model
        self.num_ctx = num_ctx
        self._http = httpx.Client(base_url=host, timeout=timeout)
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
            "options": {"num_ctx": self.num_ctx},
            "stream": False,
        }
        if self.supports_thinking():
            body["think"] = False  # thinking tokens cost time and are not part of the contract
        resp = self._http.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        total_ns = data.get("total_duration")
        return Generation(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            duration_ms=total_ns / 1e6 if total_ns is not None else None,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
        resp.raise_for_status()
        return np.asarray(resp.json()["embeddings"], dtype=np.float64)


@dataclass
class FakeLLM:
    """Returns scripted responses in order and records every call. Deterministic, no network.

    A response is a plain string, or a `Generation` when a test needs token counts.
    """

    responses: Iterable[str | Generation]
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
        return response if isinstance(response, Generation) else Generation(response)


class FakeEmbedder:
    """Hashed bag-of-words vectors. Same text -> same vector on every machine and run.

    Uses md5 (not Python's `hash()`, which is randomized per process) to pick a bucket per token.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return out
