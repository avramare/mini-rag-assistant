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


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaClient:
    """Talks to the Ollama REST API. Model names come from settings, never hard-coded."""

    def __init__(self, host: str, model: str, embed_model: str, timeout: float = 120.0) -> None:
        self.model = model
        self.embed_model = embed_model
        self._http = httpx.Client(base_url=host, timeout=timeout)

    def generate(self, system: str, prompt: str) -> str:
        resp = self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
        resp.raise_for_status()
        return np.asarray(resp.json()["embeddings"], dtype=np.float64)


@dataclass
class FakeLLM:
    """Returns scripted responses in order and records every call. Deterministic, no network."""

    responses: Iterable[str]
    model: str = "fake-llm"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue = deque(self.responses)

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._queue.popleft()


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
