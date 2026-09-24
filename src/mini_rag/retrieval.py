"""Cosine top-k retrieval over in-memory embeddings.

Security invariant: the access filter runs BEFORE ranking. Documents the user may not read are
never scored, so they cannot reach the prompt, and the user still gets up to k permitted results.

The index holds ALL docs (the filter runs per query), so the optional on-disk cache contains
embeddings of restricted docs too. Vectors are not text, but keep `.cache/` out of git.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mini_rag.documents import Document
from mini_rag.llm import Embedder
from mini_rag.users import User


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def index_key(texts: list[str], embed_model: str) -> str:
    """Hash of everything the vectors depend on: the model and the exact texts, in order.
    Any doc edit or model change gives a new key, so a stale index is never reused."""
    return hashlib.sha256(json.dumps([embed_model, texts]).encode()).hexdigest()


class Retriever:
    def __init__(self, docs: list[Document], embedder: Embedder,
                 cache_dir: Path | None = None) -> None:
        self._docs = list(docs)
        self._embedder = embedder
        self.embed_model = embedder.embed_model
        texts = [f"{d.title}\n{d.body}" for d in self._docs]
        # Same value as the index cache key: identifies the exact corpus + embedding model.
        # Recorded with every trace and eval run so runs over different corpora are never mixed.
        self.corpus_hash = index_key(texts, self.embed_model)
        self._matrix = (_normalize(self._embed_docs(texts, cache_dir)) if texts
                        else np.zeros((0, 0)))

    def _embed_docs(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return self._embedder.embed(texts)
        path = cache_dir / f"index-{self.corpus_hash}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._embedder.embed(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vectors)
        tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written index
        return vectors

    def search(self, query: str, user: User, k: int = 3) -> list[Hit]:
        allowed = [i for i, d in enumerate(self._docs) if user.can_read(d.access)]
        if not allowed or k <= 0:
            return []
        q = _normalize(self._embedder.embed([query]))[0]
        scores = self._matrix[allowed] @ q
        hits = [Hit(self._docs[i], float(s)) for i, s in zip(allowed, scores, strict=True)]
        # Sort by score desc, then id asc, so ties are stable across runs.
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:k]
