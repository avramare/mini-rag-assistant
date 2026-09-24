"""Cosine top-k retrieval over in-memory embeddings.

Security invariant: the access filter runs BEFORE ranking. Documents the user may not read are
never scored, so they cannot reach the prompt, and the user still gets up to k permitted results.
"""

from dataclasses import dataclass

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


class Retriever:
    def __init__(self, docs: list[Document], embedder: Embedder) -> None:
        self._docs = list(docs)
        self._embedder = embedder
        texts = [f"{d.title}\n{d.body}" for d in self._docs]
        self._matrix = _normalize(embedder.embed(texts)) if texts else np.zeros((0, 0))

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
