from pathlib import Path

import numpy as np
import pytest

from mini_rag.documents import Access, Document, load_documents
from mini_rag.llm import FakeEmbedder
from mini_rag.retrieval import Retriever
from mini_rag.users import User
from tests.helpers import write_doc

ORION_Q = "What is the Orion project budget?"
PUBLIC_IDS = {"orion-overview", "holiday-policy", "budget-process"}


@pytest.mark.security
def test_top1_for_restricted_best_match_is_one_public_doc(retriever: Retriever, analyst: User,
                                                          lead: User):
    # Precondition: for a cleared user the restricted doc is the best match.
    assert retriever.search(ORION_Q, lead, k=1)[0].doc.id == "orion-budget"

    hits = retriever.search(ORION_Q, analyst, k=1)

    # A filter applied after top-k would drop the restricted hit and return [], which a bare
    # `all(public)` check passes vacuously -- so the count matters as much as the access.
    assert len(hits) == 1
    assert hits[0].doc.access is Access.PUBLIC


@pytest.mark.security
def test_k_above_corpus_returns_exactly_the_public_docs(retriever: Retriever, analyst: User):
    # Catches both a leak (extra doc) and an over-eager filter (missing public doc).
    hits = retriever.search(ORION_Q, analyst, k=10)

    assert {h.doc.id for h in hits} == PUBLIC_IDS


def test_retrieval_ranks_most_similar_first(retriever: Retriever, analyst: User):
    hits = retriever.search("How many days of paid holiday do employees get?", analyst, k=3)

    assert hits[0].doc.id == "holiday-policy"
    assert hits[0].score > hits[1].score


def test_retrieval_breaks_score_ties_by_doc_id():
    # Identical text -> identical embeddings -> identical scores. Insert in reverse id order.
    docs = [Document(id=i, title="Same", access=Access.PUBLIC, body="Same body.")
            for i in ("doc-c", "doc-a", "doc-b")]
    user = User(name="u", clearance=frozenset({Access.PUBLIC}))

    hits = Retriever(docs, FakeEmbedder()).search("same body", user, k=3)

    assert [h.doc.id for h in hits] == ["doc-a", "doc-b", "doc-c"]


@pytest.mark.security
def test_user_without_clearance_gets_no_documents(retriever: Retriever):
    nobody = User(name="nobody", clearance=frozenset())

    assert retriever.search("Orion budget", nobody, k=5) == []


class CountingEmbedder(FakeEmbedder):
    """Real FakeEmbedder vectors; counts how many texts it had to embed."""

    def __init__(self, embed_model: str = "fake-embedder") -> None:
        super().__init__(embed_model=embed_model)
        self.embedded = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded += len(texts)
        return super().embed(texts)


def _build(docs_dir: Path, cache_dir: Path, embed_model: str = "fake-embedder") -> int:
    embedder = CountingEmbedder(embed_model)
    Retriever(load_documents(docs_dir), embedder, cache_dir=cache_dir)
    return embedder.embedded


def test_index_cache_is_reused_when_docs_and_model_unchanged(docs_dir: Path, tmp_path: Path):
    cache = tmp_path / "cache"
    assert _build(docs_dir, cache) == 4  # cold: all docs embedded

    assert _build(docs_dir, cache) == 0


def test_index_cache_invalidated_when_a_doc_changes(docs_dir: Path, tmp_path: Path):
    # Risk: a stale index ranks by the old text, so an edited doc is found by words it no
    # longer contains.
    cache = tmp_path / "cache"
    _build(docs_dir, cache)
    write_doc(docs_dir, "holiday-policy.md", "holiday-policy", "Holiday policy", "public",
              "Employees get 30 days of paid holiday per year.", "2026-06-01")

    assert _build(docs_dir, cache) == 4


def test_index_cache_invalidated_when_embedding_model_changes(docs_dir: Path, tmp_path: Path):
    # Risk: vectors from another model live in another space; mixing them breaks ranking.
    cache = tmp_path / "cache"
    _build(docs_dir, cache, embed_model="model-a")

    assert _build(docs_dir, cache, embed_model="model-b") == 4


def test_cached_index_ranks_same_as_fresh(docs_dir: Path, tmp_path: Path, analyst: User):
    docs = load_documents(docs_dir)
    cache = tmp_path / "cache"
    Retriever(docs, FakeEmbedder(), cache_dir=cache)  # warm the cache

    cached = Retriever(docs, FakeEmbedder(), cache_dir=cache).search(ORION_Q, analyst, k=3)
    fresh = Retriever(docs, FakeEmbedder()).search(ORION_Q, analyst, k=3)

    assert [(h.doc.id, round(h.score, 9)) for h in cached] == \
           [(h.doc.id, round(h.score, 9)) for h in fresh]


@pytest.mark.security
def test_doc_made_restricted_is_hidden_even_with_warm_cache(docs_dir: Path, tmp_path: Path,
                                                            analyst: User):
    # Same text -> same cache key, so the cached vectors are reused. Access must still come from
    # the freshly loaded docs, never from anything stored next to the vectors.
    cache = tmp_path / "cache"
    _build(docs_dir, cache)
    write_doc(docs_dir, "budget-process.md", "budget-process", "Budget process", "restricted",
              "Every project budget is reviewed each quarter by finance.")
    embedder = CountingEmbedder()
    retriever = Retriever(load_documents(docs_dir), embedder, cache_dir=cache)
    assert embedder.embedded == 0  # precondition: the warm cache was actually used

    hits = retriever.search("project budget reviewed by finance", analyst, k=10)

    assert "budget-process" not in {h.doc.id for h in hits}
