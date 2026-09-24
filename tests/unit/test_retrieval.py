import pytest

from mini_rag.documents import Access, Document
from mini_rag.llm import FakeEmbedder
from mini_rag.retrieval import Retriever
from mini_rag.users import User


@pytest.mark.parametrize(
    "k",
    [
        # k=1 with a query the restricted doc ranks first for: catches a leak through ranking.
        pytest.param(1, id="top1-restricted-best-match"),
        # k above the corpus size: catches returning everything unfiltered.
        pytest.param(10, id="k-exceeds-corpus"),
    ],
)
def test_retrieval_never_returns_doc_above_clearance(retriever: Retriever, analyst: User, k: int):
    hits = retriever.search("What is the Orion project budget?", analyst, k=k)

    assert all(h.doc.access is Access.PUBLIC for h in hits)


def test_retrieval_filters_before_ranking(retriever: Retriever, analyst: User, lead: User):
    query = "Orion project budget million euros"
    # Precondition: for a cleared user the restricted doc is the best match, so a
    # filter applied after top-k would leave the analyst with fewer than k results.
    assert retriever.search(query, lead, k=1)[0].doc.id == "orion-budget"

    hits = retriever.search(query, analyst, k=3)

    assert [h.doc.access for h in hits] == [Access.PUBLIC] * 3


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


def test_user_without_clearance_gets_no_documents(retriever: Retriever):
    nobody = User(name="nobody", clearance=frozenset())

    assert retriever.search("Orion budget", nobody, k=5) == []
