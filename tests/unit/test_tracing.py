import pytest
from pydantic import SecretStr

from mini_rag.config import Settings
from mini_rag.documents import Access, Document
from mini_rag.retrieval import Hit
from mini_rag.tracing import NoopTracer, is_restricted, make_tracer, mask_text


def hit(doc_id: str, access: Access) -> Hit:
    return Hit(Document(id=doc_id, title=doc_id, access=access, body="text"), 0.5)


@pytest.mark.parametrize("public, secret", [("", ""), ("pk-lf-x", ""), ("", "sk-lf-x")])
def test_tracer_is_noop_without_both_langfuse_keys(public: str, secret: str):
    # A half-configured .env must not create a client that fails on every export.
    settings = Settings(langfuse_public_key=SecretStr(public),
                        langfuse_secret_key=SecretStr(secret))

    assert isinstance(make_tracer(settings), NoopTracer)


def test_mask_text_keeps_length_only_when_masked():
    secret = "Codename BLUEHERON, 4.2 million euros."

    assert mask_text(secret, masked=True) == f"[masked: restricted context, {len(secret)} chars]"
    assert mask_text(secret, masked=False) == secret


def test_one_restricted_hit_among_public_ones_masks_the_call():
    assert is_restricted([hit("a", Access.PUBLIC), hit("b", Access.RESTRICTED)])
    assert not is_restricted([hit("a", Access.PUBLIC), hit("c", Access.PUBLIC)])
    assert not is_restricted([])
