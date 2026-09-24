"""Shared fixtures. pytest auto-discovers this file; its fixtures are usable in all tests below."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from mini_rag.assistant import Assistant
from mini_rag.documents import Access, load_documents
from mini_rag.llm import FakeEmbedder, FakeLLM, Generation
from mini_rag.retrieval import Retriever
from mini_rag.users import User
from tests.helpers import FIXED_TODAY, FIXTURE_DOCS, write_doc


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    for filename, fields in FIXTURE_DOCS.items():
        write_doc(tmp_path, filename, *fields)
    return tmp_path


@pytest.fixture
def analyst() -> User:
    return User(name="analyst", clearance=frozenset({Access.PUBLIC}))


@pytest.fixture
def lead() -> User:
    return User(name="lead", clearance=frozenset({Access.PUBLIC, Access.RESTRICTED}))


@pytest.fixture
def retriever(docs_dir: Path) -> Retriever:
    return Retriever(load_documents(docs_dir), FakeEmbedder())


@pytest.fixture
def make_assistant(retriever: Retriever) -> Callable[..., tuple[Assistant, FakeLLM]]:
    """Factory fixture: each test scripts its own model responses."""

    def _make(*responses: str | Generation, k: int = 3, today: date = FIXED_TODAY,
              **kwargs) -> tuple[Assistant, FakeLLM]:
        llm = FakeLLM(responses)
        return Assistant(retriever, llm, k=k, today=lambda: today, **kwargs), llm

    return _make
