from pathlib import Path

import pytest

from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import Access, DocumentError, load_documents
from tests.helpers import write_doc


def test_project_corpus_loads_with_both_access_levels():
    # Catches a malformed doc in data/docs/ before an eval run does.
    docs = load_documents(PROJECT_ROOT / "data" / "docs")

    assert {d.access for d in docs} == {Access.PUBLIC, Access.RESTRICTED}


def test_loader_parses_valid_frontmatter(docs_dir: Path):
    docs = {d.id: d for d in load_documents(docs_dir)}

    assert set(docs) == {"orion-budget", "orion-overview", "holiday-policy", "budget-process"}
    assert docs["orion-budget"].access is Access.RESTRICTED
    assert docs["holiday-policy"].title == "Holiday policy"
    assert docs["holiday-policy"].body == "Employees get 27 days of paid holiday per year."


def test_loader_skips_underscore_files(docs_dir: Path):
    write_doc(docs_dir, "_TEMPLATE.md", "example-doc", "Example", "public", "Template body.")

    assert "example-doc" not in {d.id for d in load_documents(docs_dir)}


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("No frontmatter at all.\n", id="no-delimiters"),
        pytest.param("---\ntitle: T\naccess: public\n---\nBody.\n", id="missing-id"),
        pytest.param("---\nid: a-doc\ntitle: T\n---\nBody.\n", id="missing-access"),
        pytest.param("---\nid: a-doc\ntitle: T\naccess: public\n---\n\n", id="empty-body"),
        pytest.param("---\n[not, a, mapping]\n---\nBody.\n", id="frontmatter-not-mapping"),
        pytest.param("---\nid: A Doc\ntitle: T\naccess: public\n---\nBody.\n", id="id-not-kebab"),
    ],
)
def test_loader_rejects_malformed_document(tmp_path: Path, content: str):
    (tmp_path / "bad.md").write_text(content, encoding="utf-8")

    with pytest.raises(DocumentError, match="bad.md"):
        load_documents(tmp_path)


@pytest.mark.parametrize("access", ["secret", "Public", "RESTRICTED", "''"])
def test_loader_rejects_unknown_access_level(tmp_path: Path, access: str):
    write_doc(tmp_path, "typo.md", "typo-doc", "Typo", access, "Body.")

    with pytest.raises(DocumentError, match="access"):
        load_documents(tmp_path)


def test_loader_rejects_duplicate_ids(docs_dir: Path):
    write_doc(docs_dir, "zz-copy.md", "holiday-policy", "Copy", "public", "Other body.")

    with pytest.raises(DocumentError, match="duplicate id 'holiday-policy'"):
        load_documents(docs_dir)
