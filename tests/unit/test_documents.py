import re
from datetime import date
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

    ids = {d.id for d in load_documents(docs_dir)}

    assert "example-doc" not in ids
    assert ids == {"orion-budget", "orion-overview", "holiday-policy", "budget-process"}


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


def test_loader_parses_effective_date(tmp_path: Path):
    write_doc(tmp_path, "v2.md", "policy-v2", "Policy", "public", "Body.", effective="2027-01-01")

    assert load_documents(tmp_path)[0].effective == date(2027, 1, 1)


@pytest.mark.parametrize("effective", ["2026-04-31", "next year", "2026-13-01"])
def test_loader_rejects_invalid_effective_date(tmp_path: Path, effective: str):
    # A bad date would silently break "which version is current", so it must fail loudly.
    write_doc(tmp_path, "bad.md", "bad-doc", "Bad", "public", "Body.", effective=effective)

    with pytest.raises(DocumentError, match="bad.md"):
        load_documents(tmp_path)


def _versioned(tmp_path: Path, old_access: str, old_eff: str | None,
               new_access: str, new_eff: str | None, supersedes: str = "policy") -> None:
    write_doc(tmp_path, "policy.md", "policy", "Policy", old_access, "Old.", old_eff)
    eff = f"effective: {new_eff}\n" if new_eff else ""
    (tmp_path / "policy-v2.md").write_text(
        f"---\nid: policy-v2\ntitle: Policy\naccess: {new_access}\n{eff}"
        f"supersedes: {supersedes}\n---\nNew.\n", encoding="utf-8")


def test_every_versioned_corpus_doc_declares_what_it_supersedes():
    # The loader only checks docs that set `supersedes`. A `-v2` file without it would skip the
    # access and date checks entirely, so the naming convention must be backed by the field.
    docs = load_documents(PROJECT_ROOT / "data" / "docs")
    versioned = [d for d in docs if re.search(r"-v\d+$", d.id)]

    assert versioned  # precondition: the corpus has versioned docs at all
    assert [d.id for d in versioned if d.supersedes is None] == []


@pytest.mark.parametrize(
    ("old_access", "new_access"),
    [("public", "public"), ("restricted", "restricted"), ("public", "restricted")],
)
def test_superseding_doc_with_same_or_stricter_access_loads(tmp_path: Path, old_access: str,
                                                            new_access: str):
    _versioned(tmp_path, old_access, "2025-01-01", new_access, "2026-01-01")

    assert load_documents(tmp_path)[1].supersedes == "policy"


@pytest.mark.security
def test_superseding_doc_less_strict_than_original_is_rejected(tmp_path: Path):
    # A public v2 of a restricted doc would leak its content to everyone.
    _versioned(tmp_path, "restricted", "2025-01-01", "public", "2026-01-01")

    with pytest.raises(DocumentError, match="less strict"):
        load_documents(tmp_path)


@pytest.mark.parametrize(
    ("old_eff", "new_eff", "supersedes", "reason"),
    [
        pytest.param("2026-01-01", "2025-01-01", "policy", "effective date",
                     id="earlier-than-original"),
        pytest.param("2026-01-01", "2026-01-01", "policy", "effective date", id="same-date"),
        pytest.param("2025-01-01", None, "policy", "effective date", id="new-without-date"),
        pytest.param(None, "2026-01-01", "policy", "effective date", id="old-without-date"),
        pytest.param("2025-01-01", "2026-01-01", "no-such-doc", "unknown id",
                     id="unknown-target"),
    ],
)
def test_superseding_doc_that_cannot_be_ordered_is_rejected(tmp_path: Path, old_eff, new_eff,
                                                            supersedes, reason):
    # Without a strictly later date, "which version is current" has no answer. Matching the
    # reason (not just the file) keeps a parse error from passing this test.
    _versioned(tmp_path, "public", old_eff, "public", new_eff, supersedes)

    with pytest.raises(DocumentError, match=f"policy-v2.md: .*{reason}"):
        load_documents(tmp_path)


def test_loader_rejects_duplicate_ids(docs_dir: Path):
    write_doc(docs_dir, "zz-copy.md", "holiday-policy", "Copy", "public", "Other body.")

    with pytest.raises(DocumentError, match="duplicate id 'holiday-policy'"):
        load_documents(docs_dir)
