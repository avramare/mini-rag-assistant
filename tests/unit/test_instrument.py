"""Instrument vs system under test (Phase 5): the hashes the gate refuses on."""

from pathlib import Path

import pytest

from evals.instrument import (
    EVALUATOR_FILES,
    differences,
    docs_sha256,
    evaluator_sha256,
    sticky_dirty,
    users_sha256,
)
from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import Access, load_documents
from mini_rag.users import User


def test_docs_hash_changes_with_a_doc_access_level_since_that_decides_what_a_leak_is(docs_dir):
    docs = load_documents(docs_dir)
    reclassified = [d.model_copy(update={"access": Access.PUBLIC}) for d in docs]

    assert docs_sha256(docs) != docs_sha256(reclassified)
    assert docs_sha256(docs) == docs_sha256(list(reversed(docs)))  # order of loading is not content


def test_users_hash_changes_with_a_clearance(analyst: User):
    promoted = analyst.model_copy(update={"clearance": frozenset(Access)})

    assert users_sha256({"analyst": analyst}) != users_sha256({"analyst": promoted})


def copy_evaluator_files(root: Path, newline: str) -> None:
    for rel in EVALUATOR_FILES:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text.replace("\n", newline).encode())


def test_evaluator_hash_is_the_same_for_a_crlf_checkout(tmp_path):
    # A git worktree on Windows may check files out with CRLF; that is not a code change.
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    copy_evaluator_files(lf, "\n")
    copy_evaluator_files(crlf, "\r\n")

    assert evaluator_sha256(lf) == evaluator_sha256(crlf)


# Literal paths, not EVALUATOR_FILES: dropping a file from that tuple must fail this test.
@pytest.mark.parametrize("rel", ["evals/evaluators.py", "evals/judge.py", "evals/dataset.py"])
def test_evaluator_hash_changes_with_each_file_that_produces_verdicts(tmp_path, rel):
    copy_evaluator_files(tmp_path, "\n")
    before = evaluator_sha256(tmp_path)

    edited = tmp_path / rel
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")

    assert evaluator_sha256(tmp_path) != before


@pytest.mark.parametrize("old, new, expected", [
    (False, False, False),
    (False, True, True),
    (True, False, True),   # an earlier dirty session is not washed out by a clean one
    (False, None, None),   # unknown is not clean
    (None, True, True),
    (None, False, None),   # nor does a clean session make an unknown one clean
])
def test_dirty_flag_is_sticky_across_sessions(old, new, expected):
    assert sticky_dirty(old, new) is expected


def test_differences_lists_every_differing_leaf_not_only_the_first():
    old = {"judge": {"digest": "a", "prompt_sha256": "p"}, "as_of": "2026-09-01"}
    new = {"judge": {"digest": "b", "prompt_sha256": "q"}, "as_of": "2026-09-01"}

    assert differences(old, new) == ["judge.digest: 'a' -> 'b'",
                                     "judge.prompt_sha256: 'p' -> 'q'"]
