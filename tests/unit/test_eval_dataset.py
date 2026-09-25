from collections import Counter
from pathlib import Path

import pytest

from evals.dataset import DatasetError, check_facts_in_corpus, load_dataset
from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import load_documents
from mini_rag.users import load_users
from tests.helpers import dataset_item, write_dataset

USERS = {"analyst", "lead"}
EVALS = PROJECT_ROOT / "evals"


def test_example_dataset_documents_the_schema_including_versioning():
    dataset = load_dataset(EVALS / "dataset.example.jsonl", USERS)

    assert "versioning" in {i.category for i in dataset.items}
    assert any(i.as_of for i in dataset.items), "example should show the per-item as_of override"


def test_draft_dataset_facts_appear_verbatim_in_the_real_corpus():
    # Protects against a fact that no answer can ever contain (typo, paraphrase, wrong number).
    users = load_users(PROJECT_ROOT / "data" / "users.yaml")
    dataset = load_dataset(EVALS / "dataset.draft.jsonl", set(users))

    check_facts_in_corpus(dataset, load_documents(PROJECT_ROOT / "data" / "docs"))
    assert Counter(i.category for i in dataset.items) == {
        "factual": 8, "multi_doc": 5, "unanswerable": 5, "restricted_probe": 6,
        "injection": 2, "versioning": 4}


def test_draft_versioning_asks_orion_on_both_sides_of_v2_effective_date():
    dataset = load_dataset(EVALS / "dataset.draft.jsonl", USERS)
    orion = [i for i in dataset.items if i.category == "versioning" and "Orion" in i.question]

    dates = sorted(dataset.as_of_for(i).isoformat() for i in orion)
    assert dates[0] < "2027-01-01" <= dates[-1]


@pytest.mark.parametrize("item, reason", [
    (dataset_item("x", "analyst", "Q?", category="factul"), "unknown category"),
    (dataset_item("x", "intern", "Q?"), "unknown user 'intern'"),
    (dataset_item("x", "analyst", "Q?", expected=["27 days"], should_refuse=True),
     "should_refuse items cannot have expected_facts"),
    (dataset_item("x", "analyst", "Is it 27 days of holiday?", expected=[["twenty", "27 days"]]),
     "question contains its own fact '27 days'"),
    (dataset_item("x", "analyst", "Q?", expected=[[]]), "non-empty"),
    (dataset_item("x", "lead", "Q?", category="versioning", expected=["4.2 million"]),
     "versioning items need forbidden_facts"),
], ids=["category-typo", "unknown-user", "refuse-with-facts", "answer-in-question", "empty-fact",
        "versioning-without-wrong-version"])
def test_invalid_item_is_rejected_with_reason(tmp_path: Path, item: dict, reason: str):
    path = write_dataset(tmp_path / "d.jsonl", [item])

    with pytest.raises(DatasetError, match=reason):
        load_dataset(path, USERS)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    path = write_dataset(tmp_path / "d.jsonl", [dataset_item("x", "analyst", "A?"),
                                                dataset_item("x", "lead", "B?")])

    with pytest.raises(DatasetError, match=r"duplicate ids \['x'\]"):
        load_dataset(path, USERS)


def test_missing_header_is_rejected(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="first line must be"):
        load_dataset(path, USERS)


def test_canonical_variant_must_be_in_corpus_other_variants_need_not(tmp_path: Path, docs_dir):
    docs = load_documents(docs_dir)
    ok = write_dataset(tmp_path / "ok.jsonl", [
        dataset_item("x", "analyst", "Q?", expected=[["27 days", "twenty-seven days"]])])
    bad = write_dataset(tmp_path / "bad.jsonl", [
        dataset_item("x", "analyst", "Q?", expected=[["twenty-seven days", "27 days"]])])

    check_facts_in_corpus(load_dataset(ok, USERS), docs)
    with pytest.raises(DatasetError, match="x: 'twenty-seven days'"):
        check_facts_in_corpus(load_dataset(bad, USERS), docs)


def test_generation_key_ignores_facts_but_tracks_questions_and_dates(tmp_path: Path):
    def key(**changes) -> str:
        item = dataset_item("x", "analyst", "How many days?", expected=["27 days"]) | changes
        return load_dataset(write_dataset(tmp_path / "d.jsonl", [item]), USERS).generation_key()

    base = key()
    assert key(expected_facts=["30 days"]) == base  # re-grading allowed after a fact fix
    assert key(question="How many holiday days?") != base
    assert key(as_of="2027-03-01") != base


def test_item_as_of_overrides_dataset_default(tmp_path: Path):
    path = write_dataset(tmp_path / "d.jsonl", [
        dataset_item("a", "lead", "Q1?"), dataset_item("b", "lead", "Q2?", as_of="2027-03-01")],
        as_of="2026-09-01")
    dataset = load_dataset(path, USERS)

    assert [dataset.as_of_for(i).isoformat() for i in dataset.items] == [
        "2026-09-01", "2027-03-01"]
