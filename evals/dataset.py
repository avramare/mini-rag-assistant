"""Eval dataset: JSONL with a header line, validated before anything runs.

Format (see `dataset.example.jsonl`):
    line 1:  {"dataset": {"name": "draft", "as_of": "2026-09-01"}}
    line 2+: one item per line; `as_of` per item is optional and overrides the header date.

A fact is a string or a list of accepted variants (any variant counts as a match). The first
variant is the canonical one and must appear verbatim in the corpus (`check_facts_in_corpus`).

Usage: uv run python -m evals.dataset validate <path>
"""

import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mini_rag.documents import Document

CATEGORIES = frozenset(
    {"factual", "multi_doc", "unanswerable", "restricted_probe", "injection", "versioning"}
)
# Zero tolerance in the regression gate (Phase 5); reported separately.
SAFETY_CATEGORIES = frozenset({"restricted_probe", "injection"})

Fact = str | list[str]


class DatasetError(ValueError):
    pass


def normalize(text: str) -> str:
    """Casefold and collapse whitespace. Deliberately nothing more: "4,2" vs "4.2" or "4.2 million"
    vs "4,200,000" do NOT match. List such spellings as variants of the fact instead."""
    return " ".join(text.casefold().split())


def variants(fact: Fact) -> list[str]:
    return [fact] if isinstance(fact, str) else list(fact)


class Header(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    as_of: date


class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    category: str
    user: str
    question: str = Field(min_length=1)
    expected_facts: list[Fact]
    forbidden_facts: list[Fact]
    should_refuse: bool
    as_of: date | None = None

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        # A typo would silently create a new category and hide the item from its real one.
        if value not in CATEGORIES:
            raise ValueError(f"unknown category '{value}', expected one of {sorted(CATEGORIES)}")
        return value

    @field_validator("expected_facts", "forbidden_facts")
    @classmethod
    def _non_empty_variants(cls, facts: list[Fact]) -> list[Fact]:
        for fact in facts:
            if not variants(fact) or not all(v.strip() for v in variants(fact)):
                raise ValueError("a fact must be a non-empty string or a non-empty list of them")
        return facts


class Dataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    as_of: date
    items: list[DatasetItem]
    path: Path
    sha256: str  # of the whole file: facts included, so it changes when grading changes

    @property
    def langfuse_name(self) -> str:
        return f"mini-rag-{self.name}"

    def as_of_for(self, item: DatasetItem) -> date:
        return item.as_of or self.as_of

    def generation_key(self) -> str:
        """Hash of what generation depends on (ids, questions, users, dates), NOT the facts.
        Fixing a fact typo changes `sha256` but not this key, so saved answers can be re-graded."""
        rows = [[i.id, i.question, i.user, self.as_of_for(i).isoformat()] for i in self.items]
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def load_dataset(path: Path, known_users: set[str]) -> Dataset:
    raw = path.read_bytes()
    lines = [(n, line) for n, line in enumerate(raw.decode("utf-8").splitlines(), 1)
             if line.strip()]
    if not lines:
        raise DatasetError(f"{path.name}: empty")
    try:
        first = json.loads(lines[0][1])
        header = Header.model_validate(first["dataset"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise DatasetError(f'{path.name}:1: first line must be {{"dataset": {{"name", "as_of"}}}}'
                           f" ({exc})") from None

    items: list[DatasetItem] = []
    for n, line in lines[1:]:
        try:
            item = DatasetItem.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(f"{path.name}:{n}: {exc}") from None
        _check_item(item, known_users, f"{path.name}:{n} ({item.id})")
        items.append(item)

    duplicates = [i for i, count in Counter(i.id for i in items).items() if count > 1]
    if duplicates:
        raise DatasetError(f"{path.name}: duplicate ids {duplicates}")
    return Dataset(name=header.name, as_of=header.as_of, items=items, path=path,
                   sha256=hashlib.sha256(raw).hexdigest())


def _check_item(item: DatasetItem, known_users: set[str], where: str) -> None:
    if item.user not in known_users:
        raise DatasetError(f"{where}: unknown user '{item.user}'")
    if item.should_refuse and item.expected_facts:
        raise DatasetError(f"{where}: should_refuse items cannot have expected_facts")
    question = normalize(item.question)
    for fact in item.expected_facts + item.forbidden_facts:
        for variant in variants(fact):
            if normalize(variant) in question:
                # The answer is in the question: a model can "pass" by echoing it.
                raise DatasetError(f"{where}: question contains its own fact '{variant}'")


def check_facts_in_corpus(dataset: Dataset, docs: list[Document]) -> None:
    """Every fact's canonical (first) variant must appear verbatim in some document, so a fact can
    never be unsatisfiable because of a typo or a paraphrase in the dataset."""
    corpus = [normalize(f"{d.title}\n{d.body}") for d in docs]
    missing = [
        f"{item.id}: '{variants(fact)[0]}'"
        for item in dataset.items
        for fact in item.expected_facts + item.forbidden_facts
        if not any(normalize(variants(fact)[0]) in text for text in corpus)
    ]
    if missing:
        raise DatasetError("facts not found verbatim in the corpus: " + "; ".join(missing))


def main() -> int:
    from mini_rag.config import Settings
    from mini_rag.documents import load_documents
    from mini_rag.users import load_users

    if len(sys.argv) != 3 or sys.argv[1] != "validate":
        print("usage: python -m evals.dataset validate <path>")
        return 2
    s = Settings()
    try:
        dataset = load_dataset(Path(sys.argv[2]), set(load_users(s.users_file)))
        check_facts_in_corpus(dataset, load_documents(s.docs_dir))
    except DatasetError as exc:
        print(f"[fail] {exc}")
        return 1
    counts = Counter(i.category for i in dataset.items)
    print(f"[ok]   {dataset.name}: {len(dataset.items)} items, as_of {dataset.as_of}, "
          f"sha256 {dataset.sha256[:12]}")
    for category in sorted(counts):
        print(f"       {category:<17} {counts[category]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
