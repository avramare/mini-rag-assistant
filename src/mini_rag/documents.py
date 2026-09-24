"""Load Markdown documents with YAML frontmatter from `data/docs/`."""

from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Access(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


# Higher = stricter. A newer version of a doc must never be readable by more people.
ACCESS_RANK = {Access.PUBLIC: 0, Access.RESTRICTED: 1}


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    access: Access
    body: str = Field(min_length=1)
    # Optional. Several versions of one policy can coexist; the date tells which is in force.
    effective: date | None = None
    # Id of the older version this doc replaces. Checked by `check_versions` on load.
    supersedes: str | None = None


class DocumentError(ValueError):
    """A document file is malformed. Loading fails loudly instead of skipping it."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path.name}: {reason}")
        self.path = path
        self.reason = reason


def parse_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if not text.startswith("---") or len(parts) < 3:
        raise DocumentError(path, "missing YAML frontmatter delimited by '---'")
    try:
        meta = yaml.safe_load(parts[1])
    # ValueError: PyYAML turns `2026-04-31` into a date object itself and raises plain ValueError.
    except (yaml.YAMLError, ValueError) as exc:
        raise DocumentError(path, f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise DocumentError(path, "frontmatter must be a mapping")
    try:
        # Enum match is exact: "Public" or "secret" is rejected, never defaulted.
        return Document.model_validate({**meta, "body": parts[2].strip()})
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "?" for e in exc.errors())
        raise DocumentError(path, f"invalid fields: {fields}") from exc


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every `*.md` in `docs_dir` except files starting with `_`, sorted by id."""
    docs: dict[str, Document] = {}
    paths: dict[str, Path] = {}
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        doc = parse_document(path)
        if doc.id in docs:
            raise DocumentError(path, f"duplicate id '{doc.id}'")
        docs[doc.id] = doc
        paths[doc.id] = path
    check_versions(docs, paths)
    return [docs[k] for k in sorted(docs)]


def check_versions(docs: dict[str, Document], paths: dict[str, Path]) -> None:
    """A doc that supersedes another must exist alongside it, be at least as strict, and be
    dated later. Otherwise a new version could leak an old restricted doc or never be current."""
    for doc in docs.values():
        if doc.supersedes is None:
            continue
        path = paths[doc.id]
        old = docs.get(doc.supersedes)
        if old is None:
            raise DocumentError(path, f"supersedes unknown id '{doc.supersedes}'")
        if ACCESS_RANK[doc.access] < ACCESS_RANK[old.access]:
            raise DocumentError(path, f"access '{doc.access}' is less strict than "
                                      f"superseded '{old.id}' ({old.access})")
        if doc.effective is None or old.effective is None or doc.effective <= old.effective:
            raise DocumentError(path, f"effective date must be set and later than '{old.id}'")
