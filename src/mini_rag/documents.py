"""Load Markdown documents with YAML frontmatter from `data/docs/`."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Access(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    access: Access
    body: str = Field(min_length=1)


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
    except yaml.YAMLError as exc:
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
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        doc = parse_document(path)
        if doc.id in docs:
            raise DocumentError(path, f"duplicate id '{doc.id}'")
        docs[doc.id] = doc
    return [docs[k] for k in sorted(docs)]
