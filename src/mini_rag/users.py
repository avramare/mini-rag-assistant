"""Users and their clearances from `data/users.yaml`."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from mini_rag.documents import Access


class User(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    clearance: frozenset[Access]

    def can_read(self, access: Access) -> bool:
        return access in self.clearance


class UnknownUserError(KeyError):
    pass


def load_users(path: Path) -> dict[str, User]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        name: User(name=name, clearance=frozenset(spec["clearance"]))
        for name, spec in (raw.get("users") or {}).items()
    }


def get_user(users: dict[str, User], name: str) -> User:
    try:
        return users[name]
    except KeyError:
        raise UnknownUserError(name) from None
