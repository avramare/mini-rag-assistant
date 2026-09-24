from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import Access
from mini_rag.users import UnknownUserError, get_user, load_users


def test_project_users_have_expected_clearances():
    users = load_users(PROJECT_ROOT / "data" / "users.yaml")

    assert users["analyst"].clearance == {Access.PUBLIC}
    assert users["lead"].clearance == {Access.PUBLIC, Access.RESTRICTED}


def test_unknown_user_raises_instead_of_getting_default_clearance(tmp_path: Path):
    path = tmp_path / "users.yaml"
    path.write_text("users:\n  analyst:\n    clearance: [public]\n", encoding="utf-8")

    with pytest.raises(UnknownUserError):
        get_user(load_users(path), "intruder")


def test_unknown_clearance_level_is_rejected(tmp_path: Path):
    path = tmp_path / "users.yaml"
    path.write_text("users:\n  bob:\n    clearance: [public, topsecret]\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_users(path)
