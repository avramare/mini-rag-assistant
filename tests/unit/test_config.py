import pytest

from mini_rag.config import Settings

FAKE_SECRET = "sk-lf-test-not-a-real-key"


def test_settings_do_not_expose_secrets_in_repr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", FAKE_SECRET)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", FAKE_SECRET)

    s = Settings(_env_file=None)

    assert s.langfuse_secret_key.get_secret_value() == FAKE_SECRET
    assert FAKE_SECRET not in repr(s)
    assert FAKE_SECRET not in str(s.model_dump())
