"""Settings loaded from environment variables and `.env`.

Secrets are `SecretStr`, so `repr(settings)` and tracebacks show `**********` instead of the value.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_host: str = "http://localhost:11434"
    gen_model: str = ""
    embed_model: str = ""
    judge_model: str = ""
    num_ctx: int = 4096
    max_prompt_ctx_share: float = 0.75

    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"

    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    users_file: Path = PROJECT_ROOT / "data" / "users.yaml"
    cache_dir: Path = PROJECT_ROOT / ".cache"
