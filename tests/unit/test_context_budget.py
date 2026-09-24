"""Static check that our largest possible prompt fits the context window we send to Ollama.

Uses the real corpus (the thing that grows) with the code defaults from `Settings`, never the
local `.env`. The configured values themselves are checked by `scripts/check_env.py`.
"""

from pathlib import Path

from mini_rag.assistant import CHARS_PER_TOKEN, DEFAULT_K, estimate_worst_case_prompt_tokens
from mini_rag.config import PROJECT_ROOT, Settings
from mini_rag.documents import load_documents
from tests.helpers import FIXED_TODAY, write_doc

# Code defaults, read from the model definition (not from .env), so the test tracks config.py.
DEFAULT_NUM_CTX = Settings.model_fields["num_ctx"].default
DEFAULT_SHARE = Settings.model_fields["max_prompt_ctx_share"].default


def test_real_corpus_worst_case_prompt_fits_default_num_ctx():
    docs = load_documents(PROJECT_ROOT / "data" / "docs")

    estimate = estimate_worst_case_prompt_tokens(docs, k=DEFAULT_K, today=FIXED_TODAY)

    assert estimate <= DEFAULT_SHARE * DEFAULT_NUM_CTX


def test_estimate_counts_restricted_docs(docs_dir: Path):
    # A lead's prompt can contain restricted docs, so the worst case must include them. Also
    # proves docs are counted at all: fixed prompt parts alone are far below 30k chars.
    big = "Restricted detail. " * 1600  # ~30k chars
    write_doc(docs_dir, "big-secret.md", "big-secret", "Big secret", "restricted", big)

    estimate = estimate_worst_case_prompt_tokens(load_documents(docs_dir), k=DEFAULT_K,
                                                 today=FIXED_TODAY)

    assert estimate >= len(big) / CHARS_PER_TOKEN
