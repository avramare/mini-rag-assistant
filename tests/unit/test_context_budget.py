"""Static check that our largest possible prompt fits the context window we send to Ollama.

Uses the real corpus (the thing that grows) with an explicit num_ctx, never the local `.env`.
The configured value itself is checked by `scripts/check_env.py`.
"""

from pathlib import Path

from mini_rag.assistant import estimate_worst_case_prompt_tokens
from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import load_documents
from tests.helpers import FIXED_TODAY

SHARE = 0.75


def test_real_corpus_worst_case_prompt_fits_default_num_ctx():
    docs = load_documents(PROJECT_ROOT / "data" / "docs")

    estimate = estimate_worst_case_prompt_tokens(docs, k=3, today=FIXED_TODAY)

    assert estimate <= SHARE * 4096


def test_budget_check_fails_when_num_ctx_too_small(docs_dir: Path):
    # Positive control: proves the comparison above can fail.
    estimate = estimate_worst_case_prompt_tokens(load_documents(docs_dir), k=3, today=FIXED_TODAY)

    assert estimate > SHARE * 256
