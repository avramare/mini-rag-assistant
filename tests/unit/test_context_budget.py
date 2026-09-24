"""Static check that our largest possible prompt fits the context window we send to Ollama.

The real configured values are checked by `scripts/check_env.py`; here the numbers are explicit
so the test does not depend on a local `.env`.
"""

import pytest

from mini_rag.assistant import estimate_worst_case_prompt_tokens
from mini_rag.documents import load_documents


@pytest.mark.parametrize(
    ("num_ctx", "fits"),
    [
        pytest.param(4096, True, id="default-num-ctx"),
        pytest.param(256, False, id="positive-control-too-small"),
    ],
)
def test_worst_case_prompt_fits_context_budget(docs_dir, num_ctx: int, fits: bool):
    estimate = estimate_worst_case_prompt_tokens(load_documents(docs_dir), k=3)

    assert (estimate <= 0.75 * num_ctx) is fits
