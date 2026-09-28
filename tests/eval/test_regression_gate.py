"""Regression gate as a pytest check (Phase 5): the same verdict and report as `stats gate`.

    uv run pytest -m eval tests/eval/test_regression_gate.py --candidate results/<run>.json

The candidate is a finished results file (generate + evaluate, `candidate_repeats` repeats); this
test runs no model. It fails, printing the whole gate report, when the gate fails or refuses the
candidate. Without --candidate it fails instead of skipping: a skipped gate reads as green.
"""

from pathlib import Path

import pytest

from evals.gate import run_gate

pytestmark = pytest.mark.eval


def test_candidate_is_not_a_regression_against_the_baseline(request: pytest.FixtureRequest):
    candidate: Path | None = request.config.getoption("--candidate")
    baseline: Path = request.config.getoption("--baseline")
    if candidate is None:
        pytest.fail("no candidate: pass --candidate results/<run>.json", pytrace=False)

    passed, report = run_gate(baseline, candidate)

    if not passed:
        pytest.fail(report, pytrace=False)
    print(report)  # a passing run: shown with -s
