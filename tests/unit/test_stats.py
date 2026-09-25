import copy
from datetime import UTC, datetime

import pytest

from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    EvalResult,
    Evaluation,
    RetrievedDoc,
    RunResults,
)
from evals.stats import NotComparableError, category_rates, compare, overall_rate, report

CONFIG = {
    "name": "base", "as_of": "2026-09-01", "repeats": 2,
    "dataset": {"name": "t", "sha256": "d" * 64, "generation_key": "g" * 64},
    "assistant": {"gen_model": "m", "embed_model": "e", "k": 3, "num_ctx": 4096,
                  "system_prompt_sha256": "p" * 64, "corpus_sha256": "c" * 64},
}


def answer(item: str, category: str, repeat: int, *, refusal=None, error=None, attempts=1,
           truncation=False, duration=1000.0) -> AnswerRecord:
    return AnswerRecord(
        item_id=item, repeat=repeat, category=category, user="analyst", question="Q?",
        as_of="2026-09-01", answer=None, raw_outputs=[], retrieved=[RetrievedDoc(
            id="doc", access="public")], error=error, refusal_reason=refusal, attempts=attempts,
        invalid_outputs=attempts - 1, prompt_tokens=[None], durations_ms=[duration],
        truncation_risk=truncation, trace_id=None)


def make_run(passes: dict[str, list[bool]], categories: dict[str, str],
             overrides: dict | None = None) -> RunResults:
    answers, evaluations = [], []
    for item, results in passes.items():
        for repeat, passed in enumerate(results, 1):
            answers.append(answer(item, categories[item], repeat,
                                  **(overrides or {}).get((item, repeat), {})))
            failed = [] if passed else [EvalResult(name="facts_recall", applicable=True,
                                                   passed=False)]
            evaluations.append(AnswerEvaluation(item_id=item, repeat=repeat, passed=passed,
                                                results=failed or [EvalResult(
                                                    name="schema_valid", applicable=True,
                                                    passed=True)]))
    return RunResults(config=copy.deepcopy(CONFIG), answers=answers, evaluation=Evaluation(
        evaluated_at=datetime.now(UTC), dataset_sha256="d" * 64, judge=None,
        answers=evaluations))


def rows(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines()]


CATEGORIES = {"a": "factual", "b": "factual", "c": "versioning", "d": "versioning"}
PASSES = {"a": [True, False], "b": [True, True], "c": [False, False], "d": [True, True]}


def test_rates_are_means_over_items_and_n_counts_items_not_answers():
    run = make_run(PASSES, CATEGORIES)

    assert category_rates(run) == {"factual": (0.75, 2), "versioning": (0.5, 2)}
    assert overall_rate(run) == (0.625, 4)  # 4 items, although there are 8 answers
    assert "n = 4 items" in report(run)


def test_report_shows_versioning_category_and_run_health_counters():
    run = make_run(PASSES, CATEGORIES, {
        ("a", 2): {"refusal": "uncited", "duration": 2000.0},
        ("b", 1): {"attempts": 2, "duration": 3000.0},
        ("c", 1): {"error": "invalid_output", "truncation": True, "duration": 5000.0},
    })

    text = report(run)

    assert ["versioning", "2", "50.0%"] in rows(text)
    assert "refusal_reason {'-': 7, 'uncited': 1}" in text
    assert "error {'-': 7, 'invalid_output': 1}" in text
    assert "retried 1  invalid outputs 1  truncation_risk 1" in text
    # durations: five 1.0 s, 2.0, 3.0, 5.0 -> p50 1.0 s, p95 = 3.0 + 0.65 * 2.0 = 4.3 s
    # (numpy's default linear interpolation between the 7th and 8th sorted values)
    assert "p50 1.0s  p95 4.3s" in text
    assert ["c", "versioning", "0.0%", "facts_recall", "x2"] in rows(text)


def test_compare_refuses_runs_with_different_as_of():
    base = make_run(PASSES, CATEGORIES)
    candidate = make_run(PASSES, CATEGORIES)
    candidate.config["as_of"] = "2027-03-01"

    with pytest.raises(NotComparableError, match="different as_of"):
        compare(base, candidate)


def test_compare_refuses_runs_graded_against_different_facts():
    base = make_run(PASSES, CATEGORIES)
    candidate = make_run(PASSES, CATEGORIES)
    candidate.evaluation.dataset_sha256 = "e" * 64

    with pytest.raises(NotComparableError, match="different dataset facts"):
        compare(base, candidate)


def test_compare_names_the_config_change_and_per_category_delta():
    base = make_run(PASSES, CATEGORIES)
    candidate = make_run(PASSES | {"c": [True, True]}, CATEGORIES)
    candidate.config["name"] = "cand"
    candidate.config["assistant"]["k"] = 5

    text = compare(base, candidate)

    assert "k: 3 -> 5" in text
    assert ["versioning", "2", "50.0%", "100.0%", "+50.0"] in rows(text)
