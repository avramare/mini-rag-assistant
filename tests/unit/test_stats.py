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

JUDGE = {"model": "qwen3:4b", "digest": None, "prompt_sha256": "abcdef123456" + "0" * 52,
         "pass_score": 4}

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
    return RunResults(config=copy.deepcopy(CONFIG), answers=answers, evaluations={
        "no-judge": Evaluation(evaluated_at=datetime.now(UTC), dataset_sha256="d" * 64,
                               judge=None, answers=evaluations)})


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
    assert ["c", "versioning", "0.0%", "facts_recall", "x2", "|", "retrieval", "not", "checked"
            ] in rows(text)


def test_report_splits_failing_items_into_retrieval_miss_and_generation_failure():
    run = make_run({"a": [False], "b": [False], "d": [True]}, CATEGORIES)
    recall = {"a": False, "b": True, "d": False}  # d: retrieval missed but the answer passed
    for ev in run.evaluation()[1].answers:
        ev.results.append(EvalResult(name="retrieval_recall", applicable=True,
                                     passed=recall[ev.item_id], gating=False))

    text = report(run)

    assert ["a", "factual", "0.0%", "facts_recall", "x1", "|", "retrieval", "missed", "x1"
            ] in rows(text)
    assert ["b", "factual", "0.0%", "facts_recall", "x1", "|", "retrieval", "ok"] in rows(text)
    assert not any(row[:1] == ["d"] for row in rows(text))  # a diagnostic miss is not a failure
    assert "retrieval_recall 3 2 diagnostic, not in pass rate" in " ".join(text.split())


def test_report_names_the_judge_version_and_the_other_stored_versions():
    run = make_run(PASSES, CATEGORIES)
    later = make_run({item: [True, True] for item in PASSES}, CATEGORIES).evaluation()[1]
    later.judge = JUDGE
    later.evaluated_at = later.evaluated_at.replace(year=2099)
    run.evaluations["abcdef123456"] = later

    latest, older = report(run), report(run, judge="no-")

    assert "judge qwen3:4b prompt abcdef123456" in latest
    assert "also stored: no-judge" in latest
    assert "n = 4 items): 100.0%" in latest  # the numbers come from the named version
    assert "judge none" in older and "also stored: abcdef123456" in older
    assert "n = 4 items):  62.5%" in older


def test_results_file_with_a_single_evaluation_loads_under_its_judge_version():
    # Files written before evaluations were keyed must still load, with nothing lost.
    run = make_run(PASSES, CATEGORIES)
    old_format = run.model_dump(mode="json")
    evaluation = old_format.pop("evaluations")["no-judge"]
    evaluation["judge"] = JUDGE
    del evaluation["finished_at"]  # old files were only ever saved when finished
    old_format["evaluation"] = evaluation

    loaded = RunResults.model_validate(old_format)

    assert list(loaded.evaluations) == ["abcdef123456"]
    assert loaded.evaluations["abcdef123456"].finished_at is not None
    assert overall_rate(loaded) == overall_rate(run)


def test_compare_refuses_runs_with_different_as_of():
    base = make_run(PASSES, CATEGORIES)
    candidate = make_run(PASSES, CATEGORIES)
    candidate.config["as_of"] = "2027-03-01"

    with pytest.raises(NotComparableError, match="different as_of"):
        compare(base, candidate)


def test_compare_refuses_runs_graded_against_different_facts():
    base = make_run(PASSES, CATEGORIES)
    candidate = make_run(PASSES, CATEGORIES)
    candidate.evaluation()[1].dataset_sha256 = "e" * 64

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
