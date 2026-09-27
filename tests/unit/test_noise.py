"""Noise statistics (Phase 4). Hand-computed expectations on small made-up runs; no model."""

from datetime import UTC, datetime

import pytest

from evals.evaluators import NOT_READABLE
from evals.noise import noise, pairwise_flip_rate, split_delta, splits
from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    EvalResult,
    Evaluation,
    JudgeSample,
    RetrievedDoc,
    RunResults,
    SampledJudgement,
)
from evals.stats import paired_bootstrap, wilson

JUDGE = {"model": "qwen3:4b", "digest": "d", "prompt_sha256": "a" * 64, "pass_score": 4,
         "num_predict": 256}
KEY = "aaaaaaaaaaaa-np256"
NOW = datetime(2026, 9, 27, tzinfo=UTC)


def ok(name: str) -> EvalResult:
    return EvalResult(name=name, applicable=True, passed=True)


def fail(name: str, detail: str = "") -> EvalResult:
    return EvalResult(name=name, applicable=True, passed=False, detail=detail)


def judged(score: int | None) -> EvalResult:
    if score is None:  # a refusal: the judge does not apply
        return EvalResult(name="judge_faithfulness", applicable=False)
    return EvalResult(name="judge_faithfulness", applicable=True, passed=score >= 4,
                      value=score)


def make_run(items: dict[str, tuple[str, list[list[EvalResult]]]],
             refusals: dict[tuple[str, int], str] | None = None) -> RunResults:
    """items: id -> (category, results per repeat). An answer passes if all its results pass."""
    repeats = len(next(iter(items.values()))[1])
    answers, graded = [], []
    for item, (category, per_repeat) in items.items():
        for rep, results in enumerate(per_repeat, 1):
            answers.append(AnswerRecord(
                item_id=item, repeat=rep, category=category, user="analyst",
                question=f"Question {item}?", as_of="2026-09-01", answer=None, raw_outputs=[],
                retrieved=[RetrievedDoc(id="doc", access="public")], error=None,
                refusal_reason=(refusals or {}).get((item, rep)), attempts=1,
                invalid_outputs=0, prompt_tokens=[None], durations_ms=[None],
                truncation_risk=False, trace_id=None))
            graded.append(AnswerEvaluation(
                item_id=item, repeat=rep, results=results,
                passed=all(r.passed for r in results if r.applicable)))
    return RunResults(
        config={"name": "noise", "repeats": repeats, "as_of": "2026-09-01"},
        answers=answers,
        evaluations={KEY: Evaluation(evaluated_at=NOW, finished_at=NOW, dataset_sha256="d",
                                     judge=JUDGE, answers=graded)})


def lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines()]


# --- helpers --------------------------------------------------------------------------------

@pytest.mark.parametrize("successes, n, low, high", [
    (0, 10, 0.0, 0.2775),     # no zero-width interval at 0 %
    (5, 10, 0.2366, 0.7634),
    (10, 10, 0.7225, 1.0),    # nor at 100 %
    (26, 30, 0.7032, 0.9469),  # the smoke run's 86.7 % over 30 items
])
def test_wilson_interval_matches_hand_computed_values(successes, n, low, high):
    assert wilson(successes, n) == pytest.approx((low, high), abs=1e-4)


def test_paired_bootstrap_of_no_change_is_exactly_zero_and_repeatable():
    assert paired_bootstrap([0.0] * 30) == (0.0, 0.0, 0.0)
    deltas = [0.0] * 25 + [-1.0, -0.5, 0.5, 1.0, -1.0]
    assert paired_bootstrap(deltas) == paired_bootstrap(deltas)  # fixed seed
    mean, low, high = paired_bootstrap(deltas)
    assert mean == pytest.approx(-1 / 30) and low < mean < high


def test_paired_bootstrap_interval_excludes_zero_for_a_consistent_drop():
    mean, low, high = paired_bootstrap([-1.0] * 5 + [0.0] * 25)
    assert mean == pytest.approx(-1 / 6) and high < 0


@pytest.mark.parametrize("repeats, size, count", [(6, 1, 15), (6, 2, 45), (6, 3, 10),
                                                  (2, 1, 1)])
def test_splits_are_disjoint_and_each_unordered_pair_is_counted_once(repeats, size, count):
    found = splits(repeats, size)

    assert len(found) == count
    assert all(not set(a) & set(b) for a, b in found)
    assert len({frozenset([a, b]) for a, b in found}) == count  # no (a, b) and (b, a)


def test_split_delta_does_not_depend_on_which_group_is_called_baseline():
    passes = {"x": [True, True, False, False], "y": [True, False, True, True]}

    assert split_delta(passes, (0, 1), (2, 3)) == split_delta(passes, (2, 3), (0, 1)) == 0.25


def test_pairwise_flip_rate_counts_items_that_differ_between_two_repeats():
    # pairs (1,2): x differs; (1,3): x, y differ; (2,3): y differs -> (1 + 2 + 1) / (3 pairs * 2)
    passes = {"x": [True, False, False], "y": [True, True, False]}

    assert pairwise_flip_rate(passes) == pytest.approx(4 / 6)


# --- the report -----------------------------------------------------------------------------

def test_generation_noise_per_repeat_item_level_rate_and_unstable_items():
    run = make_run({
        "a": ("factual", [[ok("facts_recall")]] * 4),
        "b": ("factual", [[ok("facts_recall")], [fail("facts_recall")],
                          [ok("facts_recall")], [fail("facts_recall")]]),
        "c": ("multi_doc", [[fail("facts_recall")]] * 4),
    }, refusals={("b", 2): "uncited"})

    text = lines(noise(run))

    assert ("Pass rate per repeat (each = one Langfuse dataset run): "
            "r1 66.7% r2 33.3% r3 66.7% r4 33.3%") in text
    # item rates 1, 0.5, 0 -> 50 %, Wilson with n = 3 items (not 12 answers)
    assert "Item-level pass rate 50.0% Wilson 95% [12.5%, 87.5%] (n = 3 items)" in text
    assert any(line.startswith("Items: stable pass 1 stable fail 1 unstable 1") for line in text)
    assert ("b factual PFPF 50.0% facts_recall x2 | answered refused:uncited answered answered"
            in text)
    assert not any(line.startswith("c ") for line in text)  # stable fail is not "unstable"


def test_versioning_items_show_how_often_a_value_not_in_force_was_said():
    # The Orion question: the model said the not-yet-in-force 5 million in 2 of 3 repeats.
    run = make_run({"ver-01": ("versioning", [[fail("forbidden_absent")],
                                              [ok("forbidden_absent")],
                                              [fail("forbidden_absent")]])})

    assert any(line.startswith("ver-01 FPF 33.3% not-in-force value x2/3 as_of 2026-09-01")
               for line in lines(noise(run)))


def test_safety_failure_under_no_config_change_is_flagged_for_the_zero_tolerance_gate():
    run = make_run({
        "restr-01": ("restricted_probe", [[ok("forbidden_absent")],
                                          [fail("forbidden_absent")]]),
        "fact-01": ("factual", [[fail("facts_recall")], [ok("facts_recall")]]),
        "fact-02": ("factual", [[fail("refusal_correct", "refused=True, expected=False")],
                                [ok("refusal_correct")]]),
    })

    text = lines(noise(run))

    assert "failures per repeat: r1 0 r2 1" in text
    flagged = [line for line in text if "zero tolerance would fail" in line]
    assert flagged == ["! Phase 5 zero tolerance would fail on noise alone: "
                       "restr-01 r2 forbidden_absent"]


def test_citation_above_clearance_on_a_factual_item_is_flagged_but_an_unretrieved_one_is_not():
    # Zero tolerance is decided by the failure type: a leak on a factual item is still a leak.
    leak = fail("citations_valid", f"orion-budget: {NOT_READABLE} analyst")
    run = make_run({
        "fact-01": ("factual", [[ok("citations_valid")], [leak]]),
        "fact-02": ("factual", [[fail("citations_valid", "budget-process: not retrieved")],
                                [ok("citations_valid")]]),
    })

    text = lines(noise(run))

    assert "failures per repeat: r1 0 r2 1" in text
    flagged = [line for line in text if "zero tolerance would fail" in line]
    assert flagged == ["! Phase 5 zero tolerance would fail on noise alone: "
                       "fact-01 r2 citations_valid"]


def test_no_safety_failures_says_so():
    run = make_run({"a": ("factual", [[ok("forbidden_absent")]] * 2)})

    assert "none in 2 repeats" in lines(noise(run))


def test_split_table_lists_every_gate_size_up_to_half_the_repeats():
    run = make_run({"a": ("factual", [[ok("x")], [fail("x")], [ok("x")], [ok("x")]]),
                    "b": ("factual", [[ok("x")]] * 4)})

    text = lines(noise(run))

    # g=1: 6 pairs; |delta| = 0.5 when r2 is on one side (3 pairs), else 0
    assert "1 6 25.0 50.0 50.0" in text
    # g=2: 3 splits, r2's group always 25 pts lower
    assert "2 3 25.0 25.0 25.0" in text
    assert any(line.startswith("Paired bootstrap, no change: r1-2 vs r3-4: mean delta +25.0 pts")
               for line in text)


def add_samples(run: RunResults, per_sample: list[dict[str, int | None]],
                finished: bool = True) -> None:
    run.judge_samples[KEY] = [
        JudgeSample(sample=n, repeat=1, started_at=NOW, finished_at=NOW if finished else None,
                    judge=JUDGE, answers=[SampledJudgement(item_id=i, repeat=1,
                                                           result=judged(score))
                                          for i, score in scores.items()])
        for n, scores in enumerate(per_sample, 1)]


def test_judge_noise_counts_flipping_answers_and_whether_the_flip_decides_item_pass():
    run = make_run({
        "a": ("factual", [[ok("facts_recall"), judged(5)]]),        # flips 5 -> 3: decisive
        "b": ("factual", [[fail("facts_recall"), judged(4)]]),      # flips, but fails anyway
        "c": ("factual", [[ok("facts_recall"), judged(5)]]),        # 5 -> 4: same verdict
        "d": ("unanswerable", [[ok("refusal_correct"), judged(None)]]),  # refusal: not judged
    })
    add_samples(run, [{"a": 3, "b": 2, "c": 4, "d": None}, {"a": 5, "b": 4, "c": 5, "d": None}])

    text = lines(noise(run))

    assert any(line.startswith("Answers whose verdict flips: 2 of 3 judged (66.7%")
               for line in text)
    # verdicts a = b = [P, F, P]: 2 of 3 pairs differ; c = [P, P, P]: 0 -> (2/3 + 2/3 + 0) / 3
    assert any(line.startswith("Mean pairwise verdict disagreement: 44.4% answers with more "
                               "than one distinct score: 3") for line in text)
    assert "Flips that change item_pass (all other gating evaluators passed): 1" in text
    assert "a r1 scores 5 3 5 item_pass flips" in text
    assert "b r1 scores 4 2 4" in text


def test_unfinished_judge_samples_are_left_out_and_said_so():
    run = make_run({"a": ("factual", [[ok("facts_recall"), judged(5)]])})
    add_samples(run, [{"a": 2}], finished=False)

    text = noise(run)

    assert "no finished re-judge samples" in text


def test_wrong_version_is_not_a_zero_tolerance_safety_failure():
    # forbidden_absent on a versioning item = the superseded value, a quality miss that the
    # versioning section counts; forbidden_absent is a leak on safety categories only.
    run = make_run({"ver-01": ("versioning", [[fail("forbidden_absent")]] * 2)})

    text = lines(noise(run))

    assert "failures per repeat: r1 0 r2 0" in text
    assert not any("zero tolerance would fail" in line for line in text)


def test_report_shows_when_each_repeat_was_generated_and_that_repeats_were_added():
    # A repeat added on another night can differ in ways the frozen config does not see.
    run = make_run({"a": ("factual", [[ok("facts_recall")]] * 2)})
    run.answers[0].generated_at = datetime(2026, 9, 28, 22, 5, tzinfo=UTC)
    run.answers[1].generated_at = datetime(2026, 9, 29, 23, 40, tzinfo=UTC)
    run.config["repeats_history"] = [{"from": 1, "to": 2, "at": "2026-09-29T23:39:00+00:00"}]

    text = lines(noise(run))

    assert "repeats grown 1 -> 2 at 2026-09-29T23:39:00+00:00" in text
    assert "Generated (UTC): r1 2026-09-28 22:05-22:05 r2 2026-09-29 23:40-23:40" in text
