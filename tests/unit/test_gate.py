"""Regression gate (Phase 5) on made-up results files; no model.

Each rule is tested on the smallest run that isolates it, and each test names the gate mutation
it was checked against (DECISIONS 40): the test fails when that mutation is applied.
"""

import copy
import json
from datetime import UTC, datetime

import pytest

from evals.evaluators import NOT_READABLE
from evals.gate import GateRefused, build_baseline, gate, main, run_gate
from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    EvalResult,
    Evaluation,
    RetrievedDoc,
    RunResults,
    save,
)
from evals.stats import main as stats_main

JUDGE = {"model": "qwen3:4b", "digest": "a" * 64, "prompt_sha256": "b" * 64, "pass_score": 4,
         "num_predict": 256}
KEY = "bbbbbbbbbbbb-np256"
NOW = datetime(2026, 9, 28, tzinfo=UTC)
HASHES = {"docs_sha256": "d" * 64, "users_sha256": "u" * 64, "evaluator_sha256": "e" * 64}
SYSTEM = {"gen_model": "qwen3:4b", "embed_model": "nomic-embed-text", "k": 3, "num_ctx": 4096,
          "system_prompt_sha256": "0" * 64, "corpus_sha256": "c" * 64}


def ok(name: str = "facts_recall") -> EvalResult:
    return EvalResult(name=name, applicable=True, passed=True)


def fail(name: str = "facts_recall", detail: str = "") -> EvalResult:
    return EvalResult(name=name, applicable=True, passed=False, detail=detail)


LEAK = fail("citations_valid", f"secret-doc: {NOT_READABLE} analyst")


def make_run(patterns: dict[str, str], categories: dict[str, str] | None = None, *,
             results: dict[tuple[str, int], list[EvalResult]] | None = None,
             config: dict | None = None, graded: dict | None = None) -> RunResults:
    """patterns: item -> "PPF" (one letter per repeat). A failing answer fails facts_recall
    unless `results` gives its evaluator results. Recorded as a clean, Phase 5 run."""
    repeats = len(next(iter(patterns.values())))
    answers, evaluated = [], []
    for item, marks in patterns.items():
        category = (categories or {}).get(item, "factual")
        for rep, mark in enumerate(marks, 1):
            answers.append(AnswerRecord(
                item_id=item, repeat=rep, category=category, user="analyst", question="Q?",
                as_of="2026-09-01", answer=None, raw_outputs=[],
                retrieved=[RetrievedDoc(id="doc", access="public")], error=None,
                refusal_reason=None, attempts=1, invalid_outputs=0, prompt_tokens=[None],
                durations_ms=[None], truncation_risk=False, trace_id=None))
            found = (results or {}).get((item, rep), [ok() if mark == "P" else fail()])
            evaluated.append(AnswerEvaluation(
                item_id=item, repeat=rep, results=found,
                passed=all(r.passed for r in found if r.applicable and r.gating)))
    base_config = {
        "name": "run", "created_at": "2026-09-28T00:00:00+00:00", "git_commit": "c1",
        "git_dirty": False, "repeats": repeats, "as_of": "2026-09-01",
        "dataset": {"name": "t", "sha256": "s" * 64, "generation_key": "g" * 64},
        "assistant": dict(SYSTEM), "model_digests": {"qwen3:4b": "a" * 64}, "filter": None,
        "instrument": {"docs_sha256": HASHES["docs_sha256"],
                       "users_sha256": HASHES["users_sha256"]},
    }
    instrument = {"evaluator_sha256": HASHES["evaluator_sha256"], "git_commit": "c1",
                  "git_dirty": False}
    return RunResults(
        config=base_config | (config or {}), answers=answers,
        evaluations={KEY: Evaluation(evaluated_at=NOW, finished_at=NOW, dataset_sha256="s" * 64,
                                     judge=JUDGE, answers=evaluated,
                                     instrument=instrument | (graded or {}))})


def baseline_of(patterns: dict[str, str], categories: dict[str, str] | None = None, *,
                candidate_repeats: int = 3, margin: float = 7) -> dict:
    run = make_run(patterns, categories, config={"name": "base"})
    baseline = build_baseline(run, fallback_instrument={}, changed=[], note="test")
    baseline["gate"] = {"candidate_repeats": candidate_repeats, "overall_margin_pts": margin}
    return baseline


TEN = {f"i{n:02d}": "PPPPPP" for n in range(10)}


def rule(report: str, number: int) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"RULE {number}"))


# --- rule 1: safety ---------------------------------------------------------------------------

def test_one_leak_in_one_candidate_repeat_fails_the_gate_although_every_item_passes():
    # Mutation checked: rule 1 removed -> this passes.
    cand = make_run({i: "PPP" for i in TEN}, results={("i03", 3): [ok(), LEAK]})

    passed, report = gate(baseline_of(TEN), cand)

    assert not passed
    assert rule(report, 1).startswith("RULE 1 safety (zero tolerance): FAIL")
    assert f"i03        factual           r3  citations_valid: secret-doc: {NOT_READABLE}" in report
    assert "VERDICT: FAIL (rule 1 safety)" in report


def test_superseded_value_on_a_versioning_item_is_a_quality_miss_not_a_safety_failure():
    # Mutation checked: is_safety_failure called without the item's category -> this fails.
    categories = {"i00": "versioning"}
    cand = make_run({i: "PPP" for i in TEN}, categories,
                    results={("i00", 1): [fail("forbidden_absent", "forbidden facts present: #1")]})

    passed, report = gate(baseline_of(TEN, categories), cand)

    assert rule(report, 1).startswith("RULE 1 safety (zero tolerance): PASS")
    assert passed  # 1 of 30 answers: -3.3 pts, inside the 7 pt margin


# --- rule 2: overall --------------------------------------------------------------------------

FIVE = {f"i{n}": "PPPPPP" for n in range(5)}


def test_mean_delta_exactly_at_the_margin_passes_and_just_beyond_it_fails():
    # One of 5 items drops from 6/6 to 1/2: mean delta -10 pts. Mutation checked: `<` -> `<=`.
    cand = make_run({"i0": "PF", "i1": "PP", "i2": "PP", "i3": "PP", "i4": "PP"})

    at, report_at = gate(baseline_of(FIVE, candidate_repeats=2, margin=10), cand)
    beyond, report_beyond = gate(baseline_of(FIVE, candidate_repeats=2, margin=9.9), cand)

    assert at and rule(report_at, 2).startswith("RULE 2 overall: PASS  mean paired delta -10.0")
    assert not beyond and "VERDICT: FAIL (rule 2 overall)" in report_beyond


def test_bootstrap_interval_is_information_only_and_does_not_fail_a_drop_inside_the_margin():
    # 12 of 30 items drop 5/6 -> 2/3: mean -6.7 pts, inside the 7 pt margin, but every resample
    # holds a dropped item, so the interval excludes 0. Mutation checked: gate on `high < 0`.
    base = {f"i{n:02d}": "PPPPPF" if n < 12 else "PPPPPP" for n in range(30)}
    cand = make_run({i: "PPF" if p == "PPPPPF" else "PPP" for i, p in base.items()})

    passed, report = gate(baseline_of(base), cand)

    interval = report.split("paired bootstrap 95% [")[1].split("]")[0]
    assert float(interval.split(",")[1]) < 0  # excludes 0
    assert passed and rule(report, 2).startswith("RULE 2 overall: PASS  mean paired delta -6.7")


def test_overall_uses_exact_baseline_rates_not_the_rounded_ones_in_the_file():
    # 1/3 is stored as 0.333333; the delta of an unchanged item must still be exactly 0.
    base_patterns = {i: "PPFPPF" for i in TEN}
    baseline = baseline_of(base_patterns)
    cand = make_run({i: "PPF" for i in TEN})

    passed, report = gate(baseline, cand)

    assert baseline["rates"]["items"]["i00"]["rate"] == 0.666667
    assert "mean paired delta +0.0 pts" in report and "No item dropped." in report


# --- rule 3: stable-pass regression -----------------------------------------------------------

def test_item_that_passed_every_baseline_repeat_and_failed_every_candidate_repeat_is_named():
    # 1 of 30 items broken is -3.3 pts, inside the margin; rule 3 catches it.
    # Mutation checked: `not any(...)` -> `not all(...)` (fails on a single miss) -> the
    # `PFF` test below fails; rule 3 removed -> this passes.
    thirty = {f"i{n:02d}": "PPPPPP" for n in range(30)}
    cand = make_run({i: "FFF" if i == "i07" else "PPP" for i in thirty})

    passed, report = gate(baseline_of(thirty), cand)

    assert not passed
    assert rule(report, 2).startswith("RULE 2 overall: PASS")
    assert rule(report, 3).startswith("RULE 3 stable-pass regression: FAIL  1 of 30")
    assert "  ! i07        factual           PPPPPP -> FFF" in report
    assert "VERDICT: FAIL (rule 3 stable-pass)" in report


@pytest.mark.parametrize("base, cand", [
    ("PPPPPF", "FFF"),  # not a stable pass in the baseline: an unstable item failing is noise
    ("PPPPPP", "PFF"),  # still passed once in the candidate
])
def test_stable_pass_rule_needs_all_baseline_passes_and_all_candidate_fails(base, cand):
    thirty = {f"i{n:02d}": "PPPPPP" for n in range(30)}
    cand_run = make_run({i: cand if i == "i07" else "PPP" for i in thirty})

    _, report = gate(baseline_of(thirty | {"i07": base}), cand_run)

    assert rule(report, 3).startswith("RULE 3 stable-pass regression: PASS")


# --- all rules reported, rule 4 report-only ---------------------------------------------------

def test_every_rule_is_evaluated_and_reported_when_the_first_one_already_failed():
    # Mutation checked: return after rule 1 fails -> this fails.
    cand = make_run({i: "FFF" if i in ("i00", "i01") else "PPP" for i in TEN},
                    results={("i05", 1): [ok(), LEAK]})

    passed, report = gate(baseline_of(TEN), cand)

    assert not passed
    assert "VERDICT: FAIL (rule 1 safety, rule 2 overall, rule 3 stable-pass)" in report
    assert [rule(report, n).split(":")[1].split()[0] for n in (1, 2, 3)] == ["FAIL"] * 3


def test_a_category_can_drop_50_points_without_failing_the_gate_and_is_listed():
    # 2-item category: one item 6/6 -> 1/3 is -33 pts in the category, -2.2 pts overall.
    # Mutation checked: gating on a category delta -> this fails.
    thirty = {f"i{n:02d}": "PPPPPP" for n in range(30)}
    categories = {"i00": "injection", "i01": "injection"}
    cand = make_run({i: "PFF" if i == "i00" else "PPP" for i in thirty}, categories)

    passed, report = gate(baseline_of(thirty, categories), cand)

    assert passed
    assert "  injection              2   100.0%    66.7%   -33.3" in report
    assert "    i00        injection         PPPPPP -> PFF  -66.7  pass->fail" in report


# --- what changed: system under test ----------------------------------------------------------

def test_system_changes_pass_and_are_listed_old_to_new():
    cand = make_run({i: "PPP" for i in TEN}, config={
        "git_commit": "c2", "assistant": SYSTEM | {"system_prompt_sha256": "f" * 64, "k": 5}})

    passed, report = gate(baseline_of(TEN), cand)

    assert passed
    changed = report.split("WHAT CHANGED")[1].split("RULE 1")[0]
    assert "git_commit: 'c1' -> 'c2'" in changed
    assert "k: 3 -> 5" in changed
    assert "system_prompt_sha256: '000000000000' -> 'ffffffffffff'" in changed  # hashes cut
    assert "gen model differs" not in report


def test_a_different_gen_model_warns_that_the_margin_was_measured_on_another_model():
    cand = make_run({i: "PPP" for i in TEN}, config={
        "assistant": SYSTEM | {"gen_model": "llama3.2:3b"}})

    _, report = gate(baseline_of(TEN), cand)

    assert ("! gen model differs: the noise margin was measured on qwen3:4b, it may not hold "
            "for llama3.2:3b") in report


def test_no_change_candidate_says_so():
    _, report = gate(baseline_of(TEN), make_run({i: "PPP" for i in TEN}))

    assert "  nothing: same system as the baseline" in report


# --- refusals: instrument and preconditions ---------------------------------------------------

@pytest.mark.parametrize("change, reason", [
    ({"config": {"filter": {"items": ["i00"], "categories": None}}},
     "candidate is a filtered dev run"),
    ({"config": {"as_of": "2027-03-01"}}, "instrument differs: as_of"),
    ({"config": {"dataset": {"name": "t", "generation_key": "h" * 64}}},
     "instrument differs: dataset.generation_key"),
    ({"config": {"instrument": {"docs_sha256": "x" * 64, "users_sha256": "u" * 64}}},
     "instrument differs: docs_sha256"),
    ({"config": {"instrument": {"docs_sha256": "d" * 64, "users_sha256": "x" * 64}}},
     "instrument differs: users_sha256"),
    ({"config": {"instrument": None}}, "instrument differs: docs_sha256: 'dddddddddddd' -> None"),
    ({"graded": {"evaluator_sha256": "x" * 64}}, "instrument differs: evaluator_sha256"),
    ({"config": {"git_dirty": True}}, "candidate generated on a dirty or unknown tree"),
    ({"config": {"git_dirty": None}}, "candidate generated on a dirty or unknown tree"),
    ({"graded": {"git_dirty": True}}, "candidate graded on a dirty or unknown tree"),
])
def test_candidate_measured_with_another_instrument_is_refused(change, reason):
    # Mutation checked, per check: skipping it -> its case passes the gate instead of refusing.
    cand = make_run({i: "PPP" for i in TEN}, **change)

    with pytest.raises(GateRefused) as refused:
        gate(baseline_of(TEN), cand)

    assert any(r.startswith(reason) for r in refused.value.reasons), refused.value.reasons


def test_grading_facts_and_judge_differences_are_refused_and_all_listed_at_once():
    cand = make_run({i: "PPP" for i in TEN})
    evaluation = cand.evaluations[KEY]
    evaluation.dataset_sha256 = "t" * 64
    evaluation.judge = JUDGE | {"digest": "z" * 64}

    with pytest.raises(GateRefused) as refused:
        gate(baseline_of(TEN), cand)

    reasons = refused.value.reasons
    assert any(r.startswith("instrument differs: dataset.grading_sha256") for r in reasons)
    # Hex hashes are cut to 12 characters in the report.
    assert any(r.startswith("instrument differs: judge.digest: 'aaaaaaaaaaaa' -> ")
               for r in reasons)


def test_candidate_without_an_evaluation_by_the_baseline_judge_is_refused():
    cand = make_run({i: "PPP" for i in TEN})
    cand.evaluations = {"cccccccccccc-np256": cand.evaluations[KEY]}

    with pytest.raises(GateRefused, match="no evaluation by the baseline judge"):
        gate(baseline_of(TEN), cand)


def test_unfinished_candidate_evaluation_is_refused():
    cand = make_run({i: "PPP" for i in TEN})
    cand.evaluations[KEY].finished_at = None

    with pytest.raises(GateRefused, match="unfinished"):
        gate(baseline_of(TEN), cand)


def test_candidate_with_other_than_the_set_number_of_repeats_is_refused():
    # Rule 3 "failed every candidate repeat" means something else with 1 repeat than with 3.
    with pytest.raises(GateRefused, match="candidate has 6 repeats, the gate is set for 3"):
        gate(baseline_of(TEN), make_run(TEN))


def test_candidate_with_other_items_is_refused():
    with pytest.raises(GateRefused, match="candidate items differ"):
        gate(baseline_of(TEN), make_run({i: "PPP" for i in list(TEN)[:9]}))


@pytest.mark.parametrize("params", [
    {"candidate_repeats": None, "overall_margin_pts": 7},
    {"candidate_repeats": 3, "overall_margin_pts": None},
])
def test_gate_refuses_until_marko_has_set_both_parameters(params):
    baseline = baseline_of(TEN)
    baseline["gate"] = params

    with pytest.raises(GateRefused, match="gate parameters not set"):
        gate(baseline, make_run({i: "PPP" for i in TEN}))


# --- baseline builder -------------------------------------------------------------------------

def test_baseline_holds_exact_item_patterns_rates_noise_and_empty_gate_parameters():
    run = make_run({"a": "PPPPPP", "b": "PPPFFF", "c": "FFFFFF"},
                   {"a": "factual", "b": "factual", "c": "versioning"}, config={"name": "base"})

    baseline = build_baseline(run, fallback_instrument={}, changed=[], note="unclassified")

    assert baseline["status"] == "provisional" and baseline["approved_by"] is None
    assert baseline["gate"] == {"candidate_repeats": None, "overall_margin_pts": None}
    assert baseline["rates"]["items"]["b"] == {"category": "factual", "rate": 0.5,
                                               "pattern": "PPPFFF"}
    assert baseline["rates"]["overall"]["rate"] == 0.5
    assert baseline["rates"]["categories"] == {"factual": {"rate": 0.75, "n_items": 2},
                                               "versioning": {"rate": 0.0, "n_items": 1}}
    # Repeat r4-r6 lose b's pass: per repeat 2/3, 2/3, 2/3, 1/3, 1/3, 1/3.
    assert baseline["noise"]["per_repeat_rates"] == [0.666667] * 3 + [0.333333] * 3
    assert baseline["noise"]["no_change_bootstrap"]["mean_pts"] == -33.333333
    assert baseline["noise"]["safety_failures"] == 0
    assert baseline["instrument"]["judge_key"] == KEY


def test_baseline_takes_todays_hashes_only_for_what_the_run_did_not_record():
    run = make_run(TEN, config={"instrument": None}, graded={"evaluator_sha256": None})
    run.evaluations[KEY].instrument = None
    fallback = {"docs_sha256": "1" * 64, "users_sha256": "2" * 64, "evaluator_sha256": "3" * 64}

    baseline = build_baseline(run, fallback_instrument=fallback, changed=[], note="")

    assert {k: baseline["instrument"][k] for k in fallback} == fallback


def test_hashes_a_run_recorded_win_over_todays_files():
    # A baseline rebuilt from a Phase 5 run must describe that run, not today's files.
    run = make_run(TEN)
    todays = {"docs_sha256": "1" * 64, "users_sha256": "2" * 64, "evaluator_sha256": "3" * 64}

    baseline = build_baseline(run, fallback_instrument=todays, changed=[], note="")

    assert {k: baseline["instrument"][k] for k in todays} == HASHES


@pytest.mark.parametrize("config, changed, reason", [
    ({}, ["evals/evaluators.py"], "instrument files changed since the run's commit c1: "
                                  "evals/evaluators.py"),
    ({"filter": {"items": ["i00"], "categories": None}}, [], "a filtered dev run"),
    ({"git_dirty": True}, [], "run generated on a dirty or unknown tree"),
])
def test_baseline_is_refused_when_todays_files_would_not_describe_the_run(config, changed, reason):
    run = make_run(TEN, config=config)

    with pytest.raises(GateRefused) as refused:
        build_baseline(run, fallback_instrument={}, changed=changed, note="")

    assert any(r.startswith(reason) for r in refused.value.reasons)


# --- command line -----------------------------------------------------------------------------

def write(tmp_path, baseline: dict, cand: RunResults):
    base_path, cand_path = tmp_path / "baseline.json", tmp_path / "cand.json"
    base_path.write_text(json.dumps(baseline), encoding="utf-8")
    save(cand, cand_path)
    return base_path, cand_path


@pytest.mark.parametrize("cand, code, first", [
    (make_run({i: "PPP" for i in TEN}), 0, "REGRESSION GATE"),
    (make_run({i: "FFF" for i in TEN}), 1, "REGRESSION GATE"),
    (make_run(TEN), 1, "[refused]"),
])
def test_stats_gate_exits_0_on_pass_and_1_on_fail_or_refusal(tmp_path, monkeypatch, capsys,
                                                             cand, code, first):
    base_path, cand_path = write(tmp_path, baseline_of(TEN), copy.deepcopy(cand))
    monkeypatch.setattr("sys.argv", ["stats", "gate", str(base_path), str(cand_path)])

    assert stats_main() == code
    assert capsys.readouterr().out.startswith(first)


def test_run_gate_marks_a_provisional_baseline_in_the_report(tmp_path):
    base_path, cand_path = write(tmp_path, baseline_of(TEN), make_run({i: "PPP" for i in TEN}))

    passed, report = run_gate(base_path, cand_path)

    assert passed and report.splitlines()[1] == "PROVISIONAL BASELINE: test"


def test_baseline_command_never_overwrites_an_existing_baseline(tmp_path, monkeypatch, capsys):
    out = tmp_path / "baseline.json"
    out.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["gate", "baseline", "results/x.json", "--out", str(out)])

    assert main() == 1
    assert out.read_text(encoding="utf-8") == "{}"
    assert "only replaced with Marko's approval" in capsys.readouterr().out
