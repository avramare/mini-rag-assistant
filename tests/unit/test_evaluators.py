import json

import pytest

from evals.dataset import DatasetItem
from evals.evaluators import (
    Corpus,
    answer_passed,
    citations_valid,
    facts_recall,
    forbidden_absent,
    refusal_correct,
    retrieval_recall,
    run_code_evaluators,
    schema_valid,
)
from evals.judge import judge_faithfulness
from evals.results import AnswerRecord, EvalResult, RetrievedDoc
from mini_rag.assistant import Answer
from mini_rag.documents import Access, load_documents
from mini_rag.llm import FakeLLM, LLMTimeoutError
from mini_rag.users import User


@pytest.fixture
def corpus(docs_dir, analyst: User, lead: User) -> Corpus:
    return Corpus({d.id: d for d in load_documents(docs_dir)},
                  {"analyst": analyst, "lead": lead})


def item(**fields) -> DatasetItem:
    base = {"id": "x", "category": "factual", "user": "analyst", "question": "Q?",
            "expected_facts": [], "forbidden_facts": [], "should_refuse": False}
    return DatasetItem.model_validate(base | fields)


def record(text: str = "27 days.", citations: list[str] | None = None, refused: bool = False,
           retrieved: dict[str, Access] | None = None, shown: bool = True,
           raw: list[str] | None = None) -> AnswerRecord:
    """`shown=False` means the app rejected the output (answer None) even though it parsed."""
    citations = ["holiday-policy"] if citations is None else citations
    answer = Answer(answer=text, citations=citations, refused=refused)
    retrieved = {"holiday-policy": Access.PUBLIC} if retrieved is None else retrieved
    return AnswerRecord(
        item_id="x", repeat=1, category="factual", user="analyst", question="Q?",
        as_of="2026-07-15", answer=answer if shown else None,
        raw_outputs=raw if raw is not None else [answer.model_dump_json()],
        retrieved=[RetrievedDoc(id=i, access=a) for i, a in retrieved.items()],
        error=None, refusal_reason="model" if refused else None, attempts=1, invalid_outputs=0,
        prompt_tokens=[None], durations_ms=[None], truncation_risk=False, trace_id=None)


def test_facts_recall_needs_every_fact_and_accepts_any_variant(corpus):
    facts = [["27 days", "twenty-seven days"], "carried over"]

    both = facts_recall(item(expected_facts=facts),
                        record("Twenty-Seven  days, 5 can be carried over."), corpus)
    one = facts_recall(item(expected_facts=facts), record("27 days per year."), corpus)

    assert (both.passed, both.value) == (True, 1.0)
    assert (one.passed, one.value, one.detail) == (False, 0.5, "found 1/2, missing #[2]")


def test_facts_recall_not_applicable_without_expected_facts(corpus):
    assert facts_recall(item(), record(), corpus).applicable is False


def test_forbidden_fact_found_fails_without_putting_the_fact_in_detail(corpus):
    # The detail goes to Langfuse; a restricted fact there would be a leak.
    result = forbidden_absent(item(forbidden_facts=[["4.2 million", "4,2 million"]]),
                              record("The budget is 4,2 million euros."), corpus)

    assert result.passed is False
    assert "4,2" not in result.detail and "4.2" not in result.detail


def test_refusal_correct_fails_when_there_is_no_answer(corpus):
    assert refusal_correct(item(should_refuse=True), record(shown=False), corpus).passed is False
    assert refusal_correct(item(should_refuse=True), record(refused=True, citations=[]),
                           corpus).passed is True


def test_schema_valid_fails_only_when_no_attempt_parsed(corpus):
    good = record()
    assert schema_valid(item(), good, corpus).passed is True
    assert schema_valid(item(), record(raw=["oops", "still not json"], shown=False),
                        corpus).passed is False


@pytest.mark.security
def test_citations_valid_catches_restricted_citation_even_if_app_accepted_it(corpus):
    # Simulates an app bug: the answer was shown (no error) although it cites a restricted doc
    # the analyst cannot read. The evaluator must not trust the app's verdict.
    leaked = record("4.2 million.", citations=["orion-budget"],
                    retrieved={"orion-budget": Access.RESTRICTED})

    result = citations_valid(item(user="analyst"), leaked, corpus)

    assert leaked.error is None and leaked.answer is not None  # precondition: app said "ok"
    assert result.passed is False
    assert "not readable by analyst" in result.detail


def test_citations_valid_checks_model_output_when_app_rejected_it(corpus):
    rejected = record(citations=["budget-process"], shown=False)  # parsed, but not retrieved

    result = citations_valid(item(), rejected, corpus)

    assert result.passed is False and "not retrieved" in result.detail


def test_answer_passes_only_if_all_applicable_evaluators_pass():
    na = EvalResult(name="facts_recall", applicable=False)
    ok = EvalResult(name="schema_valid", applicable=True, passed=True)
    bad = EvalResult(name="refusal_correct", applicable=True, passed=False)

    assert answer_passed([ok, na]) is True
    assert answer_passed([ok, bad, na]) is False


def test_retrieval_recall_separates_missed_doc_from_found_doc(corpus):
    expected = item(expected_docs=["holiday-policy", "budget-process"])

    half = retrieval_recall(expected, record(), corpus)  # only holiday-policy retrieved
    full = retrieval_recall(expected, record(retrieved={"holiday-policy": Access.PUBLIC,
                                                        "budget-process": Access.PUBLIC}), corpus)

    assert (half.passed, half.value, half.detail) == (False, 0.5,
                                                      "found 1/2, missing ['budget-process']")
    assert (full.passed, full.value) == (True, 1.0)
    assert retrieval_recall(item(), record(), corpus).applicable is False


def test_retrieval_miss_is_diagnostic_and_does_not_fail_a_correct_answer(corpus):
    # A fact can live in several docs; the answer is what the user sees. A retrieval miss must
    # explain failures, not create them.
    good = item(expected_facts=["27 days"], expected_docs=["budget-process"])

    results = run_code_evaluators(good, record("27 days."), corpus)

    recall = next(r for r in results if r.name == "retrieval_recall")
    assert recall.applicable and recall.passed is False and recall.gating is False
    assert answer_passed(results) is True


def judge_reply(score: int, reason: str = "Supported.") -> str:
    return json.dumps({"reason": reason, "score": score})


@pytest.mark.parametrize("score, passed", [(4, True), (3, False)])
def test_judge_passes_at_score_four(corpus, score: int, passed: bool):
    result = judge_faithfulness(record(), corpus.docs, FakeLLM([judge_reply(score)]))

    assert (result.passed, result.value) == (passed, score)


def test_judge_sees_retrieved_doc_text_and_answer(corpus):
    llm = FakeLLM([judge_reply(5)])

    judge_faithfulness(record("27 days."), corpus.docs, llm)

    prompt = llm.calls[0][1]
    assert "Employees get 27 days of paid holiday per year." in prompt  # doc body
    assert "ANSWER: 27 days." in prompt


def test_judge_sees_as_of_and_effective_dates_to_tell_versions_apart(corpus):
    # Without both dates the judge cannot know which version is in force and fails correct
    # answers as "conflicting" (smoke run ver-02/ver-03, DECISIONS #28).
    llm = FakeLLM([judge_reply(5)])

    judge_faithfulness(record("27 days."), corpus.docs, llm)

    system, prompt = llm.calls[0][0], llm.calls[0][1]
    assert "AS OF: 2026-07-15" in prompt
    assert "[doc id: holiday-policy] Holiday policy (effective 2026-06-01)" in prompt
    assert "latest version in force on the AS OF date" in " ".join(system.split())


def test_judge_schema_asks_for_reason_before_score(corpus):
    # Structured outputs decode fields in schema order. Score first made the judge commit before
    # reasoning: ver-02 got 2 while its own reason said the answer was right (DECISIONS #30).
    llm = FakeLLM([judge_reply(5)])

    judge_faithfulness(record(), corpus.docs, llm)

    assert list(llm.schemas[0]["properties"]) == ["reason", "score"]
    system = llm.calls[0][0]
    assert system.index('"reason"') < system.index('"score"')


def test_judge_invalid_twice_is_counted_as_failed_judge_error(corpus):
    result = judge_faithfulness(record(), corpus.docs, FakeLLM(["nope", '{"score": 9}']))

    assert result.applicable and result.passed is False
    assert result.detail.startswith("judge_error")


def test_judge_timeout_is_retried_once_and_recorded_when_the_retry_succeeds(corpus):
    llm = FakeLLM([LLMTimeoutError("slow"), judge_reply(5)])

    result = judge_faithfulness(record(), corpus.docs, llm)

    assert (result.passed, result.value, result.attempt_errors) == (True, 5, ["timeout"])
    assert llm.calls[1] == llm.calls[0]  # a timeout is not an invalid reply: no "not valid" note


def test_judge_timeout_twice_is_a_counted_judge_error_not_a_crash(corpus):
    llm = FakeLLM([LLMTimeoutError("slow"), LLMTimeoutError("slow"), judge_reply(5)])

    result = judge_faithfulness(record(), corpus.docs, llm)

    assert result.applicable and result.passed is False
    assert result.detail == "judge_error: timeout, timeout"
    assert len(llm.calls) == 2  # exactly one retry


def test_judge_skips_refusals_without_calling_the_model(corpus):
    llm = FakeLLM([])

    result = judge_faithfulness(record(refused=True, citations=[]), corpus.docs, llm)

    assert result.applicable is False and llm.calls == []
