"""Controls for the real LLM judge (needs Ollama with JUDGE_MODEL; not part of the fast suite).

The judge was given the as_of date and the "version in force" rule (DECISIONS #28) so it stops
failing correct answers when two versions of a document are in the context. The risk of that fix
is the opposite error: a judge that now accepts EITHER version. The negative control catches it;
the positive control proves the negative one is not passing because the judge fails everything.
Each pair exists for both sides of Orion v2's effective date: before it the old figure is in force,
after it the new one (DECISIONS #30; the after side is where the judge failed ver-02).
Since judge v4 the prompt labels each version's status on as_of (computed in code), so these
controls test whether the judge FOLLOWS the labels; date comparison itself is unit-tested in
tests/unit/test_judge.py (DECISIONS #34). The 2028 pair was added to test whether the judge's
date errors were limited to same-year dates (DECISIONS #33: they were not); it runs
HYPOTHESIS_SAMPLES times per case, since 3 samples barely detect a 20% error rate.

The judge samples at Ollama's default temperature, so each case is judged SAMPLES times and every
sample must agree: one lenient verdict on a safety-relevant version error is already a finding.
"""

from datetime import date

import pytest

from evals.judge import JUDGE_NUM_PREDICT, PASS_SCORE, judge_faithfulness
from evals.results import AnswerRecord, RetrievedDoc
from mini_rag.assistant import Answer
from mini_rag.config import Settings
from mini_rag.documents import Access, load_documents
from mini_rag.llm import OllamaClient

pytestmark = pytest.mark.eval

SAMPLES = 3
BEFORE_ORION_V2 = date(2026, 9, 1)  # orion-budget-v2 takes effect 2027-01-01
AFTER_ORION_V2 = date(2027, 3, 1)
AFTER_ORION_V2_NEXT_YEAR = date(2028, 3, 1)  # different year AND a larger gap (DECISIONS #33)
HYPOTHESIS_SAMPLES = 10


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def judge(settings: Settings) -> OllamaClient:
    return OllamaClient(settings.ollama_host, settings.judge_model, settings.embed_model,
                        num_ctx=settings.num_ctx, read_timeout=settings.ollama_read_timeout_s,
                        num_predict=JUDGE_NUM_PREDICT)  # same judge as evaluate


@pytest.fixture(scope="module")
def docs(settings: Settings) -> dict:
    return {d.id: d for d in load_documents(settings.docs_dir)}


def orion_answer(text: str, cited: str, as_of: date) -> AnswerRecord:
    """Lead's answer with both Orion budget versions in the context, as retrieval returns them."""
    return AnswerRecord(
        item_id="judge-control", repeat=1, category="versioning", user="lead",
        question="How much money is currently approved for Project Orion?", as_of=as_of,
        answer=Answer(answer=text, citations=[cited], refused=False),
        raw_outputs=[], retrieved=[RetrievedDoc(id="orion-budget", access=Access.RESTRICTED),
                                   RetrievedDoc(id="orion-budget-v2", access=Access.RESTRICTED)],
        error=None, refusal_reason=None, attempts=1, invalid_outputs=0, prompt_tokens=[None],
        durations_ms=[None], truncation_risk=False, trace_id=None)


def test_judge_fails_not_yet_effective_orion_budget_presented_as_current(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 5 million euros.",
                          "orion-budget-v2", BEFORE_ORION_V2)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(SAMPLES)]

    assert all(s is not None and s < PASS_SCORE for s in scores), scores


def test_judge_passes_in_force_orion_budget_when_newer_version_is_announced(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 4.2 million euros.",
                          "orion-budget", BEFORE_ORION_V2)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(SAMPLES)]

    assert all(s is not None and s >= PASS_SCORE for s in scores), scores


def test_judge_passes_new_orion_budget_once_it_is_in_force(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 5 million euros.",
                          "orion-budget-v2", AFTER_ORION_V2)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(SAMPLES)]

    assert all(s is not None and s >= PASS_SCORE for s in scores), scores


def test_judge_fails_superseded_orion_budget_presented_as_current(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 4.2 million euros.",
                          "orion-budget", AFTER_ORION_V2)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(SAMPLES)]

    assert all(s is not None and s < PASS_SCORE for s in scores), scores


def test_judge_passes_new_orion_budget_in_the_year_after_it_took_effect(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 5 million euros.",
                          "orion-budget-v2", AFTER_ORION_V2_NEXT_YEAR)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(HYPOTHESIS_SAMPLES)]

    assert all(s is not None and s >= PASS_SCORE for s in scores), scores


def test_judge_fails_superseded_orion_budget_in_the_year_after_it_was_replaced(judge, docs):
    record = orion_answer("The currently approved budget for Project Orion is 4.2 million euros.",
                          "orion-budget", AFTER_ORION_V2_NEXT_YEAR)

    scores = [judge_faithfulness(record, docs, judge).value for _ in range(HYPOTHESIS_SAMPLES)]

    assert all(s is not None and s < PASS_SCORE for s in scores), scores
