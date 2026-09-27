"""`version_status`: the judge's own, code-side answer to "which version is in force on as_of".

qwen3:4b misordered 2027-01-01 and 2027-03-01 (DECISIONS #30), so v4 stops asking it to compare
dates (#34). These tests are now where date comparison is checked: same-year pairs (where the
judge failed), different-year pairs, and the boundary day.
"""

import json
from datetime import date

import pytest

from evals.judge import VersionStatus, judge_faithfulness, version_status
from evals.results import AnswerRecord, RetrievedDoc
from mini_rag.assistant import Answer
from mini_rag.documents import Access, Document
from mini_rag.llm import FakeLLM

IN_FORCE, SUPERSEDED, NOT_YET = (VersionStatus.IN_FORCE, VersionStatus.SUPERSEDED,
                                 VersionStatus.NOT_YET_IN_FORCE)


def doc(doc_id: str, effective: str | None = None, supersedes: str | None = None) -> Document:
    return Document(id=doc_id, title=doc_id, access=Access.PUBLIC, body="Text.",
                    effective=date.fromisoformat(effective) if effective else None,
                    supersedes=supersedes)


# Same dates as the real Orion pair: v2 takes effect 2027-01-01.
ORION = {d.id: d for d in [doc("budget", "2026-03-14"),
                           doc("budget-v2", "2027-01-01", supersedes="budget"),
                           doc("overview")]}


@pytest.mark.parametrize("as_of, old, new", [
    # same year as v2's effective date: where the LLM judge misordered the dates
    ("2027-03-01", SUPERSEDED, IN_FORCE),
    ("2027-01-01", SUPERSEDED, IN_FORCE),  # the effective day itself counts ("not after")
    # same year as v1's effective date, v2 announced
    ("2026-12-31", IN_FORCE, NOT_YET),
    ("2026-09-01", IN_FORCE, NOT_YET),
    # a different year from both
    ("2028-03-01", SUPERSEDED, IN_FORCE),
    ("2025-06-01", NOT_YET, NOT_YET),  # before either version: nothing is in force yet
])
def test_version_in_force_is_decided_by_date_for_same_and_different_years(as_of, old, new):
    status = version_status(ORION, date.fromisoformat(as_of))

    assert (status["budget"], status["budget-v2"]) == (old, new)


def test_docs_outside_a_version_chain_get_no_status():
    # A label on every doc would tell the judge a standalone doc is "one of several versions".
    assert "overview" not in version_status(ORION, date(2027, 3, 1))


def test_first_version_is_superseded_through_a_chain_of_three():
    docs = {d.id: d for d in [doc("p", "2025-01-01"), doc("p-v2", "2026-01-01", supersedes="p"),
                              doc("p-v3", "2027-01-01", supersedes="p-v2")]}

    mid, late = version_status(docs, date(2026, 6, 1)), version_status(docs, date(2027, 6, 1))

    assert (mid["p"], mid["p-v2"], mid["p-v3"]) == (SUPERSEDED, IN_FORCE, NOT_YET)
    assert (late["p"], late["p-v2"], late["p-v3"]) == (SUPERSEDED, SUPERSEDED, IN_FORCE)


def test_judge_prompt_labels_each_version_with_its_status_on_the_items_as_of():
    # The status comes from the whole corpus: the prompt labels exactly what was retrieved.
    llm = FakeLLM([json.dumps({"reason": "Supported.", "score": 5})])
    record = AnswerRecord(
        item_id="x", repeat=1, category="versioning", user="lead", question="Budget?",
        as_of=date(2027, 3, 1), answer=Answer(answer="5M.", citations=["budget-v2"],
                                               refused=False),
        raw_outputs=[], retrieved=[RetrievedDoc(id=i, access=Access.PUBLIC)
                                   for i in ["budget", "budget-v2", "overview"]],
        error=None, refusal_reason=None, attempts=1, invalid_outputs=0, prompt_tokens=[None],
        durations_ms=[None], truncation_risk=False, trace_id=None)

    judge_faithfulness(record, ORION, llm)

    prompt = llm.calls[0][1]
    assert "[doc id: budget] budget (effective 2026-03-14; status on AS OF: superseded)" in prompt
    assert ("[doc id: budget-v2] budget-v2 (effective 2027-01-01; status on AS OF: in force)"
            in prompt)
    assert "[doc id: overview] overview\n" in prompt
