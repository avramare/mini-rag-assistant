import json
from datetime import date

import pytest

from mini_rag.assistant import ANSWER_SCHEMA, SYSTEM_PROMPT, UNCITED_REFUSAL
from mini_rag.llm import Generation
from mini_rag.users import User
from tests.helpers import RESTRICTED_SECRET


def reply(answer: str = "27 days.", citations: list[str] | None = None, refused: bool = False):
    return json.dumps({"answer": answer, "citations": ["holiday-policy"] if citations is None
                       else citations, "refused": refused})


HOLIDAY_Q = "How many days of paid holiday do employees get?"
REFUSAL = reply(answer="The documents do not say.", citations=[], refused=True)


def test_valid_answer_passes_through_unaltered_with_single_attempt(make_assistant, analyst: User):
    assistant, llm = make_assistant(reply())

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    assert result.answer.answer == "27 days."
    assert result.answer.citations == ["holiday-policy"]
    assert result.refusal_reason is None
    assert (result.attempts, result.invalid_outputs) == (1, 0)


def test_model_is_asked_for_answer_json_schema(make_assistant, analyst: User):
    assistant, llm = make_assistant(reply())

    assistant.answer(HOLIDAY_Q, analyst)

    assert llm.schemas == [ANSWER_SCHEMA]
    assert ANSWER_SCHEMA["additionalProperties"] is False


def test_invalid_json_retried_once_then_succeeds(make_assistant, analyst: User):
    assistant, llm = make_assistant("Sure! Here is the answer: 27 days", reply())

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    assert (result.attempts, result.invalid_outputs) == (2, 1)
    assert "not valid JSON" in llm.calls[1][1]


def test_retry_keeps_system_prompt_and_filtered_context(make_assistant, analyst: User):
    assistant, llm = make_assistant("not json", reply())

    assistant.answer(HOLIDAY_Q, analyst)

    (system_1, prompt_1), (system_2, prompt_2) = llm.calls
    assert system_1 == system_2 == SYSTEM_PROMPT
    assert "[doc id: holiday-policy]" in prompt_1
    assert prompt_2.startswith(prompt_1)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("not json", id="not-json"),
        pytest.param('{"answer": "x"}', id="missing-fields"),
        pytest.param('{"answer": "x", "citations": "holiday-policy", "refused": false}',
                     id="citations-not-list"),
        pytest.param('{"answer": "x", "citations": [], "refused": false, "note": "hi"}',
                     id="extra-field"),
    ],
)
def test_invalid_output_twice_is_counted_not_raised(make_assistant, analyst: User, bad: str):
    # Only two responses scripted: a third call would make FakeLLM raise, proving max one retry.
    assistant, llm = make_assistant(bad, bad)

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.error == "invalid_output"
    assert result.answer is None
    assert (result.attempts, result.invalid_outputs) == (2, 2)
    assert result.raw_outputs == [bad, bad]


@pytest.mark.parametrize(
    "cited",
    [
        # Exists and is public, but k=1 retrieved only holiday-policy. Independent of access.
        pytest.param(["budget-process"], id="real-doc-not-retrieved"),
        pytest.param(["holiday-policy", "made-up-doc"], id="made-up-doc"),
    ],
)
def test_citation_outside_retrieved_set_is_rejected(make_assistant, lead: User,
                                                    cited: list[str]):
    assistant, _ = make_assistant(reply(citations=cited), k=1)

    result = assistant.answer(HOLIDAY_Q, lead)

    assert result.retrieved_ids == ["holiday-policy"]  # precondition
    assert result.error == "invalid_citation"
    assert result.answer is None


def test_non_refused_answer_without_citations_becomes_uncited_refusal(make_assistant,
                                                                      analyst: User):
    ungrounded = reply(answer="Probably 25 days.", citations=[])
    assistant, _ = make_assistant(ungrounded)

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    assert result.answer.refused is True
    assert result.answer.answer == UNCITED_REFUSAL
    assert result.refusal_reason == "uncited"
    assert result.raw_outputs == [ungrounded]  # the original stays available for evals


def test_user_without_clearance_gets_empty_context_and_refusal_survives(make_assistant):
    nobody = User(name="nobody", clearance=frozenset())
    assistant, llm = make_assistant(REFUSAL)

    result = assistant.answer(HOLIDAY_Q, nobody)

    assert result.retrieved_ids == []
    assert "(no documents available)" in llm.calls[0][1]
    assert result.ok
    assert result.answer.refused is True
    assert result.refusal_reason == "model"


def test_citation_with_empty_context_is_rejected(make_assistant):
    nobody = User(name="nobody", clearance=frozenset())
    assistant, _ = make_assistant(reply())

    result = assistant.answer(HOLIDAY_Q, nobody)

    assert result.error == "invalid_citation"


def test_llm_prompt_contains_no_restricted_text_for_analyst(make_assistant, analyst: User):
    assistant, llm = make_assistant(REFUSAL)

    assistant.answer("What is the Orion project budget and codename?", analyst)

    system, prompt = llm.calls[0]
    assert RESTRICTED_SECRET not in prompt
    assert "4.2 million" not in prompt


def test_llm_prompt_includes_restricted_text_for_cleared_user(make_assistant, lead: User):
    # Positive control for the test above: proves the needle would be found if it leaked.
    assistant, llm = make_assistant(reply(answer="4.2 million euros.", citations=["orion-budget"]))

    result = assistant.answer("What is the Orion project budget and codename?", lead)

    assert RESTRICTED_SECRET in llm.calls[0][1]
    assert result.ok


def test_prompt_shows_effective_date_so_model_can_pick_current_version(make_assistant,
                                                                        analyst: User):
    assistant, llm = make_assistant(reply())

    assistant.answer(HOLIDAY_Q, analyst)

    assert "[doc id: holiday-policy] Holiday policy (effective 2026-06-01)" in llm.calls[0][1]


@pytest.mark.parametrize(
    "today",
    [
        pytest.param(date(2026, 12, 31), id="before-future-version"),
        pytest.param(date(2027, 1, 2), id="after-future-version"),
    ],
)
def test_prompt_states_injected_date_not_wall_clock(make_assistant, analyst: User, today: date):
    # "Latest effective date not after today" only works if the model is told the right today.
    assistant, llm = make_assistant(reply(), today=today)

    assistant.answer(HOLIDAY_Q, analyst)

    assert llm.calls[0][1].startswith(f"Today is {today.isoformat()}.")


@pytest.mark.parametrize(("prompt_tokens", "expected"),
                         [(3500, True), (3072, False), (None, False)])
def test_truncation_risk_flags_prompt_near_context_limit(make_assistant, analyst: User,
                                                         prompt_tokens: int | None,
                                                         expected: bool):
    # 0.75 * 4096 = 3072. Ollama truncates silently, so the reported token count is our only signal.
    assistant, _ = make_assistant(Generation(reply(), prompt_tokens=prompt_tokens),
                                  num_ctx=4096, max_prompt_ctx_share=0.75)

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.truncation_risk is expected
    assert result.prompt_tokens == [prompt_tokens]
