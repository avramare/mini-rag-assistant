import json

import pytest

from mini_rag.users import User
from tests.helpers import RESTRICTED_SECRET


def reply(answer: str = "27 days.", citations: list[str] | None = None, refused: bool = False):
    return json.dumps({"answer": answer, "citations": citations or ["holiday-policy"],
                       "refused": refused})


HOLIDAY_Q = "How many days of paid holiday do employees get?"


def test_valid_json_answer_is_parsed(make_assistant, analyst: User):
    assistant, llm = make_assistant(reply())

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    assert result.answer.answer == "27 days."
    assert result.answer.citations == ["holiday-policy"]
    assert (result.attempts, result.invalid_outputs) == (1, 0)


def test_invalid_json_retried_once_then_succeeds(make_assistant, analyst: User):
    assistant, llm = make_assistant("Sure! Here is the answer: 27 days", reply())

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    assert (result.attempts, result.invalid_outputs) == (2, 1)
    assert "not valid JSON" in llm.calls[1][1]


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


@pytest.mark.parametrize("cited", [["orion-budget"], ["holiday-policy", "made-up-doc"]])
def test_citation_outside_retrieved_set_is_rejected(make_assistant, analyst: User,
                                                    cited: list[str]):
    assistant, _ = make_assistant(reply(citations=cited))

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.error == "invalid_citation"
    assert result.answer is None
    assert "orion-budget" not in result.retrieved_ids


def test_llm_prompt_contains_no_restricted_text_for_analyst(make_assistant, analyst: User):
    assistant, llm = make_assistant(reply(answer="I cannot answer.", refused=True, citations=[]))

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
