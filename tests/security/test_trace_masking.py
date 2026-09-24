"""Restricted text must never leave the process through tracing.

These tests use a REAL Langfuse client. Only the edges are swapped: spans go to an in-memory OTel
exporter instead of the network, and the client's HTTP client answers 500 and records every request.
What the exporter receives is exactly what would be sent to Langfuse, so the assertions check the
full export (names, attributes, events, resource), not our own payload dicts.
"""

import json
from collections.abc import Iterator

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from mini_rag.assistant import Assistant
from mini_rag.config import Settings
from mini_rag.llm import Generation
from mini_rag.tracing import LangfuseTracer, make_tracer
from mini_rag.users import User
from tests.helpers import RESTRICTED_SECRET

pytestmark = pytest.mark.security

BUDGET_FIGURE = "4.2 million"  # from the restricted fixture doc orion-budget
ORION_Q = "What is the Orion project budget and codename?"
HOLIDAY_Q = "How many days of paid holiday do employees get?"


def reply(answer: str, citations: list[str]) -> str:
    return json.dumps({"answer": answer, "citations": citations, "refused": False})


def assert_run_config_on(root: dict, assistant: Assistant) -> None:
    # Masking must hide content, never the config: without it, masked runs can't be compared.
    config = assistant.run_config()
    assert None not in config.values(), "precondition: Langfuse drops None metadata values"
    for key, value in config.items():
        assert root[f"langfuse.observation.metadata.{key}"] == value, key


class Capture:
    """A real LangfuseTracer whose output lands in memory, plus a log of attempted HTTP requests."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.requests: list[httpx.Request] = []
        settings = Settings(langfuse_public_key=SecretStr("pk-lf-test-masking"),
                            langfuse_secret_key=SecretStr("sk-lf-test-masking"),
                            langfuse_host="https://langfuse.invalid")
        self.tracer = make_tracer(
            settings,
            span_exporter=self.exporter,
            tracer_provider=TracerProvider(),  # private provider: no global OTel state
            httpx_client=httpx.Client(transport=httpx.MockTransport(self._refuse)),
        )
        assert isinstance(self.tracer, LangfuseTracer)

    def _refuse(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(500)

    def spans(self) -> tuple[ReadableSpan, ...]:
        self.tracer.flush()  # spans are exported in batches; flush so the exporter has them all
        return self.exporter.get_finished_spans()

    def sent(self) -> str:
        """Everything exported, as one string: the haystack for leak checks."""
        return "\n".join(span.to_json() for span in self.spans())

    def attrs(self, name: str) -> dict:
        (span,) = [s for s in self.spans() if s.name == name]
        return dict(span.attributes)


@pytest.fixture(scope="module")
def shared_capture() -> Capture:
    # Module scope: Langfuse keeps one client per public key for the whole process, so all tests
    # here share one client and reset the capture between tests.
    return Capture()


@pytest.fixture
def capture(shared_capture: Capture) -> Iterator[Capture]:
    shared_capture.exporter.clear()
    shared_capture.requests.clear()
    yield shared_capture
    assert shared_capture.requests == [], "tracing must not touch the network in tests"


def test_lead_orion_trace_exports_neither_codename_nor_budget(make_assistant, capture: Capture,
                                                              lead: User):
    assistant, llm = make_assistant(Generation(
        reply(f"The budget is {BUDGET_FIGURE} euros; codename {RESTRICTED_SECRET}.",
              ["orion-budget"]),
        prompt_tokens=321, completion_tokens=17, duration_ms=5.0,
    ), tracer=capture.tracer, num_ctx=4096)

    result = assistant.answer(ORION_Q, lead, dataset_item_id="q005")

    # Preconditions: the secret was in the prompt and in the answer, so a leak was possible.
    assert "orion-budget" in result.retrieved_ids
    assert RESTRICTED_SECRET in llm.calls[0][1]
    assert RESTRICTED_SECRET in result.answer.answer

    sent = capture.sent()
    assert RESTRICTED_SECRET not in sent
    assert BUDGET_FIGURE not in sent
    assert ORION_Q not in sent  # the question is part of the prompt and can name secrets too

    # What stays visible for debugging: structure, not content.
    root = capture.attrs("answer")
    assert root["user.id"] == "lead"
    assert root["langfuse.observation.metadata.masked"] is True
    retrieved = json.loads(root["langfuse.observation.metadata.retrieved"])
    assert ("orion-budget", "restricted") in {(r["id"], r["access"]) for r in retrieved}
    assert_run_config_on(root, assistant)
    assert root["langfuse.observation.metadata.dataset_item_id"] == "q005"
    gen = capture.attrs("generation")
    assert gen["langfuse.observation.model.name"] == llm.model
    assert json.loads(gen["langfuse.observation.usage_details"]) == {"input": 321, "output": 17}


def test_public_only_trace_keeps_prompt_and_answer_text(make_assistant, capture: Capture,
                                                        analyst: User):
    # Positive control: proves the export does carry prompt and answer text, so the absence checks
    # in the other tests are not passing because nothing is exported at all.
    answer_text = "Employees get 27 days per year."
    assistant, _ = make_assistant(reply(answer_text, ["holiday-policy"]), tracer=capture.tracer,
                                  num_ctx=4096)

    result = assistant.answer(HOLIDAY_Q, analyst)

    assert result.ok
    sent = capture.sent()
    assert answer_text in sent
    assert HOLIDAY_Q in sent
    assert "27 days of paid holiday per year" in sent  # doc body: only reachable via the prompt
    root = capture.attrs("answer")
    assert root["langfuse.observation.metadata.masked"] is False
    assert_run_config_on(root, assistant)


def test_restricted_retrieval_masks_even_when_answer_cites_only_public_doc(make_assistant,
                                                                          capture: Capture,
                                                                          lead: User):
    # The model saw the restricted doc, leaks its figure, but cites only a public doc.
    # Masking must follow what was retrieved, not what was cited.
    assistant, _ = make_assistant(reply(f"Orion costs {BUDGET_FIGURE} euros.", ["orion-overview"]),
                                  tracer=capture.tracer)

    result = assistant.answer(ORION_Q, lead)

    assert {"orion-budget", "orion-overview"} <= set(result.retrieved_ids)
    assert result.ok and result.answer.citations == ["orion-overview"]
    assert BUDGET_FIGURE not in capture.sent()


def test_every_attempt_output_masked_on_retry(make_assistant, capture: Capture, lead: User):
    assistant, _ = make_assistant(
        f"not json, but it mentions {RESTRICTED_SECRET}",
        reply(f"Codename {RESTRICTED_SECRET}.", ["orion-budget"]),
        tracer=capture.tracer,
    )

    result = assistant.answer(ORION_Q, lead)

    assert result.attempts == 2
    attempts = [s.attributes["langfuse.observation.metadata.attempt"]
                for s in capture.spans() if s.name == "generation"]
    assert sorted(attempts) == [1, 2]
    assert RESTRICTED_SECRET not in capture.sent()
