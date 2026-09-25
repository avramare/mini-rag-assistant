"""Nothing restricted may leave the process through the eval pipeline's Langfuse calls.

A REAL Langfuse client talks to an httpx MockTransport that records every request body and answers
with minimal valid JSON, so the assertions check exactly what would be sent over the network.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from evals.dataset import load_dataset, variants
from evals.publish import LangfusePublisher, publish_scores
from evals.results import AnswerRecord, EvalResult, RetrievedDoc
from mini_rag.assistant import Answer
from mini_rag.config import Settings
from mini_rag.tracing import make_langfuse
from tests.helpers import RESTRICTED_SECRET, dataset_item, write_dataset

pytestmark = pytest.mark.security

NOW = datetime.now(UTC).isoformat()


def fake_langfuse_api(request: httpx.Request) -> httpx.Response:
    """Just enough of each response for the SDK to parse it."""
    path = request.url.path
    if "ingestion" in path:
        return httpx.Response(207, json={"successes": [], "errors": []})
    common = {"id": "x", "createdAt": NOW, "updatedAt": NOW, "metadata": None}
    if "dataset-run-items" in path:
        return httpx.Response(200, json=common | {
            "datasetRunId": "r", "datasetRunName": "r", "datasetItemId": "i", "traceId": "t"})
    if "dataset-items" in path:
        return httpx.Response(200, json=common | {
            "status": "ACTIVE", "input": None, "expectedOutput": None, "datasetId": "d",
            "datasetName": "d", "sourceTraceId": None, "sourceObservationId": None,
            "mediaReferences": []})
    if "datasets" in path:
        return httpx.Response(200, json=common | {"name": "d", "projectId": "p"})
    return httpx.Response(200, json={})


class Wire:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return fake_langfuse_api(request)

    def sent(self) -> str:
        return "\n".join(r.content.decode("utf-8", "replace") for r in self.requests)


@pytest.fixture(scope="module")
def shared() -> Iterator[tuple[Wire, LangfusePublisher]]:
    wire = Wire()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LANGFUSE_TRACING_ENABLED", "true")
        client = make_langfuse(
            Settings(langfuse_public_key=SecretStr("pk-lf-test-publish"),
                     langfuse_secret_key=SecretStr("sk-lf-test-publish"),
                     langfuse_host="https://langfuse.invalid"),
            httpx_client=httpx.Client(transport=httpx.MockTransport(wire.handle)))
        yield wire, LangfusePublisher(client)


@pytest.fixture
def wire_and_publisher(shared) -> tuple[Wire, LangfusePublisher]:
    wire, publisher = shared
    wire.requests.clear()
    return wire, publisher


RESTRICTED_FACTS = [["4.2 million", "4,2 million"], RESTRICTED_SECRET]


def test_dataset_upload_sends_questions_but_no_facts(tmp_path: Path, wire_and_publisher):
    wire, publisher = wire_and_publisher
    path = write_dataset(tmp_path / "d.jsonl", [
        dataset_item("probe", "analyst", "What is the budget of the pilot?",
                     category="restricted_probe", forbidden=RESTRICTED_FACTS,
                     should_refuse=True),
        dataset_item("cleared", "lead", "What is the pilot budget and its codename?",
                     category="restricted_probe", expected=RESTRICTED_FACTS),
    ])
    dataset = load_dataset(path, {"analyst", "lead"})

    publisher.upload_dataset(dataset)
    publisher.flush()

    sent = wire.sent()
    assert "What is the pilot budget and its codename?" in sent  # control: bodies are captured
    assert "restricted_probe" in sent
    for fact in RESTRICTED_FACTS:
        for variant in variants(fact):
            assert variant not in sent


def record(masked: bool, text: str) -> AnswerRecord:
    access = "restricted" if masked else "public"
    return AnswerRecord(
        item_id="x", repeat=1, category="restricted_probe", user="lead", question="Q?",
        as_of="2026-09-01", answer=Answer(answer=text, citations=["doc"], refused=False),
        raw_outputs=[], retrieved=[RetrievedDoc(id="doc", access=access)], error=None,
        refusal_reason=None, attempts=1, invalid_outputs=0, prompt_tokens=[None],
        durations_ms=[None], truncation_risk=False, trace_id="a" * 32)


def judged(reason: str) -> list[EvalResult]:
    return [EvalResult(name="judge_faithfulness", applicable=True, passed=True, value=5,
                       detail=reason)]


def test_judge_reason_quoting_restricted_answer_is_masked(wire_and_publisher):
    wire, publisher = wire_and_publisher
    reason = f"The answer states 4.2 million and {RESTRICTED_SECRET}, as the context does."

    publish_scores(publisher, record(True, "4.2 million"), judged(reason), passed=True)
    publisher.flush()

    sent = wire.sent()
    assert "judge_faithfulness" in sent  # the score itself was sent
    assert RESTRICTED_SECRET not in sent and "4.2 million" not in sent


def test_judge_reason_for_public_context_is_sent_unmasked(wire_and_publisher):
    # Positive control for the test above: the reason does reach the wire when it may.
    wire, publisher = wire_and_publisher

    publish_scores(publisher, record(False, "27 days"),
                   judged("The answer says 27 days, as stated."), passed=True)
    publisher.flush()

    assert "The answer says 27 days, as stated." in wire.sent()
    assert json.dumps("item_pass") in wire.sent()
