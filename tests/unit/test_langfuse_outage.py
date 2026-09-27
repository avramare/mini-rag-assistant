"""A REAL Langfuse client against an httpx MockTransport that answers every request with 503:
the errors the SDK really raises must be caught by `FailSafePublisher`, not only a fake's.

Dataset upload and run links are synchronous API calls: they raise, and are recorded. Scores are
only queued (sent by a background thread; the SDK logs delivery errors itself), so they are not
exercised here; the fake-publisher tests in test_run_experiment cover their bookkeeping.
"""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from evals.dataset import load_dataset
from evals.publish import FailSafePublisher, LangfusePublisher
from evals.results import PublishFailure
from mini_rag.config import Settings
from mini_rag.tracing import make_langfuse
from tests.helpers import dataset_item, write_dataset


class Unavailable:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        return httpx.Response(503, json={"message": "Service Unavailable"})


@pytest.fixture
def down(monkeypatch) -> Iterator[tuple[Unavailable, LangfusePublisher]]:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    # The SDK retries a 5xx twice with 1 s, 2 s backoff (~3 s per call). Zero delay keeps the
    # suite fast with the same requests and errors; setattr fails loudly if the SDK renames it.
    monkeypatch.setattr("langfuse.api.core.http_client.INITIAL_RETRY_DELAY_SECONDS", 0.0)
    server = Unavailable()
    client = make_langfuse(
        Settings(langfuse_public_key=SecretStr("pk-lf-test-outage"),
                 langfuse_secret_key=SecretStr("sk-lf-test-outage"),
                 langfuse_host="https://langfuse.invalid"),
        httpx_client=httpx.Client(transport=httpx.MockTransport(server)))
    yield server, LangfusePublisher(client)
    client.shutdown()


def test_real_sdk_503_on_dataset_upload_and_run_links_is_recorded_not_raised(tmp_path: Path,
                                                                             down):
    server, publisher = down
    dataset = load_dataset(write_dataset(tmp_path / "d.jsonl", [
        dataset_item("hol", "analyst", "Holiday?", expected=["27 days"])]), {"analyst"})
    failures: list[PublishFailure] = []
    safe = FailSafePublisher(publisher, failures.append, log=lambda _: None)

    safe.upload_dataset(dataset)
    safe.link(dataset, dataset.items[0], "run-r1", "a" * 32, {}, repeat=1)

    # control: the SDK really called the failing API, retries included (3 tries per call)
    assert server.requests.count("/api/public/v2/datasets") == 3
    assert [(f.kind, f.item_id, f.run_name) for f in failures] == [
        ("dataset", None, None), ("link", "hol", "run-r1")]
    assert all("503" in f.error or "Unavailable" in f.error for f in failures)
