"""OllamaClient request/response contract, via httpx.MockTransport (real client code, no network).

Ollama never errors on a wrong or missing `num_ctx`: it uses its own default and silently
truncates. These tests are the only guard that what we compare against is what we sent.
"""

import json

import httpx
import pytest

from mini_rag.assistant import ANSWER_SCHEMA
from mini_rag.llm import LLMTimeoutError, OllamaClient

CHAT_REPLY = {
    "message": {"role": "assistant",
                "content": '{"answer": "x", "citations": [], "refused": true}'},
    "prompt_eval_count": 812,
    "eval_count": 40,
    "total_duration": 1_500_000_000,  # nanoseconds
}


class FakeOllama:
    """Records every request body per path and answers like the Ollama API."""

    def __init__(self, capabilities: list[str], chat_reply: dict = CHAT_REPLY) -> None:
        self.capabilities = capabilities
        self.chat_reply = chat_reply
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": self.capabilities})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json=self.chat_reply)
        return httpx.Response(404)

    def chat_bodies(self) -> list[dict]:
        return [body for path, body in self.requests if path == "/api/chat"]


def make_client(server: FakeOllama, num_ctx: int = 4096,
                num_predict: int | None = None) -> OllamaClient:
    return OllamaClient("http://ollama.test", "gen-model", "embed-model", num_ctx=num_ctx,
                        num_predict=num_predict, transport=httpx.MockTransport(server))


def test_chat_request_sends_configured_num_ctx_and_answer_schema():
    server = FakeOllama(capabilities=["completion"])

    make_client(server, num_ctx=6144).generate("sys", "prompt", ANSWER_SCHEMA)

    (body,) = server.chat_bodies()
    assert body["options"]["num_ctx"] == 6144
    assert body["format"] == ANSWER_SCHEMA
    assert body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "prompt"}]


@pytest.mark.parametrize(
    ("capabilities", "expected_think"),
    [
        pytest.param(["completion", "thinking"], False, id="thinking-model-gets-think-false"),
        # Sending `think` to a model without the capability is an Ollama error.
        pytest.param(["completion"], "absent", id="plain-model-gets-no-think-key"),
    ],
)
def test_thinking_disabled_only_for_models_that_support_it(capabilities, expected_think):
    server = FakeOllama(capabilities=capabilities)

    make_client(server).generate("sys", "prompt", ANSWER_SCHEMA)

    (body,) = server.chat_bodies()
    assert body.get("think", "absent") == expected_think


def test_capabilities_are_fetched_once_per_client():
    server = FakeOllama(capabilities=["thinking"])
    client = make_client(server)

    client.generate("sys", "p1", ANSWER_SCHEMA)
    client.generate("sys", "p2", ANSWER_SCHEMA)

    assert [path for path, _ in server.requests] == ["/api/show", "/api/chat", "/api/chat"]


def test_usage_numbers_are_mapped_from_ollama_reply():
    server = FakeOllama(capabilities=[])

    gen = make_client(server).generate("sys", "prompt", ANSWER_SCHEMA)

    assert gen.text == CHAT_REPLY["message"]["content"]
    assert (gen.prompt_tokens, gen.completion_tokens) == (812, 40)
    assert gen.duration_ms == pytest.approx(1500.0)


def test_output_cap_is_sent_only_when_configured():
    # The judge caps its reply length; the assistant must keep Ollama's default (no cap).
    capped, uncapped = FakeOllama(capabilities=[]), FakeOllama(capabilities=[])

    make_client(capped, num_predict=256).generate("sys", "prompt", ANSWER_SCHEMA)
    make_client(uncapped).generate("sys", "prompt", ANSWER_SCHEMA)

    assert capped.chat_bodies()[0]["options"]["num_predict"] == 256
    assert "num_predict" not in uncapped.chat_bodies()[0]["options"]


@pytest.mark.parametrize("done_reason, truncated", [("length", True), ("stop", False)])
def test_reply_cut_off_at_the_output_cap_is_flagged_truncated(done_reason, truncated):
    # A cut-off reply is usually still invalid JSON, but not always; the flag is what says why.
    server = FakeOllama(capabilities=[], chat_reply=CHAT_REPLY | {"done_reason": done_reason})

    assert make_client(server).generate("sys", "prompt").truncated is truncated


def test_missing_usage_numbers_become_none_not_zero():
    # Zero would read as "tiny prompt" and hide truncation risk; None means "unknown".
    server = FakeOllama(capabilities=[], chat_reply={"message": {"content": "{}"}})

    gen = make_client(server).generate("sys", "prompt", ANSWER_SCHEMA)

    assert (gen.prompt_tokens, gen.completion_tokens, gen.duration_ms) == (None, None, None)
    assert gen.truncated is False  # a missing done_reason is no evidence of a cut-off reply


def test_model_digest_found_for_tag_listed_with_latest_suffix():
    # `ollama list` shows "nomic-embed-text:latest" for a model configured as "nomic-embed-text";
    # a missed lookup would record None and lose which weights the run used.
    tags = {"models": [{"name": "qwen3:4b", "digest": "aaa"},
                       {"name": "nomic-embed-text:latest", "digest": "bbb"}]}
    client = OllamaClient("http://ollama", "qwen3:4b", "nomic-embed-text", num_ctx=4096,
                          transport=httpx.MockTransport(lambda r: httpx.Response(200, json=tags)))

    assert client.model_digests(["qwen3:4b", "nomic-embed-text", "missing"]) == {
        "qwen3:4b": "aaa", "nomic-embed-text": "bbb", "missing": None}


def test_unload_asks_ollama_to_drop_the_model_now():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append({"path": request.url.path, **json.loads(request.content)})
        return httpx.Response(200, json={})

    client = OllamaClient("http://ollama", "qwen3:4b", "e", num_ctx=4096,
                          transport=httpx.MockTransport(handler))
    client.unload("qwen3:4b")

    assert sent == [{"path": "/api/generate", "model": "qwen3:4b", "keep_alive": 0}]


def test_read_timeout_is_configurable_and_surfaces_as_llm_timeout():
    # The judge retries only LLMTimeoutError; an httpx exception leaking through would abort the
    # whole evaluate pass instead of becoming one counted judge_error.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": []})
        raise httpx.ReadTimeout("slow", request=request)

    client = OllamaClient("http://ollama", "qwen3:4b", "e", num_ctx=4096, read_timeout=45,
                          transport=httpx.MockTransport(handler))

    with pytest.raises(LLMTimeoutError, match="within 45s"):
        client.generate("sys", "prompt")
    assert client._http.timeout.read == 45  # the value reaches httpx, not only the message
