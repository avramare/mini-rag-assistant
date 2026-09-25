"""Langfuse tracing for `Assistant.answer`, with masking for restricted context.

One trace per call: a root `answer` span, a `retrieval` span and one `generation` per attempt.

Masking rule: if ANY retrieved doc is restricted, the question, the prompt and every model output
are replaced by `[masked: ...]` before they reach the Langfuse SDK. Doc ids, access levels, token
counts, durations and flags stay visible. The decision is made here, per call, because Langfuse's
own `mask` hook sees one value at a time and cannot know what this call retrieved.
See DECISIONS.md #18.

Without Langfuse keys, `make_tracer` returns `NoopTracer`: nothing is imported or sent.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Protocol

from mini_rag.config import Settings
from mini_rag.documents import Access
from mini_rag.llm import Generation
from mini_rag.retrieval import Hit
from mini_rag.users import User

if TYPE_CHECKING:  # only for type hints; the real import is lazy (langfuse takes ~1 s to import)
    from langfuse import Langfuse

    from mini_rag.assistant import AnswerResult


def is_restricted(hits: list[Hit]) -> bool:
    """Mask decision for one call: based on what was RETRIEVED (it is all in the prompt),
    not on what the model cited. A model can leak a figure from a doc it does not cite."""
    return any(h.doc.access == Access.RESTRICTED for h in hits)


def mask_text(text: str, masked: bool) -> str:
    # Length only. No hash: a short answer like "4.2 million euros" can be guessed and confirmed
    # by hashing candidates. No prefix: the first words of an answer are often the secret.
    return f"[masked: restricted context, {len(text)} chars]" if masked else text


class AnswerTrace(Protocol):
    """What `Assistant.answer` reports during one call. Each context manager yields a `record`
    callback; the span is timed from entering the block to leaving it."""

    trace_id: str | None

    def retrieval(self, question: str) -> AbstractContextManager[Callable[[list[Hit]], None]]: ...

    def generation(self, attempt: int, system: str,
                   prompt: str) -> AbstractContextManager[Callable[[Generation], None]]: ...

    def finish(self, result: "AnswerResult") -> None: ...


class Tracer(Protocol):
    def answer_trace(self, question: str, user: User, run_config: dict, *,
                     dataset_item_id: str | None = None) -> AbstractContextManager[AnswerTrace]: ...

    def flush(self) -> None: ...


def _ignore(_: Any) -> None:
    return None


class _NoopAnswerTrace:
    trace_id = None

    @contextmanager
    def retrieval(self, question: str) -> Iterator[Callable[[list[Hit]], None]]:
        yield _ignore

    @contextmanager
    def generation(self, attempt: int, system: str,
                   prompt: str) -> Iterator[Callable[[Generation], None]]:
        yield _ignore

    def finish(self, result: "AnswerResult") -> None:
        return None


class NoopTracer:
    """Used when Langfuse keys are missing, and in every test that does not test tracing."""

    @contextmanager
    def answer_trace(self, question: str, user: User, run_config: dict, *,
                     dataset_item_id: str | None = None) -> Iterator[AnswerTrace]:
        yield _NoopAnswerTrace()

    def flush(self) -> None:
        return None


class _LangfuseAnswerTrace:
    def __init__(self, client: "Langfuse", root: Any, question: str, run_config: dict,
                 dataset_item_id: str | None) -> None:
        self._client = client
        self._root = root
        self.trace_id: str | None = root.trace_id
        self._question = question
        self._run_config = run_config
        self._model = run_config["gen_model"]
        self._dataset_item_id = dataset_item_id
        self._hits: list[Hit] = []
        # Fail closed: until retrieval has told us otherwise, treat the call as restricted.
        self._masked = True

    def _retrieved(self) -> list[dict]:
        return [{"id": h.doc.id, "access": h.doc.access.value, "score": round(h.score, 4)}
                for h in self._hits]

    @contextmanager
    def retrieval(self, question: str) -> Iterator[Callable[[list[Hit]], None]]:
        with self._client.start_as_current_observation(name="retrieval",
                                                       as_type="retriever") as span:
            def record(hits: list[Hit]) -> None:
                self._hits = list(hits)
                self._masked = is_restricted(hits)
                span.update(input=mask_text(question, self._masked), output=self._retrieved())

            yield record

    @contextmanager
    def generation(self, attempt: int, system: str,
                   prompt: str) -> Iterator[Callable[[Generation], None]]:
        with self._client.start_as_current_observation(
            name="generation", as_type="generation", model=self._model,
            input={"system": system, "prompt": mask_text(prompt, self._masked)},
            metadata={"attempt": attempt, "masked": self._masked},
        ) as span:
            def record(gen: Generation) -> None:
                usage = {"input": gen.prompt_tokens, "output": gen.completion_tokens}
                span.update(
                    output=mask_text(gen.text, self._masked),
                    usage_details={k: v for k, v in usage.items() if v is not None},
                    metadata={"duration_ms": gen.duration_ms},
                )

            yield record

    def finish(self, result: "AnswerResult") -> None:
        answer = result.answer
        self._root.update(
            input=mask_text(self._question, self._masked),
            output=None if answer is None else {
                "answer": mask_text(answer.answer, self._masked),
                "citations": answer.citations,
                "refused": answer.refused,
            },
            metadata={
                # Run config is never masked, so masked and unmasked runs stay comparable.
                **self._run_config,
                "dataset_item_id": self._dataset_item_id,
                "masked": self._masked,
                "retrieved": self._retrieved(),
                "attempts": result.attempts,
                "invalid_outputs": result.invalid_outputs,
                "prompt_tokens": result.prompt_tokens,
                "durations_ms": result.durations_ms,
                "error": result.error,
                "refusal_reason": result.refusal_reason,
                "truncation_risk": result.truncation_risk,
            },
        )


class LangfuseTracer:
    def __init__(self, client: "Langfuse") -> None:
        self._client = client

    @contextmanager
    def answer_trace(self, question: str, user: User, run_config: dict, *,
                     dataset_item_id: str | None = None) -> Iterator[AnswerTrace]:
        from langfuse import propagate_attributes

        with self._client.start_as_current_observation(name="answer", as_type="span") as root:
            with propagate_attributes(user_id=user.name, trace_name="answer"):
                yield _LangfuseAnswerTrace(self._client, root, question, run_config,
                                           dataset_item_id)

    def flush(self) -> None:
        self._client.flush()


def make_langfuse(settings: Settings, **client_kwargs: Any) -> "Langfuse | None":
    """A Langfuse client, or None when either key is missing. `client_kwargs` go to
    `Langfuse(...)`; tests use them to plug in an in-memory exporter and a fake HTTP client."""
    public_key = settings.langfuse_public_key.get_secret_value()
    secret_key = settings.langfuse_secret_key.get_secret_value()
    if not (public_key and secret_key):
        return None

    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key,
                    base_url=settings.langfuse_host, **client_kwargs)


def make_tracer(settings: Settings, **client_kwargs: Any) -> Tracer:
    client = make_langfuse(settings, **client_kwargs)
    return NoopTracer() if client is None else LangfuseTracer(client)
