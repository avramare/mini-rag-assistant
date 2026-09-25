"""Everything the eval pipeline sends to Langfuse: dataset items, run links and scores.

We do NOT use the SDK's `run_experiment`: it writes each item's input and expected output onto
its own spans unmasked, bypassing the trace masking (DECISIONS 18), and it runs task and evaluators
in one pass. Instead each answer's own (masked) trace is linked to a dataset run, and scores are
added later.

What never leaves the process: expected/forbidden facts (restricted for restricted items), and any
evaluator comment for an answer whose context held a restricted doc.
"""

import hashlib
from typing import TYPE_CHECKING, Any, Protocol

from evals.dataset import Dataset, DatasetItem
from evals.results import AnswerRecord, EvalResult
from mini_rag.tracing import mask_text

if TYPE_CHECKING:
    from langfuse import Langfuse


class Publisher(Protocol):
    def upload_dataset(self, dataset: Dataset) -> None: ...

    def link(self, dataset: Dataset, item: DatasetItem, run_name: str, trace_id: str,
             metadata: dict[str, Any]) -> None: ...

    def score(self, trace_id: str, name: str, value: float, comment: str, score_id: str,
              metadata: dict[str, str] | None) -> None: ...

    def flush(self) -> None: ...


class NoopPublisher:
    """Used without Langfuse keys and in tests that do not test publishing."""

    def upload_dataset(self, dataset: Dataset) -> None:
        return None

    def link(self, dataset: Dataset, item: DatasetItem, run_name: str, trace_id: str,
             metadata: dict[str, Any]) -> None:
        return None

    def score(self, trace_id: str, name: str, value: float, comment: str, score_id: str,
              metadata: dict[str, str] | None) -> None:
        return None

    def flush(self) -> None:
        return None


def langfuse_item_id(dataset: Dataset, item: DatasetItem) -> str:
    # Langfuse item ids are unique across ALL datasets of a project, so prefix the dataset name.
    return f"{dataset.langfuse_name}:{item.id}"


def item_upload(dataset: Dataset, item: DatasetItem) -> dict[str, Any]:
    """The only item fields that go to Langfuse. Facts stay local (DECISIONS 19)."""
    return {
        "input": {"question": item.question, "user": item.user},
        "metadata": {"item_id": item.id, "category": item.category,
                     "should_refuse": item.should_refuse,
                     "as_of": dataset.as_of_for(item).isoformat()},
    }


class LangfusePublisher:
    def __init__(self, client: "Langfuse") -> None:
        self._client = client

    def upload_dataset(self, dataset: Dataset) -> None:
        # Both calls upsert: re-uploading an edited dataset updates the items in place.
        self._client.create_dataset(name=dataset.langfuse_name,
                                    metadata={"as_of": dataset.as_of.isoformat(),
                                              "sha256": dataset.sha256})
        for item in dataset.items:
            self._client.create_dataset_item(dataset_name=dataset.langfuse_name,
                                             id=langfuse_item_id(dataset, item),
                                             **item_upload(dataset, item))

    def link(self, dataset: Dataset, item: DatasetItem, run_name: str, trace_id: str,
             metadata: dict[str, Any]) -> None:
        self._client.api.dataset_run_items.create(
            run_name=run_name, dataset_item_id=langfuse_item_id(dataset, item),
            trace_id=trace_id, metadata=metadata,
        )

    def score(self, trace_id: str, name: str, value: float, comment: str, score_id: str,
              metadata: dict[str, str] | None) -> None:
        self._client.create_score(trace_id=trace_id, name=name, value=value,
                                  data_type="NUMERIC", comment=comment, score_id=score_id,
                                  metadata=metadata)

    def flush(self) -> None:
        self._client.flush()


# Scores whose value depends on the judge; the others depend only on the answer and the facts.
JUDGE_DEPENDENT_SCORES = frozenset({"judge_faithfulness", "item_pass"})


def score_id(trace_id: str, name: str, judge_key: str) -> str:
    """Deterministic, so re-publishing overwrites a score instead of adding one. Judge-dependent
    scores also key on the judge version: the same prompt overwrites its own scores, a new prompt
    adds its scores next to the old ones. Code evaluator ids stay as before (trace + name)."""
    parts = [trace_id, name, judge_key] if name in JUDGE_DEPENDENT_SCORES else [trace_id, name]
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:32]


def publish_scores(publisher: Publisher, record: AnswerRecord, results: list[EvalResult],
                   passed: bool, judge_key: str) -> None:
    if record.trace_id is None:  # generated without tracing: nothing to attach scores to
        return
    trace_id = record.trace_id

    def send(name: str, value: float, comment: str) -> None:
        metadata = {"judge": judge_key} if name in JUDGE_DEPENDENT_SCORES else None
        publisher.score(trace_id, name, value, comment, score_id(trace_id, name, judge_key),
                        metadata)

    for r in results:
        if r.applicable:
            value = r.value if r.value is not None else float(bool(r.passed))
            # The judge's reason can quote the answer, so it is masked exactly like the trace.
            send(r.name, value, mask_text(r.detail, record.masked))
    send("item_pass", float(passed), "")
