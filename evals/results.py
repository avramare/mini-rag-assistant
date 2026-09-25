"""The results file `results/<run>.json`: config, saved answers (pass 1) and evaluation (pass 2).

Local only (gitignored): answers to lead questions contain restricted text, like the docs.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from mini_rag.assistant import Answer, AnswerResult
from mini_rag.documents import Access

RESULTS_DIR = Path("results")

# Config keys that may differ when resuming a run (they describe the process, not the experiment).
VOLATILE_CONFIG_KEYS = frozenset({"created_at", "git_commit", "git_dirty"})


class RetrievedDoc(BaseModel):
    id: str
    access: Access


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    repeat: int
    category: str
    user: str
    question: str
    as_of: date
    answer: Answer | None  # what the user was shown
    raw_outputs: list[str]  # what the model said, per attempt; evaluators check it independently
    retrieved: list[RetrievedDoc]
    error: str | None
    refusal_reason: str | None
    attempts: int
    invalid_outputs: int
    prompt_tokens: list[int | None]
    durations_ms: list[float | None]
    truncation_risk: bool
    trace_id: str | None

    @property
    def masked(self) -> bool:
        """Same rule as the tracer: any restricted doc in the context."""
        return any(d.access == Access.RESTRICTED for d in self.retrieved)

    @classmethod
    def from_result(cls, result: AnswerResult, *, item_id: str, repeat: int, category: str,
                    user: str, question: str, as_of: date,
                    access_by_id: dict[str, Access]) -> "AnswerRecord":
        return cls(
            item_id=item_id, repeat=repeat, category=category, user=user, question=question,
            as_of=as_of, answer=result.answer, raw_outputs=result.raw_outputs,
            retrieved=[RetrievedDoc(id=i, access=access_by_id[i]) for i in result.retrieved_ids],
            error=result.error, refusal_reason=result.refusal_reason, attempts=result.attempts,
            invalid_outputs=result.invalid_outputs, prompt_tokens=result.prompt_tokens,
            durations_ms=result.durations_ms, truncation_risk=result.truncation_risk,
            trace_id=result.trace_id,
        )


class EvalResult(BaseModel):
    name: str
    applicable: bool
    passed: bool | None = None  # None when not applicable
    value: float | None = None  # numeric score where one exists (recall share, judge 1-5)
    # False for diagnostics (retrieval_recall): reported, but not part of the answer's pass/fail.
    gating: bool = True
    # Sent to Langfuse as the score comment. Code evaluators never put fact text here (facts of
    # restricted items are restricted); they refer to facts by position ("#2"). The judge's reason
    # can quote the answer, so publish_scores masks it like the trace.
    detail: str = ""


class AnswerEvaluation(BaseModel):
    item_id: str
    repeat: int
    results: list[EvalResult]
    passed: bool


class Evaluation(BaseModel):
    evaluated_at: datetime
    dataset_sha256: str  # the facts used for grading; compare refuses different values
    judge: dict[str, Any] | None  # model, digest, prompt sha256; None with --no-judge
    answers: list[AnswerEvaluation]


class RunResults(BaseModel):
    config: dict[str, Any]
    answers: list[AnswerRecord] = []
    evaluation: Evaluation | None = None

    def done(self) -> set[tuple[str, int]]:
        return {(a.item_id, a.repeat) for a in self.answers}


def results_path(name: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"{name}.json"


def save(run: RunResults, path: Path) -> None:
    """Atomic: write a temp file, then rename. A crash never leaves a half-written results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(run.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(path)


def load(path: Path) -> RunResults:
    return RunResults.model_validate_json(path.read_text(encoding="utf-8"))


def stable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in VOLATILE_CONFIG_KEYS}
