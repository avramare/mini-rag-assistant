"""The results file `results/<run>.json`: config, saved answers (pass 1) and evaluation (pass 2).

Local only (gitignored): answers to lead questions contain restricted text, like the docs.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from mini_rag.assistant import Answer, AnswerResult
from mini_rag.documents import Access

RESULTS_DIR = Path("results")

# Config keys that may differ when resuming a run (they describe the process, not the experiment).
VOLATILE_CONFIG_KEYS = frozenset({"created_at", "git_commit", "git_dirty"})
# May change on resume, but only upward: the runner refuses fewer repeats and records the growth
# in `repeats_history`. More repeats of the same config are more of the same experiment.
GROWABLE_CONFIG_KEYS = frozenset({"repeats", "langfuse_runs", "repeats_history"})


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
    # Ollama's split of each attempt's duration; empty in files written before it was recorded.
    prompt_eval_ms: list[float | None] = []
    eval_ms: list[float | None] = []
    load_ms: list[float | None] = []
    truncation_risk: bool
    trace_id: str | None
    # When the answer was generated; repeats added by a later `generate --resume` show their own
    # dates. None in files written before it was recorded.
    generated_at: datetime | None = None

    @property
    def masked(self) -> bool:
        """Same rule as the tracer: any restricted doc in the context."""
        return any(d.access == Access.RESTRICTED for d in self.retrieved)

    @classmethod
    def from_result(cls, result: AnswerResult, *, item_id: str, repeat: int, category: str,
                    user: str, question: str, as_of: date,
                    access_by_id: dict[str, Access],
                    generated_at: datetime | None = None) -> "AnswerRecord":
        return cls(
            item_id=item_id, repeat=repeat, category=category, user=user, question=question,
            as_of=as_of, answer=result.answer, raw_outputs=result.raw_outputs,
            retrieved=[RetrievedDoc(id=i, access=access_by_id[i]) for i in result.retrieved_ids],
            error=result.error, refusal_reason=result.refusal_reason, attempts=result.attempts,
            invalid_outputs=result.invalid_outputs, prompt_tokens=result.prompt_tokens,
            durations_ms=result.durations_ms, prompt_eval_ms=result.prompt_eval_ms,
            eval_ms=result.eval_ms, load_ms=result.load_ms, truncation_risk=result.truncation_risk,
            trace_id=result.trace_id, generated_at=generated_at,
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
    # Failed attempts before this result ("timeout", "invalid_output"); the report counts them
    # even when the retry succeeded.
    attempt_errors: list[str] = []
    # Judge cost: wall clock over all attempts (incl. timeouts), and the tokens of the reply that
    # was used. None for code evaluators and for results saved before these were recorded.
    duration_ms: float | None = None
    output_tokens: int | None = None
    # Ollama's split of the used attempt's time: reading the prompt, writing the reply, loading
    # the model.
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None
    load_ms: float | None = None


class AnswerEvaluation(BaseModel):
    item_id: str
    repeat: int
    results: list[EvalResult]
    passed: bool


class Evaluation(BaseModel):
    evaluated_at: datetime  # when it started
    # None while unfinished: saved after every answer, continued with `evaluate --resume`.
    # No default, so new code must set it; only old files get it filled in below.
    finished_at: datetime | None
    dataset_sha256: str  # the facts used for grading; compare refuses different values
    judge: dict[str, Any] | None  # model, digest, prompt sha256; None with --no-judge
    answers: list[AnswerEvaluation]
    # evaluator code hash and git state while grading (evals.instrument); None before Phase 5.
    instrument: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _saved_whole(cls, data: Any) -> Any:
        """Files written before incremental saves only ever held finished evaluations."""
        if isinstance(data, dict) and "finished_at" not in data:
            data = {**data, "finished_at": data.get("evaluated_at")}
        return data


class SampledJudgement(BaseModel):
    item_id: str
    repeat: int
    result: EvalResult


class JudgeSample(BaseModel):
    """One extra pass of the same judge over one repeat's saved answers (Phase 4 judge noise).
    Kept apart from `evaluations`: a sample never replaces the grades reports and compare use."""

    sample: int  # 1..K; the evaluation itself is the sample the report calls 0
    repeat: int
    started_at: datetime
    finished_at: datetime | None
    judge: dict[str, Any]
    answers: list[SampledJudgement]


class PublishFailure(BaseModel):
    """A Langfuse call that failed; the run went on (local results are the source of truth).
    `run_experiment publish` re-sends what can be re-sent."""

    kind: Literal["dataset", "link", "score"]
    error: str
    at: datetime
    item_id: str | None = None
    repeat: int | None = None
    run_name: str | None = None  # the Langfuse dataset run of a link
    trace_id: str | None = None
    score: str | None = None  # score name


NO_JUDGE = "no-judge"


def evaluation_key(judge: dict[str, Any] | None) -> str:
    """Evaluations are stored per judge version: re-running the same prompt overwrites its own
    evaluation, a new prompt is stored next to the old one (and so are its Langfuse scores).
    An output cap is part of the version (a cut-off reply grades differently): `<sha>-np<cap>`."""
    if not judge:
        return NO_JUDGE
    cap = judge.get("num_predict")
    return judge["prompt_sha256"][:12] + (f"-np{cap}" if cap else "")


class RunResults(BaseModel):
    config: dict[str, Any]
    answers: list[AnswerRecord] = []
    # evaluation_key -> evaluation, oldest first: the runner moves the one it writes to the end.
    # Order, not `evaluated_at`: two evaluations can share a timestamp (Windows clock ticks ~15 ms).
    evaluations: dict[str, Evaluation] = {}
    # evaluation_key -> re-samples of that judge (`run_experiment rejudge`); local only.
    judge_samples: dict[str, list[JudgeSample]] = {}
    publish_failures: list[PublishFailure] = []

    @model_validator(mode="before")
    @classmethod
    def _one_evaluation_to_keyed(cls, data: Any) -> Any:
        """Files written before evaluations were keyed hold a single `evaluation`."""
        if isinstance(data, dict) and "evaluation" in data:
            data = dict(data)
            old = data.pop("evaluation")
            if old is not None:
                judge = old.judge if isinstance(old, Evaluation) else old.get("judge")
                data["evaluations"] = {evaluation_key(judge): old, **data.get("evaluations", {})}
        return data

    def done(self) -> set[tuple[str, int]]:
        return {(a.item_id, a.repeat) for a in self.answers}

    def evaluation(self, key: str | None = None) -> tuple[str, Evaluation]:
        """The evaluation stored under `key` (a unique prefix is enough); the latest without one."""
        if not self.evaluations:
            raise ValueError("run has no evaluation yet; run `evaluate` first")
        if key is None:
            latest = next(reversed(self.evaluations))
            return latest, self.evaluations[latest]
        matches = [k for k in self.evaluations if k.startswith(key)]
        if len(matches) != 1:
            raise ValueError(f"judge version {key!r} matches {matches or 'nothing'}; "
                             f"stored: {sorted(self.evaluations)}")
        return matches[0], self.evaluations[matches[0]]


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
    return {k: v for k, v in config.items()
            if k not in VOLATILE_CONFIG_KEYS | GROWABLE_CONFIG_KEYS}
