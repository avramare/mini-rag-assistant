"""Two-pass experiment runner.

Pass 1, generate: ask every dataset item `--repeats` times, save each answer to results/<name>.json.
Pass 2, evaluate: grade the saved answers (code evaluators + LLM judge) and push scores to Langfuse.

Two passes so only one large model is in memory at a time (the gen model is unloaded after pass 1,
everything but the judge before pass 2), and so the judge can be re-run without regenerating.

Usage:
    uv run python -m evals.run_experiment generate --name <run> --repeats 5 [--dataset P] [--resume]
    uv run python -m evals.run_experiment evaluate --name <run> [--no-judge]
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from evals.dataset import (
    Dataset,
    DatasetError,
    check_expected_docs,
    check_facts_in_corpus,
    load_dataset,
)
from evals.evaluators import Corpus, answer_passed, run_code_evaluators
from evals.judge import JUDGE_PROMPT_SHA256, PASS_SCORE, judge_faithfulness
from evals.publish import LangfusePublisher, NoopPublisher, Publisher, publish_scores
from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    Evaluation,
    RunResults,
    load,
    results_path,
    save,
    stable_config,
)
from mini_rag.assistant import Assistant
from mini_rag.documents import Access
from mini_rag.llm import LLMClient
from mini_rag.retrieval import corpus_hash
from mini_rag.users import User

DEFAULT_DATASET = Path("evals/dataset.jsonl")

Log = Callable[[str], None]


class RunError(RuntimeError):
    """The run cannot proceed safely (config mismatch, stale corpus, ...)."""


def run_names(name: str, repeats: int) -> list[str]:
    # One Langfuse dataset run per repeat: the UI compare view of r1 vs r2 shows the noise directly.
    return [f"{name}-r{r}" for r in range(1, repeats + 1)]


def build_config(name: str, dataset: Dataset, assistant: Assistant, repeats: int,
                 model_digests: dict[str, str | None], git: dict[str, Any]) -> dict[str, Any]:
    assistant_config = assistant.run_config()
    assistant_config.pop("today")  # per item: dataset as_of, or the item's override
    return {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **git,
        "repeats": repeats,
        "as_of": dataset.as_of.isoformat(),
        "dataset": {"name": dataset.name, "path": dataset.path.as_posix(),
                    "sha256": dataset.sha256, "generation_key": dataset.generation_key(),
                    "langfuse_name": dataset.langfuse_name, "items": len(dataset.items)},
        "assistant": assistant_config,
        "model_digests": model_digests,
        "langfuse_runs": run_names(name, repeats),
    }


def generate(dataset: Dataset, assistant_for: Callable[[date], Assistant],
             users: dict[str, User], access_by_id: dict[str, Access], config: dict[str, Any],
             path: Path, publisher: Publisher, *, resume: bool = False,
             log: Log = print) -> RunResults:
    if path.exists():
        if not resume:
            raise RunError(f"{path} exists; use --resume to continue it or pick a new --name")
        run = load(path)
        if stable_config(run.config) != stable_config(config):
            raise RunError(f"{path} was generated with a different config; not resuming")
    else:
        run = RunResults(config=config)

    publisher.upload_dataset(dataset)
    repeats = config["repeats"]
    total, done = len(dataset.items) * repeats, run.done()
    for repeat, run_name in enumerate(run_names(config["name"], repeats), 1):
        for item in dataset.items:
            if (item.id, repeat) in done:
                continue
            as_of = dataset.as_of_for(item)
            t0 = time.perf_counter()
            result = assistant_for(as_of).answer(item.question, users[item.user],
                                                 dataset_item_id=item.id)
            record = AnswerRecord.from_result(
                result, item_id=item.id, repeat=repeat, category=item.category, user=item.user,
                question=item.question, as_of=as_of, access_by_id=access_by_id)
            run.answers.append(record)
            save(run, path)  # after every answer: a crash loses at most the current one
            if record.trace_id:
                publisher.link(dataset, item, run_name, record.trace_id, config["assistant"])
            status = record.error or record.refusal_reason or "answered"
            log(f"[{len(run.answers)}/{total}] {item.id} r{repeat} {status} "
                f"({time.perf_counter() - t0:.1f}s)")
    return run


def evaluate(path: Path, dataset: Dataset, corpus: Corpus, judge: LLMClient | None,
             judge_info: dict[str, Any] | None, publisher: Publisher,
             log: Log = print) -> RunResults:
    run = load(path)
    expected = run.config["dataset"]["generation_key"]
    if dataset.generation_key() != expected:
        raise RunError("dataset questions/users/dates changed since generation; regenerate")
    assistant_config = run.config["assistant"]
    if corpus_hash(list(corpus.docs.values()),
                   assistant_config["embed_model"]) != assistant_config["corpus_sha256"]:
        raise RunError("corpus changed since generation; the judge would see other documents")
    missing = len(dataset.items) * run.config["repeats"] - len(run.answers)
    if missing:
        raise RunError(f"{missing} answers missing; finish with `generate --resume` first")

    items = {i.id: i for i in dataset.items}
    evaluations = []
    for n, record in enumerate(run.answers, 1):
        item = items[record.item_id]
        results = run_code_evaluators(item, record, corpus)
        if judge is not None:
            results.append(judge_faithfulness(record, corpus.docs, judge))
        evaluations.append(AnswerEvaluation(item_id=record.item_id, repeat=record.repeat,
                                            results=results, passed=answer_passed(results)))
        if judge is not None:
            log(f"[{n}/{len(run.answers)}] {record.item_id} r{record.repeat} "
                f"{'pass' if evaluations[-1].passed else 'FAIL'}")

    run.evaluation = Evaluation(evaluated_at=datetime.now(UTC), dataset_sha256=dataset.sha256,
                                judge=judge_info, answers=evaluations)
    save(run, path)  # local results first: a Langfuse outage must not lose the grading
    for record, ev in zip(run.answers, evaluations, strict=True):
        publish_scores(publisher, record, ev.results, ev.passed)
    return run


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def main() -> int:
    from mini_rag.config import Settings
    from mini_rag.documents import load_documents
    from mini_rag.llm import OllamaClient
    from mini_rag.retrieval import Retriever
    from mini_rag.tracing import LangfuseTracer, NoopTracer, make_langfuse
    from mini_rag.users import load_users

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--name", required=True)
    gen.add_argument("--repeats", type=int, default=1)
    gen.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    gen.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", required=True)
    ev.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    s = Settings()
    users = load_users(s.users_file)
    docs = load_documents(s.docs_dir)
    langfuse = make_langfuse(s)
    publisher: Publisher = NoopPublisher() if langfuse is None else LangfusePublisher(langfuse)
    path = results_path(args.name)

    try:
        if args.command == "generate":
            dataset = load_dataset(args.dataset, set(users))
            check_facts_in_corpus(dataset, docs)
            check_expected_docs(dataset, docs, users)
            client = OllamaClient(s.ollama_host, s.gen_model, s.embed_model, num_ctx=s.num_ctx)
            retriever = Retriever(docs, client, cache_dir=s.cache_dir)
            tracer = NoopTracer() if langfuse is None else LangfuseTracer(langfuse)
            assistants: dict[date, Assistant] = {}

            def assistant_for(as_of: date) -> Assistant:
                if as_of not in assistants:
                    assistants[as_of] = Assistant(
                        retriever, client, today=lambda: as_of, num_ctx=s.num_ctx,
                        max_prompt_ctx_share=s.max_prompt_ctx_share, tracer=tracer)
                return assistants[as_of]

            config = build_config(args.name, dataset, assistant_for(dataset.as_of), args.repeats,
                                  client.model_digests([s.gen_model, s.embed_model]),
                                  _git_info())
            generate(dataset, assistant_for, users, {d.id: d.access for d in docs}, config,
                     path, publisher, resume=args.resume)
            tracer.flush()
            client.unload(s.gen_model)
            print(f"unloaded {s.gen_model}; loaded now: {client.loaded_models()}")
        else:
            run = load(path)
            dataset = load_dataset(Path(run.config["dataset"]["path"]), set(users))
            check_facts_in_corpus(dataset, docs)
            check_expected_docs(dataset, docs, users)
            judge = judge_info = None
            if not args.no_judge:
                judge = OllamaClient(s.ollama_host, s.judge_model, s.embed_model,
                                     num_ctx=s.num_ctx)
                for model in judge.loaded_models():
                    if model not in {s.judge_model, f"{s.judge_model}:latest"}:
                        judge.unload(model)  # only the judge stays in memory
                print(f"loaded before judging: {judge.loaded_models()}")
                judge_info = {"model": s.judge_model,
                              "digest": judge.model_digests([s.judge_model])[s.judge_model],
                              "prompt_sha256": JUDGE_PROMPT_SHA256, "pass_score": PASS_SCORE}
            evaluate(path, dataset, Corpus({d.id: d for d in docs}, users), judge, judge_info,
                     publisher)
            print(f"evaluated {path}; report: uv run python -m evals.stats report {path}")
    except (RunError, DatasetError) as exc:
        print(f"[fail] {exc}")
        return 1
    finally:
        publisher.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
