"""Two-pass experiment runner.

Pass 1, generate: ask every dataset item `--repeats` times, save each answer to results/<name>.json.
Pass 2, evaluate: grade the saved answers (code evaluators + LLM judge) and push scores to Langfuse.

Two passes so only one large model is in memory at a time (the gen model is unloaded after pass 1,
everything but the judge before pass 2), and so the judge can be re-run without regenerating.

Usage:
    uv run python -m evals.run_experiment generate --name <run> --repeats 5 [--dataset P] [--resume]
        [--items id1,id2] [--category c1,c2]    # filtered dev run; not comparable with full runs
    uv run python -m evals.run_experiment evaluate --name <run> [--no-judge] [--resume]
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
    filter_dataset,
    load_dataset,
)
from evals.evaluators import Corpus, answer_passed, run_code_evaluators
from evals.judge import JUDGE_NUM_PREDICT, JUDGE_PROMPT_SHA256, PASS_SCORE, judge_faithfulness
from evals.publish import LangfusePublisher, NoopPublisher, Publisher, publish_scores
from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    Evaluation,
    RunResults,
    evaluation_key,
    load,
    results_path,
    save,
    stable_config,
)
from mini_rag.assistant import Assistant
from mini_rag.documents import Access
from mini_rag.llm import LLMClient, OllamaClient
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
                 model_digests: dict[str, str | None], git: dict[str, Any],
                 run_filter: dict[str, list[str] | None] | None = None) -> dict[str, Any]:
    """`dataset` is already filtered by `run_filter`; the filter is kept so that evaluate can
    select the same items and compare can refuse a filtered run against a full one."""
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
        "filter": run_filter,
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
                publisher.link(dataset, item, run_name, record.trace_id, link_metadata(config))
            status = record.error or record.refusal_reason or "answered"
            log(f"[{len(run.answers)}/{total}] {item.id} r{repeat} {status} "
                f"({time.perf_counter() - t0:.1f}s)")
    return run


def link_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Dataset-run metadata in Langfuse: a filtered dev run must be recognisable there too."""
    if config.get("filter") is None:
        return config["assistant"]
    return {**config["assistant"], "filter": config["filter"]}


def evaluate(path: Path, dataset: Dataset, corpus: Corpus, judge: LLMClient | None,
             judge_info: dict[str, Any] | None, publisher: Publisher, *, resume: bool = False,
             log: Log = print) -> RunResults:
    run = load(path)
    dataset = filter_dataset(dataset, run.config.get("filter"))  # the items this run generated
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

    key = evaluation_key(judge_info)  # same judge prompt: overwrite; new prompt: add alongside
    evaluation = start_or_resume(run, key, dataset.sha256, judge_info, resume, log)
    items = {i.id: i for i in dataset.items}
    done = {(ev.item_id, ev.repeat) for ev in evaluation.answers}
    for n, record in enumerate(run.answers, 1):
        if (record.item_id, record.repeat) in done:
            continue
        item = items[record.item_id]
        results = run_code_evaluators(item, record, corpus)
        if judge is not None:
            results.append(judge_faithfulness(record, corpus.docs, judge))
        evaluation.answers.append(AnswerEvaluation(
            item_id=record.item_id, repeat=record.repeat, results=results,
            passed=answer_passed(results)))
        save(run, path)  # after every answer: a crash loses at most the current one
        if judge is not None:
            log(f"[{n}/{len(run.answers)}] {record.item_id} r{record.repeat} "
                f"{'pass' if evaluation.answers[-1].passed else 'FAIL'}")

    evaluation.finished_at = datetime.now(UTC)
    save(run, path)  # local results first: a Langfuse outage must not lose the grading
    # All answers, also those graded before a resume: score ids are deterministic, so a score
    # that was already sent is overwritten with the same value.
    by_answer = {(ev.item_id, ev.repeat): ev for ev in evaluation.answers}
    for record in run.answers:
        ev = by_answer[(record.item_id, record.repeat)]
        publish_scores(publisher, record, ev.results, ev.passed, key)
    return run


def start_or_resume(run: RunResults, key: str, dataset_sha256: str,
                    judge_info: dict[str, Any] | None, resume: bool, log: Log) -> Evaluation:
    """Resumable = an unfinished evaluation by the same judge (model digest too) against the same
    facts; only then can its grades be mixed with new ones. Without --resume a resumable one is
    not thrown away by accident; one that can no longer be resumed is replaced."""
    old = run.evaluations.pop(key, None)  # re-inserted below: the one written last is latest
    unfinished = old is not None and old.finished_at is None
    resumable = unfinished and (old.dataset_sha256, old.judge) == (dataset_sha256, judge_info)
    if resume:
        if not resumable:
            raise RunError(f"no unfinished evaluation by judge {key} with these facts and this "
                           "judge model to resume; run evaluate without --resume")
        log(f"resuming judge {key}: {len(old.answers)} of {len(run.answers)} answers done")
        run.evaluations[key] = old
        return old
    if resumable:
        raise RunError(f"judge {key} has an unfinished evaluation ({len(old.answers)} of "
                       f"{len(run.answers)} answers); continue it with --resume")
    if unfinished:
        log(f"replacing unfinished evaluation by judge {key}: other facts or judge model")
    run.evaluations[key] = Evaluation(evaluated_at=datetime.now(UTC), finished_at=None,
                                      dataset_sha256=dataset_sha256, judge=judge_info,
                                      answers=[])
    return run.evaluations[key]


def judge_client(host: str, model: str, embed_model: str, num_ctx: int,
                 read_timeout: float) -> OllamaClient:
    return OllamaClient(host, model, embed_model, num_ctx=num_ctx, read_timeout=read_timeout,
                        num_predict=JUDGE_NUM_PREDICT)


def describe_judge(client: OllamaClient, digest: str | None) -> dict[str, Any]:
    """Recorded with the evaluation and part of its key. Read from the client that will judge,
    so the recorded cap cannot differ from the one actually sent."""
    return {"model": client.model, "digest": digest, "prompt_sha256": JUDGE_PROMPT_SHA256,
            "pass_score": PASS_SCORE, "num_predict": client.num_predict}


def _csv(value: str) -> list[str]:
    return sorted({v.strip() for v in value.split(",") if v.strip()})


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
    gen.add_argument("--items", type=_csv, help="only these item ids (comma separated)")
    gen.add_argument("--category", type=_csv, help="only these categories (comma separated)")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", required=True)
    ev.add_argument("--no-judge", action="store_true")
    ev.add_argument("--resume", action="store_true")
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
            run_filter = ({"items": args.items, "categories": args.category}
                          if args.items or args.category else None)
            dataset = filter_dataset(dataset, run_filter)
            client = OllamaClient(s.ollama_host, s.gen_model, s.embed_model, num_ctx=s.num_ctx,
                                  read_timeout=s.ollama_read_timeout_s)
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
                                  _git_info(), run_filter)
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
                judge = judge_client(s.ollama_host, s.judge_model, s.embed_model, s.num_ctx,
                                     s.ollama_read_timeout_s)
                for model in judge.loaded_models():
                    if model not in {s.judge_model, f"{s.judge_model}:latest"}:
                        judge.unload(model)  # only the judge stays in memory
                print(f"loaded before judging: {judge.loaded_models()}")
                judge_info = describe_judge(
                    judge, judge.model_digests([s.judge_model])[s.judge_model])
            evaluate(path, dataset, Corpus({d.id: d for d in docs}, users), judge, judge_info,
                     publisher, resume=args.resume)
            print(f"evaluated {path}; report: uv run python -m evals.stats report {path}")
    except (RunError, DatasetError) as exc:
        print(f"[fail] {exc}")
        return 1
    finally:
        publisher.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
