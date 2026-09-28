"""Two-pass experiment runner.

Pass 1, generate: ask every dataset item `--repeats` times, save each answer to results/<name>.json.
Pass 2, evaluate: grade the saved answers (code evaluators + LLM judge) and push scores to Langfuse.

Two passes so only one large model is in memory at a time (the gen model is unloaded after pass 1,
everything but the judge before pass 2), and so the judge can be re-run without regenerating.

Usage:
    uv run python -m evals.run_experiment generate --name <run> --repeats 5 [--dataset P] [--resume]
        [--items id1,id2] [--category c1,c2]    # filtered dev run; not comparable with full runs
    uv run python -m evals.run_experiment evaluate --name <run> [--no-judge] [--resume]
    uv run python -m evals.run_experiment rejudge --name <run> [--repeat 1] --samples 3 [--resume]
    uv run python -m evals.run_experiment publish --name <run>   # re-send what Langfuse missed
    uv run python -m evals.run_experiment freeze --dataset P --out evals/frozen/<name>.json
    # --frozen <file> on generate/evaluate: refuse to run when anything differs from the file
"""

import argparse
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
from evals.frozen import (
    FrozenConfigError,
    check_frozen,
    check_tree,
    frozen_view,
    load_frozen,
    write_frozen,
)
from evals.instrument import docs_sha256, evaluator_sha256, git_info, sticky_dirty, users_sha256
from evals.judge import JUDGE_NUM_PREDICT, JUDGE_PROMPT_SHA256, PASS_SCORE, judge_faithfulness
from evals.publish import (
    FailSafePublisher,
    LangfusePublisher,
    NoopPublisher,
    Publisher,
    publish_scores,
)
from evals.results import (
    AnswerEvaluation,
    AnswerRecord,
    Evaluation,
    JudgeSample,
    RunResults,
    SampledJudgement,
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
        grow_repeats(run, config["repeats"], log)
        run.config["git_dirty"] = sticky_dirty(run.config.get("git_dirty"),
                                               config.get("git_dirty"))
        save(run, path)
        config = run.config
    else:
        run = RunResults(config=config)

    safe = FailSafePublisher(publisher, run.publish_failures.append, log)
    safe.upload_dataset(dataset)
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
                question=item.question, as_of=as_of, access_by_id=access_by_id,
                generated_at=datetime.now(UTC))
            run.answers.append(record)
            save(run, path)  # after every answer: a crash loses at most the current one
            if record.trace_id:
                failures = len(run.publish_failures)
                safe.link(dataset, item, run_name, record.trace_id, link_metadata(config),
                          repeat=repeat)
                if len(run.publish_failures) > failures:
                    save(run, path)  # the failure is recorded, so `publish` can re-send it
            status = record.error or record.refusal_reason or "answered"
            log(f"[{len(run.answers)}/{total}] {item.id} r{repeat} {status} "
                f"({time.perf_counter() - t0:.1f}s)")
    return run


def grow_repeats(run: RunResults, repeats: int, log: Log) -> None:
    """Repeats may only grow: more repeats of the same config add answers, fewer would leave
    stored ones outside the run. `repeats` is not frozen, so the growth is recorded instead."""
    stored = run.config["repeats"]
    if repeats < stored:
        raise RunError(f"repeats can only grow: the run has {stored}, asked for {repeats}")
    if repeats == stored:
        return
    run.config["repeats"] = repeats
    run.config["langfuse_runs"] = run_names(run.config["name"], repeats)
    run.config.setdefault("repeats_history", []).append(
        {"from": stored, "to": repeats, "at": datetime.now(UTC).isoformat(timespec="seconds")})
    log(f"repeats {stored} -> {repeats}: adding the missing repeats")


def link_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Dataset-run metadata in Langfuse: a filtered dev run must be recognisable there too."""
    if config.get("filter") is None:
        return config["assistant"]
    return {**config["assistant"], "filter": config["filter"]}


def check_run_current(run: RunResults, dataset: Dataset, corpus: Corpus) -> Dataset:
    """The saved answers still belong to this dataset and corpus, and none is missing.
    Returns the dataset filtered to the items the run generated."""
    dataset = filter_dataset(dataset, run.config.get("filter"))
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
    return dataset


def evaluate(path: Path, dataset: Dataset, corpus: Corpus, judge: LLMClient | None,
             judge_info: dict[str, Any] | None, publisher: Publisher, *, resume: bool = False,
             instrument: dict[str, Any] | None = None, log: Log = print) -> RunResults:
    """`instrument`: evaluator code hash and git state of this session (evals.instrument)."""
    run = load(path)
    dataset = check_run_current(run, dataset, corpus)

    key = evaluation_key(judge_info)  # same judge prompt: overwrite; new prompt: add alongside
    evaluation = start_or_resume(run, key, dataset.sha256, judge_info, resume, log, instrument)
    if evaluation.finished_at is not None:  # --resume on a finished evaluation: nothing to do
        return run
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
    send_scores(run, key, FailSafePublisher(publisher, run.publish_failures.append, log))
    save(run, path)  # with any Langfuse failures recorded
    return run


def send_scores(run: RunResults, key: str, publisher: Publisher) -> None:
    by_answer = {(ev.item_id, ev.repeat): ev for ev in run.evaluations[key].answers}
    for record in run.answers:
        ev = by_answer[(record.item_id, record.repeat)]
        publish_scores(publisher, record, ev.results, ev.passed, key)


def republish(path: Path, dataset: Dataset, publisher: Publisher,
              log: Log = print) -> RunResults:
    """Re-send what a Langfuse outage kept from arriving: the dataset upload and the links that
    failed, and ALL scores of every finished evaluation (queued scores can fail after queueing,
    unseen; deterministic ids make a re-send an overwrite). Failures of this attempt replace the
    recorded ones."""
    run = load(path)
    dataset = filter_dataset(dataset, run.config.get("filter"))
    if dataset.generation_key() != run.config["dataset"]["generation_key"]:
        raise RunError("dataset questions/users/dates changed since generation")
    old, run.publish_failures = run.publish_failures, []
    safe = FailSafePublisher(publisher, run.publish_failures.append, log)
    if any(f.kind == "dataset" for f in old):
        safe.upload_dataset(dataset)
    items = {i.id: i for i in dataset.items}
    links = [f for f in old if f.kind == "link"]
    for f in links:
        safe.link(dataset, items[f.item_id], f.run_name, f.trace_id, link_metadata(run.config),
                  repeat=f.repeat)
    finished_keys = [k for k, ev in run.evaluations.items() if ev.finished_at is not None]
    for key in finished_keys:
        send_scores(run, key, safe)
    safe.flush()
    save(run, path)
    untraced = sum(a.trace_id is None for a in run.answers)
    log(f"re-sent {len(links)} link(s) and the scores of {len(finished_keys)} evaluation(s); "
        f"failed again: {len(run.publish_failures)}; answers without a trace (nothing to "
        f"attach to): {untraced}")
    return run


def rejudge(path: Path, dataset: Dataset, corpus: Corpus, judge: LLMClient,
            judge_info: dict[str, Any], repeat: int, samples: int, *, resume: bool = False,
            log: Log = print) -> RunResults:
    """Judge noise: grade one repeat's saved answers `samples` more times with the judge that
    made the stored evaluation. Samples are stored next to the evaluation, never over it.
    --resume continues unfinished samples and adds missing ones; without it, existing samples
    are refused, so a re-run never throws measured samples away."""
    run = load(path)
    check_run_current(run, dataset, corpus)
    key = evaluation_key(judge_info)
    if key not in run.evaluations or run.evaluations[key].finished_at is None:
        raise RunError(f"no finished evaluation by judge {key}; run evaluate first (the "
                       "samples are compared with it)")
    if run.evaluations[key].judge != judge_info:
        raise RunError(f"judge {key} now differs from the one that evaluated this run "
                       f"({run.evaluations[key].judge} -> {judge_info}); samples would mix judges")
    records = [r for r in run.answers if r.repeat == repeat]
    if not records:
        raise RunError(f"run has no repeat {repeat}")
    existing = run.judge_samples.setdefault(key, [])
    if existing and not resume:
        raise RunError(f"judge {key} already has {len(existing)} sample(s); continue or extend "
                       "them with --resume")
    if any(s.repeat != repeat or s.judge != judge_info for s in existing):
        raise RunError(f"stored samples of judge {key} are of another repeat or judge model; "
                       "they cannot be extended with these")

    for number in range(1, samples + 1):
        sample = next((s for s in existing if s.sample == number), None)
        if sample is None:
            sample = JudgeSample(sample=number, repeat=repeat, started_at=datetime.now(UTC),
                                 finished_at=None, judge=judge_info, answers=[])
            existing.append(sample)
        if sample.finished_at is not None:
            log(f"sample {number} already finished")
            continue
        done = {(a.item_id, a.repeat) for a in sample.answers}
        for record in records:
            if (record.item_id, record.repeat) in done:
                continue
            sample.answers.append(SampledJudgement(
                item_id=record.item_id, repeat=record.repeat,
                result=judge_faithfulness(record, corpus.docs, judge)))
            save(run, path)  # after every answer: a crash loses at most the current one
            log(f"sample {number}/{samples} {record.item_id} r{record.repeat}")
        sample.finished_at = datetime.now(UTC)
        save(run, path)
    return run


def start_or_resume(run: RunResults, key: str, dataset_sha256: str,
                    judge_info: dict[str, Any] | None, resume: bool, log: Log,
                    instrument: dict[str, Any] | None = None) -> Evaluation:
    """Resumable = an unfinished evaluation by the same judge (model digest too) against the same
    facts; only then can its grades be mixed with new ones. Without --resume a resumable one is
    not thrown away by accident; one that can no longer be resumed is replaced.

    --resume means "continue or start, never throw away", so a night script can always pass it:
    nothing stored -> start; resumable -> continue; finished by the same judge and facts -> return
    it as is (the caller does nothing), or reopen it when repeats were added since; anything else
    would be replaced -> refuse."""
    old = run.evaluations.pop(key, None)  # re-inserted below: the one written last is latest
    unfinished = old is not None and old.finished_at is None
    # Evaluator code is compared only when both sides recorded it (files before Phase 5 did not).
    old_code = (old.instrument or {}).get("evaluator_sha256") if old is not None else None
    new_code = (instrument or {}).get("evaluator_sha256")
    same = (old is not None
            and (old.dataset_sha256, old.judge) == (dataset_sha256, judge_info)
            and (old_code is None or new_code is None or old_code == new_code))
    resumable = unfinished and same
    if resume:
        if old is not None and not same:
            run.evaluations[key] = old
            raise RunError(f"no unfinished evaluation by judge {key} with these facts, this "
                           "judge model and this evaluator code to resume, and the stored one "
                           "would be replaced; run evaluate without --resume")
        if old is not None:
            if old.instrument is not None and instrument is not None:
                old.instrument["git_dirty"] = sticky_dirty(old.instrument.get("git_dirty"),
                                                           instrument.get("git_dirty"))
            elif not old.answers:
                old.instrument = instrument
            if not unfinished and len(old.answers) < len(run.answers):
                # Repeats were added after it finished: grade the new answers, keep the old.
                old.finished_at = None
                state = f"extending: {len(run.answers) - len(old.answers)} new answers"
            else:
                state = ("already finished" if not unfinished else
                         f"resuming: {len(old.answers)} of {len(run.answers)} answers done")
            log(f"judge {key} {state}")
            run.evaluations[key] = old
            return old
        log(f"judge {key}: nothing to resume, starting")
    if resumable:
        raise RunError(f"judge {key} has an unfinished evaluation ({len(old.answers)} of "
                       f"{len(run.answers)} answers); continue it with --resume")
    if unfinished:
        log(f"replacing unfinished evaluation by judge {key}: other facts or judge model")
    run.evaluations[key] = Evaluation(evaluated_at=datetime.now(UTC), finished_at=None,
                                      dataset_sha256=dataset_sha256, judge=judge_info,
                                      answers=[], instrument=instrument)
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


ALL_SECTIONS = ("dataset", "assistant", "model_digests", "judge")


def apply_frozen(path: Path, live: dict[str, Any], log: Log = print,
                 tree_status: str | None = None) -> dict[str, str]:
    """Refuse unless the live config equals the frozen file and no tracked file is changed.
    Returns what the run config records about it."""
    for name in check_tree(tree_status):
        log(f"[warn] untracked file (not part of the frozen check): {name}")
    frozen, sha256 = load_frozen(path)
    check_frozen(frozen, live, path, ALL_SECTIONS)
    log(f"config matches frozen {path} ({sha256[:12]})")
    return {"path": path.as_posix(), "sha256": sha256}


def check_run_frozen(run: RunResults, frozen: dict[str, str]) -> None:
    """Evaluating under a frozen config only means something if the answers were generated
    under the same one."""
    recorded = run.config.get("frozen")
    if recorded is None or recorded["sha256"] != frozen["sha256"]:
        raise RunError(f"run {run.config['name']} was not generated with --frozen "
                       f"{frozen['path']} (recorded: {recorded})")


def _csv(value: str) -> list[str]:
    return sorted({v.strip() for v in value.split(",") if v.strip()})


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
    gen.add_argument("--frozen", type=Path, help="refuse to run unless the config equals this")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", required=True)
    ev.add_argument("--no-judge", action="store_true")
    ev.add_argument("--resume", action="store_true")
    ev.add_argument("--frozen", type=Path)
    rj = sub.add_parser("rejudge")
    rj.add_argument("--name", required=True)
    rj.add_argument("--repeat", type=int, default=1)
    rj.add_argument("--samples", type=int, required=True)
    rj.add_argument("--resume", action="store_true")
    rj.add_argument("--frozen", type=Path)
    pb = sub.add_parser("publish")
    pb.add_argument("--name", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    fr.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    s = Settings()
    users = load_users(s.users_file)
    docs = load_documents(s.docs_dir)
    langfuse = make_langfuse(s)
    publisher: Publisher = NoopPublisher() if langfuse is None else LangfusePublisher(langfuse)
    path = results_path(args.name) if hasattr(args, "name") else None

    def current_judge_info(client: OllamaClient) -> dict[str, Any]:
        """What the judge would be, without loading it: for the frozen check before generating."""
        judge = judge_client(s.ollama_host, s.judge_model, s.embed_model, s.num_ctx,
                             s.ollama_read_timeout_s)
        return describe_judge(judge, client.model_digests([s.judge_model])[s.judge_model])

    try:
        if args.command in ("generate", "freeze"):
            dataset = full = load_dataset(args.dataset, set(users))
            check_facts_in_corpus(dataset, docs)
            check_expected_docs(dataset, docs, users)
            run_filter = ({"items": args.items, "categories": args.category}
                          if args.command == "generate" and (args.items or args.category)
                          else None)
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

            digests = client.model_digests([s.gen_model, s.embed_model])
            assistant_config = assistant_for(dataset.as_of).run_config()
            live = frozen_view(full, assistant_config, digests, current_judge_info(client))
            if args.command == "freeze":
                write_frozen(live, args.out)
                print(f"wrote {args.out}; review it and commit it before the frozen run")
                return 0
            config = build_config(args.name, dataset, assistant_for(dataset.as_of), args.repeats,
                                  digests, git_info(), run_filter)
            config["instrument"] = {"docs_sha256": docs_sha256(docs),
                                    "users_sha256": users_sha256(users)}
            if args.frozen:
                config["frozen"] = apply_frozen(args.frozen, live)
            generate(dataset, assistant_for, users, {d.id: d.access for d in docs}, config,
                     path, publisher, resume=args.resume)
            try:
                tracer.flush()
            except Exception as exc:  # traces are sent in the background; never lose the run
                print(f"[warn] Langfuse trace flush failed ({exc}); answers are saved locally")
            client.unload(s.gen_model)
            print(f"unloaded {s.gen_model}; loaded now: {client.loaded_models()}")
        elif args.command == "publish":
            if langfuse is None:
                raise RunError("no Langfuse keys configured; nothing to publish to")
            run = load(path)
            dataset = load_dataset(Path(run.config["dataset"]["path"]), set(users))
            republish(path, dataset, publisher)
        else:
            run = load(path)
            dataset = load_dataset(Path(run.config["dataset"]["path"]), set(users))
            check_facts_in_corpus(dataset, docs)
            check_expected_docs(dataset, docs, users)
            judge = judge_info = None
            if not getattr(args, "no_judge", False):
                judge = judge_client(s.ollama_host, s.judge_model, s.embed_model, s.num_ctx,
                                     s.ollama_read_timeout_s)
                for model in judge.loaded_models():
                    if model not in {s.judge_model, f"{s.judge_model}:latest"}:
                        judge.unload(model)  # only the judge stays in memory
                print(f"loaded before judging: {judge.loaded_models()}")
                judge_info = describe_judge(
                    judge, judge.model_digests([s.judge_model])[s.judge_model])
            if args.frozen:
                frozen = apply_frozen(args.frozen, frozen_view(
                    dataset, run.config["assistant"], run.config["model_digests"], judge_info))
                check_run_frozen(run, frozen)
            corpus = Corpus({d.id: d for d in docs}, users)
            if args.command == "rejudge":
                rejudge(path, dataset, corpus, judge, judge_info, args.repeat, args.samples,
                        resume=args.resume)
                print(f"re-judged {path}; noise: uv run python -m evals.stats noise {path}")
            else:
                evaluate(path, dataset, corpus, judge, judge_info, publisher,
                         resume=args.resume,
                         instrument={"evaluator_sha256": evaluator_sha256(), **git_info()})
                print(f"evaluated {path}; report: uv run python -m evals.stats report {path}")
    except (RunError, DatasetError, FrozenConfigError) as exc:
        print(f"[fail] {exc}")
        return 1
    finally:
        FailSafePublisher(publisher, lambda _: None).flush()  # a failed flush only warns
    return 0


if __name__ == "__main__":
    sys.exit(main())
