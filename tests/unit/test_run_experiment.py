import json
from collections import Counter
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from evals.dataset import Dataset, DatasetError, filter_dataset, load_dataset
from evals.evaluators import Corpus
from evals.frozen import FrozenConfigError, frozen_view, write_frozen
from evals.judge import JUDGE_NUM_PREDICT
from evals.publish import NoopPublisher
from evals.results import evaluation_key, load
from evals.run_experiment import (
    RunError,
    apply_frozen,
    build_config,
    check_run_frozen,
    describe_judge,
    evaluate,
    generate,
    judge_client,
    link_metadata,
    rejudge,
    republish,
)
from evals.stats import report
from mini_rag.assistant import Assistant
from mini_rag.documents import load_documents
from mini_rag.llm import FakeEmbedder, FakeLLM, LLMTimeoutError
from mini_rag.retrieval import Retriever
from mini_rag.tracing import NoopTracer, _NoopAnswerTrace
from mini_rag.users import User
from tests.helpers import dataset_item, write_dataset, write_doc

HOLIDAY_Q = "How many days of paid holiday do employees get?"
ORION_Q = "What is the Orion project budget?"
HOLIDAY = json.dumps({"answer": "27 days.", "citations": ["holiday-policy"], "refused": False})
ORION = json.dumps({"answer": "4.2 million euros.", "citations": ["orion-budget"],
                    "refused": False})
JUDGE_OK = json.dumps({"reason": "Supported.", "score": 5})
JUDGE_BAD = json.dumps({"reason": "Not supported.", "score": 2})
JUDGE_V1 = {"model": "fake-judge", "digest": None, "prompt_sha256": "1" * 64, "pass_score": 4}
JUDGE_V2 = JUDGE_V1 | {"prompt_sha256": "2" * 64}


class Env:
    """Fixture corpus + 2-item dataset (one item with an as_of override), 2 repeats."""

    def __init__(self, tmp_path: Path, docs_dir: Path, analyst: User, lead: User) -> None:
        self.tmp_path, self.docs_dir = tmp_path, docs_dir
        self.users = {"analyst": analyst, "lead": lead}
        self.docs = load_documents(docs_dir)
        self.retriever = Retriever(self.docs, FakeEmbedder())
        self.path = tmp_path / "results" / "run.json"
        self.write_items(ORION_Q)

    def write_items(self, orion_question: str, orion_facts: list | None = None) -> Dataset:
        write_dataset(self.tmp_path / "d.jsonl", [
            dataset_item("hol", "analyst", HOLIDAY_Q, expected=["27 days"]),
            dataset_item("orion", "lead", orion_question, category="versioning",
                         expected=orion_facts or ["4.2 million"], forbidden=["5 million"],
                         as_of="2027-03-01"),
        ])
        self.dataset = load_dataset(self.tmp_path / "d.jsonl", set(self.users))
        return self.dataset

    def assistant_for(self, llm: FakeLLM, k: int = 3, tracer=None):
        return lambda as_of: Assistant(self.retriever, llm, k=k, today=lambda: as_of,
                                       num_ctx=4096, tracer=tracer)

    def config(self, llm: FakeLLM, k: int = 3, run_filter: dict | None = None,
               repeats: int = 2) -> dict:
        dataset = filter_dataset(self.dataset, run_filter)
        return build_config("run", dataset, self.assistant_for(llm, k)(dataset.as_of),
                            repeats, {"fake-llm": None}, {"git_commit": "abc", "git_dirty": False},
                            run_filter)

    def generate(self, *responses: str, resume: bool = False, k: int = 3,
                 run_filter: dict | None = None, publisher=None, tracer=None,
                 repeats: int = 2):
        llm = FakeLLM(responses)
        assistant_for = self.assistant_for(llm, k, tracer)
        run = generate(filter_dataset(self.dataset, run_filter), assistant_for,
                       self.users, {d.id: d.access for d in self.docs},
                       self.config(llm, k, run_filter, repeats), self.path,
                       publisher or NoopPublisher(), resume=resume, log=lambda _: None)
        return run, llm

    def evaluate(self, judge: FakeLLM | None, judge_info: dict | None = None,
                 resume: bool = False, publisher=None):
        corpus = Corpus({d.id: d for d in load_documents(self.docs_dir)}, self.users)
        return evaluate(self.path, self.dataset, corpus, judge,
                        judge_info or (JUDGE_V1 if judge else None),
                        publisher or NoopPublisher(), resume=resume, log=lambda _: None)


@pytest.fixture
def env(tmp_path, docs_dir, analyst, lead) -> Env:
    return Env(tmp_path, docs_dir, analyst, lead)


def test_generate_saves_every_answer_with_run_config_and_per_item_date(env: Env):
    run, llm = env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    saved = load(env.path)
    assert saved == run
    assert [(a.item_id, a.repeat) for a in saved.answers] == [
        ("hol", 1), ("orion", 1), ("hol", 2), ("orion", 2)]
    assert {a.item_id: a.as_of for a in saved.answers} == {
        "hol": date(2026, 7, 15), "orion": date(2027, 3, 1)}
    assert "Today is 2027-03-01" in llm.calls[1][1]  # the override reached the prompt
    config = saved.config
    assert config["as_of"] == "2026-07-15"
    assert config["dataset"]["sha256"] == env.dataset.sha256
    assert config["assistant"]["corpus_sha256"] == env.retriever.corpus_hash
    assert config["langfuse_runs"] == ["run-r1", "run-r2"]


def test_crashed_run_keeps_finished_answers_and_resume_completes_it(env: Env):
    with pytest.raises(AssertionError, match="ran out"):  # FakeLLM empty = model crash
        env.generate(HOLIDAY)
    first = load(env.path).answers
    assert len(first) == 1

    run, llm = env.generate(ORION, HOLIDAY, ORION, resume=True)

    assert len(run.answers) == 4 and run.answers[0] == first[0]
    assert len(llm.calls) == 3  # the finished answer was not asked again


def test_existing_run_is_not_overwritten_without_resume(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    with pytest.raises(RunError, match="use --resume"):
        env.generate(HOLIDAY)


def test_resume_refuses_a_changed_config(env: Env):
    with pytest.raises(AssertionError):
        env.generate(HOLIDAY)

    with pytest.raises(RunError, match="different config"):
        env.generate(ORION, HOLIDAY, ORION, resume=True, k=2)


def test_judge_rerun_regrades_saved_answers_without_regenerating(env: Env):
    generated, _ = env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    first = env.evaluate(FakeLLM([JUDGE_OK] * 4))
    second = env.evaluate(FakeLLM([JUDGE_BAD] * 4))

    assert second.answers == generated.answers  # untouched by pass 2
    assert all(ev.passed for ev in first.evaluation()[1].answers)
    # Same judge version: its verdict is replaced, not stored twice.
    assert list(second.evaluations) == ["111111111111"]
    assert not any(ev.passed for ev in second.evaluation()[1].answers)
    assert load(env.path).evaluations == second.evaluations


def test_new_judge_version_is_stored_next_to_the_old_one(env: Env):
    # Losing the old judge's grades would make a judge change impossible to compare.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_BAD] * 4), JUDGE_V1)

    run = env.evaluate(FakeLLM([JUDGE_OK] * 4), JUDGE_V2)

    saved = load(env.path)
    assert set(saved.evaluations) == {"111111111111", "222222222222"}
    assert not any(ev.passed for ev in saved.evaluations["111111111111"].answers)
    assert all(ev.passed for ev in saved.evaluations["222222222222"].answers)
    assert run.evaluation()[0] == "222222222222"  # latest is the default for reports


def test_same_judge_prompt_with_an_output_cap_is_another_judge_version(env: Env):
    # A cap changes which replies survive, so it must not overwrite the uncapped grades.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_OK] * 4), JUDGE_V1)

    env.evaluate(FakeLLM([JUDGE_OK] * 4), JUDGE_V1 | {"num_predict": 200})

    assert set(load(env.path).evaluations) == {"111111111111", "111111111111-np200"}


def test_judge_that_evaluates_sends_the_cap_its_results_are_stored_under():
    # A cap recorded in judge_info but not sent would file uncapped grades under a capped key.
    client = judge_client("http://ollama.test", "judge", "embed", 4096, 60.0)

    info = describe_judge(client, None)

    assert client.num_predict == JUDGE_NUM_PREDICT == info["num_predict"]
    assert evaluation_key(info).endswith(f"-np{JUDGE_NUM_PREDICT}")


def test_evaluate_refuses_a_changed_corpus(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    write_doc(env.docs_dir, "holiday-policy.md", "holiday-policy", "Holiday policy", "public",
              "Employees get 30 days of paid holiday per year.", "2026-06-01")

    with pytest.raises(RunError, match="corpus changed"):
        env.evaluate(None)


def test_evaluate_refuses_changed_questions_but_regrades_changed_facts(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    old_sha = env.dataset.sha256

    env.write_items(ORION_Q, orion_facts=["BLUEHERON"])
    _, evaluation = env.evaluate(None).evaluation()
    assert evaluation.dataset_sha256 != old_sha
    assert not any(ev.passed for ev in evaluation.answers if ev.item_id == "orion")

    env.write_items("What did the board approve for Orion?")
    with pytest.raises(RunError, match="regenerate"):
        env.evaluate(None)


def test_evaluate_refuses_an_incomplete_run(env: Env):
    with pytest.raises(AssertionError):
        env.generate(HOLIDAY)

    with pytest.raises(RunError, match="3 answers missing"):
        env.evaluate(None)


def test_judge_timeout_mid_run_becomes_one_counted_judge_error_and_the_run_finishes(env: Env):
    # Before: an Ollama read timeout escaped evaluate and aborted the whole pass.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    judge = FakeLLM([JUDGE_OK, LLMTimeoutError("slow"), LLMTimeoutError("slow"), JUDGE_OK,
                     LLMTimeoutError("slow"), JUDGE_OK])

    run = env.evaluate(judge)

    _, evaluation = load(env.path).evaluation()
    assert evaluation.finished_at is not None
    assert [ev.passed for ev in evaluation.answers] == [True, False, True, True]
    judged = [r for ev in evaluation.answers for r in ev.results
              if r.name == "judge_faithfulness"]
    assert judged[1].detail == "judge_error: timeout, timeout"
    assert judged[3].passed and judged[3].attempt_errors == ["timeout"]  # retry succeeded
    assert "judge errors (counted as fails) 1  judge timeouts (incl. retried) 3" in report(run)


def test_crashed_evaluation_keeps_graded_answers_and_resume_finishes_it(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    with pytest.raises(AssertionError, match="ran out"):  # judge dies on the third answer
        env.evaluate(FakeLLM([JUDGE_OK, JUDGE_BAD]))

    saved = load(env.path)
    _, partial = saved.evaluation()
    assert [(ev.item_id, ev.passed) for ev in partial.answers] == [("hol", True),
                                                                   ("orion", False)]
    assert partial.finished_at is None
    with pytest.raises(ValueError, match="unfinished"):  # no rates over half a run
        report(saved)

    judge = FakeLLM([JUDGE_OK, JUDGE_OK])
    run = env.evaluate(judge, resume=True)

    _, evaluation = run.evaluation()
    assert len(judge.calls) == 2  # graded answers are not judged again
    assert [(ev.item_id, ev.repeat) for ev in evaluation.answers] == [
        ("hol", 1), ("orion", 1), ("hol", 2), ("orion", 2)]
    assert evaluation.answers[:2] == partial.answers
    assert evaluation.finished_at is not None and load(env.path) == run


def test_unfinished_evaluation_is_not_restarted_by_accident(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    with pytest.raises(AssertionError):
        env.evaluate(FakeLLM([JUDGE_OK]))

    with pytest.raises(RunError, match="continue it with --resume"):
        env.evaluate(FakeLLM([JUDGE_OK] * 4))


def test_resume_refuses_a_different_judge_model(env: Env):
    # Mixing grades from two judge models in one evaluation would measure neither.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    with pytest.raises(AssertionError):
        env.evaluate(FakeLLM([JUDGE_OK]))

    with pytest.raises(RunError, match="no unfinished evaluation"):
        env.evaluate(FakeLLM([JUDGE_OK] * 3), JUDGE_V1 | {"digest": "sha256:new"}, resume=True)


VERSIONING_ONLY = {"items": None, "categories": ["versioning"]}


@pytest.mark.parametrize("run_filter, label", [
    (VERSIONING_ONLY, "categories versioning"),
    ({"items": ["orion"], "categories": None}, "items orion"),
])
def test_filtered_run_asks_only_selected_items_and_evaluate_grades_the_same_items(
        env: Env, run_filter, label):
    run, llm = env.generate(ORION, ORION, run_filter=run_filter)

    assert [(a.item_id, a.repeat) for a in run.answers] == [("orion", 1), ("orion", 2)]
    assert len(llm.calls) == 2 and run.config["filter"] == run_filter
    # evaluate gets the whole dataset file and re-applies the run's filter; without it the
    # item hash would not match and "hol" would count as missing answers.
    evaluated = env.evaluate(FakeLLM([JUDGE_OK] * 2))
    assert [(ev.item_id, ev.passed) for ev in evaluated.evaluation()[1].answers] == [
        ("orion", True), ("orion", True)]
    assert f"FILTERED dev run: {label}; not comparable" in report(evaluated)


@pytest.mark.parametrize("run_filter, reason", [
    ({"items": ["orion", "orinon"], "categories": None}, "unknown items/categories"),
    ({"items": None, "categories": ["factul"]}, "unknown items/categories"),
    ({"items": ["hol"], "categories": ["versioning"]}, "matches no items"),
])
def test_filter_typo_or_empty_selection_is_refused_not_run_over_nothing(env: Env, run_filter,
                                                                       reason):
    with pytest.raises(DatasetError, match=reason):
        filter_dataset(env.dataset, run_filter)


def test_filtered_run_is_marked_in_langfuse_dataset_run_metadata(env: Env):
    llm = FakeLLM([])

    full, filtered = env.config(llm), env.config(llm, run_filter=VERSIONING_ONLY)

    assert "filter" not in link_metadata(full)
    assert link_metadata(filtered) == {**filtered["assistant"], "filter": VERSIONING_ONLY}


def test_frozen_run_records_the_file_and_evaluate_refuses_a_run_generated_without_it(
        env: Env, tmp_path):
    llm = FakeLLM([])
    config = env.config(llm)
    live = frozen_view(env.dataset, config["assistant"], config["model_digests"], JUDGE_V1)
    write_frozen(live, tmp_path / "frozen.json")
    warnings: list[str] = []

    recorded = apply_frozen(tmp_path / "frozen.json", live, warnings.append,
                            tree_status="?? notes.md\n")

    assert recorded["path"].endswith("frozen.json") and len(recorded["sha256"]) == 64
    assert any("untracked file" in w and "notes.md" in w for w in warnings)
    run, _ = env.generate(HOLIDAY, ORION, HOLIDAY, ORION)  # generated without --frozen
    with pytest.raises(RunError, match="not generated with --frozen"):
        check_run_frozen(run, recorded)
    run.config["frozen"] = recorded
    check_run_frozen(run, recorded)  # same file: accepted


def test_frozen_check_refuses_before_anything_runs_when_the_judge_changed(env: Env, tmp_path):
    # generate checks the judge too: a night run must not fail hours later at evaluate.
    llm = FakeLLM([])
    config = env.config(llm)
    write_frozen(frozen_view(env.dataset, config["assistant"], config["model_digests"],
                             JUDGE_V1), tmp_path / "frozen.json")
    live = frozen_view(env.dataset, config["assistant"], config["model_digests"], JUDGE_V2)

    with pytest.raises(FrozenConfigError, match="judge.prompt_sha256"):
        apply_frozen(tmp_path / "frozen.json", live, tree_status="")


def rejudge_env(env: Env, judge: FakeLLM, samples: int, *, resume: bool = False,
                judge_info: dict = JUDGE_V1):
    corpus = Corpus({d.id: d for d in load_documents(env.docs_dir)}, env.users)
    return rejudge(env.path, env.dataset, corpus, judge, judge_info, 1, samples, resume=resume,
                   log=lambda _: None)


def test_rejudge_stores_samples_next_to_the_evaluation_and_never_over_it(env: Env):
    # Judge noise must be measured on the same answers without touching the grades reports use.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    before = env.evaluate(FakeLLM([JUDGE_OK] * 4)).evaluations

    judge = FakeLLM([JUDGE_BAD, JUDGE_OK, JUDGE_OK, JUDGE_BAD])
    rejudge_env(env, judge, samples=2)

    saved = load(env.path)
    assert saved.evaluations == before
    samples = saved.judge_samples["111111111111"]
    assert [(s.sample, s.repeat, s.finished_at is not None) for s in samples] == [
        (1, 1, True), (2, 1, True)]
    assert [[(a.item_id, a.repeat, a.result.passed) for a in s.answers] for s in samples] == [
        [("hol", 1, False), ("orion", 1, True)], [("hol", 1, True), ("orion", 1, False)]]
    assert len(judge.calls) == 4  # repeat 1 only: 2 answers x 2 samples


def test_crashed_rejudge_resumes_without_losing_or_repeating_a_judgement(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_OK] * 4))
    with pytest.raises(AssertionError, match="ran out"):
        rejudge_env(env, FakeLLM([JUDGE_OK, JUDGE_BAD, JUDGE_OK]), samples=2)

    judge = FakeLLM([JUDGE_OK, JUDGE_OK, JUDGE_OK])
    run = rejudge_env(env, judge, samples=3, resume=True)  # also extends 2 -> 3 samples

    samples = run.judge_samples["111111111111"]
    assert len(judge.calls) == 3  # sample 2's second answer + both of sample 3
    assert [len({(a.item_id, a.repeat) for a in s.answers}) for s in samples] == [2, 2, 2]
    assert [len(s.answers) for s in samples] == [2, 2, 2]  # no duplicates
    assert samples[0].answers[1].result.passed is False  # kept from before the crash


def test_rejudge_without_resume_refuses_to_discard_existing_samples(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_OK] * 4))
    rejudge_env(env, FakeLLM([JUDGE_OK] * 2), samples=1)

    with pytest.raises(RunError, match="already has 1 sample"):
        rejudge_env(env, FakeLLM([JUDGE_OK] * 2), samples=1)


def test_rejudge_refuses_a_judge_model_other_than_the_one_that_evaluated(env: Env):
    # Mixing two judge builds would report their difference as judge noise.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_OK] * 4))

    with pytest.raises(RunError, match="samples would mix judges"):
        rejudge_env(env, FakeLLM([JUDGE_OK] * 2), samples=1,
                    judge_info=JUDGE_V1 | {"digest": "sha256:new"})


def test_rejudge_needs_a_finished_evaluation_to_compare_with(env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    with pytest.raises(RunError, match="run evaluate first"):
        rejudge_env(env, FakeLLM([JUDGE_OK] * 2), samples=1)


class TracedIds(NoopTracer):
    """Like NoopTracer, but every answer gets a trace id, so the runner links and scores it."""

    def __init__(self) -> None:
        self.count = 0

    @contextmanager
    def answer_trace(self, question, user, run_config, *, dataset_item_id=None):
        self.count += 1
        trace = _NoopAnswerTrace()
        trace.trace_id = f"{self.count:032x}"
        yield trace


class LangfuseDown:
    """Every call fails like an unreachable Langfuse; `up` turns it back on and records."""

    def __init__(self) -> None:
        self.up = False
        self.links: list[tuple[str, str]] = []
        self.scores: list[tuple[str, str]] = []

    def _check(self) -> None:
        if not self.up:
            raise ConnectionError("503 Service Unavailable")

    def upload_dataset(self, dataset) -> None:
        self._check()

    def link(self, dataset, item, run_name, trace_id, metadata) -> None:
        self._check()
        self.links.append((item.id, run_name))

    def score(self, trace_id, name, value, comment, score_id, metadata) -> None:
        self._check()
        self.scores.append((trace_id, name))

    def flush(self) -> None:
        self._check()


def test_langfuse_outage_never_fails_the_run_and_publish_resends_what_was_missed(env: Env):
    down = LangfuseDown()

    env.generate(HOLIDAY, ORION, HOLIDAY, ORION, publisher=down, tracer=TracedIds())
    run = env.evaluate(FakeLLM([JUDGE_OK] * 4), publisher=down)

    assert len(run.answers) == 4 and run.evaluation()[1].finished_at is not None
    saved = load(env.path)
    kinds = Counter(f.kind for f in saved.publish_failures)
    # 1 upload + 4 links + per answer 6 applicable scores (5 evaluators with judge, item_pass)
    assert kinds["dataset"] == 1 and kinds["link"] == 4 and kinds["score"] > 0
    assert "Langfuse NOT sent: dataset upload 1  links 4" in report(saved)

    down.up = True
    republished = republish(env.path, env.dataset, down, log=lambda _: None)

    assert sorted(down.links) == [("hol", "run-r1"), ("hol", "run-r2"),
                                  ("orion", "run-r1"), ("orion", "run-r2")]
    assert len(down.scores) == kinds["score"]  # every score, once
    assert republished.publish_failures == [] and load(env.path).publish_failures == []
    assert "Langfuse: no send failures recorded" in report(load(env.path))


def test_publish_keeps_what_failed_again_for_the_next_attempt(env: Env):
    down = LangfuseDown()
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION, publisher=down, tracer=TracedIds())

    republish(env.path, env.dataset, down, log=lambda _: None)  # still down

    assert Counter(f.kind for f in load(env.path).publish_failures) == {"dataset": 1, "link": 4}


def test_evaluate_resume_with_nothing_stored_starts_so_a_night_script_can_always_pass_it(
        env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    run = env.evaluate(FakeLLM([JUDGE_OK] * 4), resume=True)

    assert run.evaluation()[1].finished_at is not None
    assert len(run.evaluation()[1].answers) == 4


def test_evaluate_resume_on_a_finished_evaluation_changes_nothing(env: Env):
    # A re-run of the night script after a later step crashed must not re-judge or re-time it.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    before = env.evaluate(FakeLLM([JUDGE_OK] * 4)).evaluations

    judge = FakeLLM([])
    env.evaluate(judge, resume=True)

    assert judge.calls == [] and load(env.path).evaluations == before


def test_evaluate_resume_refuses_to_replace_a_finished_evaluation_graded_on_other_facts(
        env: Env):
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    env.evaluate(FakeLLM([JUDGE_OK] * 4))
    env.write_items(ORION_Q, orion_facts=["BLUEHERON"])

    with pytest.raises(RunError, match="would be replaced"):
        env.evaluate(FakeLLM([JUDGE_OK] * 4), resume=True)
    assert load(env.path).evaluation()[1].finished_at is not None  # still there


# --- repeats may only grow (a night run can be extended, never shrunk) -----------------------

def test_resume_with_more_repeats_adds_only_the_missing_ones_and_records_when(env: Env):
    first, _ = env.generate(HOLIDAY, ORION, HOLIDAY, ORION)

    run, llm = env.generate(HOLIDAY, ORION, resume=True, repeats=3)

    assert len(llm.calls) == 2  # r3 only; r1-r2 were not asked again
    assert run.answers[:4] == first.answers
    assert [(a.item_id, a.repeat) for a in run.answers[4:]] == [("hol", 3), ("orion", 3)]
    saved = load(env.path).config
    assert saved["repeats"] == 3
    assert saved["langfuse_runs"] == ["run-r1", "run-r2", "run-r3"]
    assert [(h["from"], h["to"]) for h in saved["repeats_history"]] == [(2, 3)]
    assert all(a.generated_at is not None for a in run.answers)
    assert min(a.generated_at for a in run.answers[4:]) >= max(
        a.generated_at for a in first.answers)


def test_resume_with_fewer_repeats_refuses_and_leaves_the_run_untouched(env: Env):
    # Fewer repeats would leave stored answers outside the run's own repeat count.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    before = env.path.read_bytes()

    with pytest.raises(RunError, match="repeats can only grow: the run has 2, asked for 1"):
        env.generate(HOLIDAY, resume=True, repeats=1)

    assert env.path.read_bytes() == before


def test_grown_run_is_not_reported_until_evaluate_resume_grades_the_new_repeat(env: Env):
    # Without reopening, evaluate --resume would say "already finished" and the noise report
    # would cover r1-r2 only, or crash on r3.
    env.generate(HOLIDAY, ORION, HOLIDAY, ORION)
    old = env.evaluate(FakeLLM([JUDGE_OK] * 4)).evaluation()[1].answers
    env.generate(HOLIDAY, ORION, resume=True, repeats=3)

    with pytest.raises(ValueError, match="covers 4 of 6 answers"):
        report(load(env.path))

    judge = FakeLLM([JUDGE_OK] * 2)
    run = env.evaluate(judge, resume=True)

    evaluation = run.evaluation()[1]
    assert len(judge.calls) == 2  # only the new answers were judged
    assert evaluation.finished_at is not None and evaluation.answers[:4] == old
    assert {ev.repeat for ev in evaluation.answers[4:]} == {3}
    report(load(env.path))  # complete again
