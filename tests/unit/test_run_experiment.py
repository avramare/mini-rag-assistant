import json
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
)
from evals.stats import report
from mini_rag.assistant import Assistant
from mini_rag.documents import load_documents
from mini_rag.llm import FakeEmbedder, FakeLLM, LLMTimeoutError
from mini_rag.retrieval import Retriever
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

    def assistant_for(self, llm: FakeLLM, k: int = 3):
        return lambda as_of: Assistant(self.retriever, llm, k=k, today=lambda: as_of,
                                       num_ctx=4096)

    def config(self, llm: FakeLLM, k: int = 3, run_filter: dict | None = None) -> dict:
        dataset = filter_dataset(self.dataset, run_filter)
        return build_config("run", dataset, self.assistant_for(llm, k)(dataset.as_of),
                            2, {"fake-llm": None}, {"git_commit": "abc", "git_dirty": False},
                            run_filter)

    def generate(self, *responses: str, resume: bool = False, k: int = 3,
                 run_filter: dict | None = None):
        llm = FakeLLM(responses)
        run = generate(filter_dataset(self.dataset, run_filter), self.assistant_for(llm, k),
                       self.users, {d.id: d.access for d in self.docs},
                       self.config(llm, k, run_filter), self.path, NoopPublisher(),
                       resume=resume, log=lambda _: None)
        return run, llm

    def evaluate(self, judge: FakeLLM | None, judge_info: dict | None = None,
                 resume: bool = False):
        corpus = Corpus({d.id: d for d in load_documents(self.docs_dir)}, self.users)
        return evaluate(self.path, self.dataset, corpus, judge,
                        judge_info or (JUDGE_V1 if judge else None), NoopPublisher(),
                        resume=resume, log=lambda _: None)


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
