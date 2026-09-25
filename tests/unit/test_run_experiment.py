import json
from datetime import date
from pathlib import Path

import pytest

from evals.dataset import Dataset, load_dataset
from evals.evaluators import Corpus
from evals.publish import NoopPublisher
from evals.results import load
from evals.run_experiment import RunError, build_config, evaluate, generate
from mini_rag.assistant import Assistant
from mini_rag.documents import load_documents
from mini_rag.llm import FakeEmbedder, FakeLLM
from mini_rag.retrieval import Retriever
from mini_rag.users import User
from tests.helpers import dataset_item, write_dataset, write_doc

HOLIDAY_Q = "How many days of paid holiday do employees get?"
ORION_Q = "What is the Orion project budget?"
HOLIDAY = json.dumps({"answer": "27 days.", "citations": ["holiday-policy"], "refused": False})
ORION = json.dumps({"answer": "4.2 million euros.", "citations": ["orion-budget"],
                    "refused": False})
JUDGE_OK = json.dumps({"score": 5, "reason": "Supported."})
JUDGE_BAD = json.dumps({"score": 2, "reason": "Not supported."})


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
                         expected=orion_facts or ["4.2 million"], as_of="2027-03-01"),
        ])
        self.dataset = load_dataset(self.tmp_path / "d.jsonl", set(self.users))
        return self.dataset

    def assistant_for(self, llm: FakeLLM, k: int = 3):
        return lambda as_of: Assistant(self.retriever, llm, k=k, today=lambda: as_of,
                                       num_ctx=4096)

    def config(self, llm: FakeLLM, k: int = 3) -> dict:
        return build_config("run", self.dataset, self.assistant_for(llm, k)(self.dataset.as_of),
                            2, {"fake-llm": None}, {"git_commit": "abc", "git_dirty": False})

    def generate(self, *responses: str, resume: bool = False, k: int = 3):
        llm = FakeLLM(responses)
        run = generate(self.dataset, self.assistant_for(llm, k), self.users,
                       {d.id: d.access for d in self.docs}, self.config(llm, k), self.path,
                       NoopPublisher(), resume=resume, log=lambda _: None)
        return run, llm

    def evaluate(self, judge: FakeLLM | None):
        corpus = Corpus({d.id: d for d in load_documents(self.docs_dir)}, self.users)
        return evaluate(self.path, self.dataset, corpus, judge, None, NoopPublisher(),
                        log=lambda _: None)


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
    assert all(ev.passed for ev in first.evaluation.answers)
    assert not any(ev.passed for ev in second.evaluation.answers)  # judge verdict replaced
    assert load(env.path).evaluation == second.evaluation


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
    run = env.evaluate(None)
    assert run.evaluation.dataset_sha256 != old_sha
    assert not any(ev.passed for ev in run.evaluation.answers if ev.item_id == "orion")

    env.write_items("What did the board approve for Orion?")
    with pytest.raises(RunError, match="regenerate"):
        env.evaluate(None)


def test_evaluate_refuses_an_incomplete_run(env: Env):
    with pytest.raises(AssertionError):
        env.generate(HOLIDAY)

    with pytest.raises(RunError, match="3 answers missing"):
        env.evaluate(None)
