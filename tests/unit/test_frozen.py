"""Frozen config for the noise run: a night run over a changed config would measure nothing."""

import copy
import json
from pathlib import Path

import pytest

from evals.dataset import load_dataset
from evals.frozen import (
    FrozenConfigError,
    check_frozen,
    check_tree,
    diff_frozen,
    frozen_view,
    load_frozen,
    parse_porcelain,
    write_frozen,
)
from tests.helpers import dataset_item, write_dataset

ASSISTANT = {"gen_model": "qwen3:4b", "embed_model": "nomic", "k": 3, "num_ctx": 4096,
             "system_prompt_sha256": "p" * 64, "corpus_sha256": "c" * 64,
             "today": "2026-07-15"}
DIGESTS = {"qwen3:4b": "sha256:gen", "nomic": "sha256:emb"}
JUDGE = {"model": "qwen3:4b", "digest": "sha256:gen", "prompt_sha256": "j" * 64,
         "pass_score": 4, "num_predict": 256}


@pytest.fixture
def dataset(tmp_path: Path):
    write_dataset(tmp_path / "d.jsonl", [
        dataset_item("hol", "analyst", "Holiday?", expected=["27 days"]),
        dataset_item("orion", "lead", "Orion?", category="versioning", expected=["4.2 million"],
                     forbidden=["5 million"]),
    ])
    return load_dataset(tmp_path / "d.jsonl", {"analyst", "lead"})


@pytest.fixture
def frozen(dataset) -> dict:
    return frozen_view(dataset, ASSISTANT, DIGESTS, JUDGE)


def changed(view: dict, dotted: str, value) -> dict:
    out = copy.deepcopy(view)
    *path, last = dotted.split(".")
    target = out
    for key in path:
        target = target[key]
    target[last] = value
    return out


@pytest.mark.parametrize("field, value", [
    ("dataset.sha256", "other"),
    ("dataset.generation_key", "other"),
    ("dataset.as_of", "2027-03-01"),
    ("assistant.k", 5),
    ("assistant.num_ctx", 8192),
    ("assistant.system_prompt_sha256", "q" * 64),
    ("assistant.corpus_sha256", "d" * 64),
    ("assistant.gen_model", "qwen3:8b"),
    ("model_digests.qwen3:4b", "sha256:repulled"),
    ("judge.prompt_sha256", "v6" * 32),
    ("judge.digest", "sha256:repulled"),
    ("judge.num_predict", 512),
])
def test_every_frozen_field_that_changes_refuses_the_run(frozen, field, value):
    live = changed(frozen, field, value)

    with pytest.raises(FrozenConfigError, match="differs from frozen") as exc:
        check_frozen(frozen, live, Path("f.json"))

    assert field in str(exc.value)  # the message names the field, not just "mismatch"


def test_all_differences_are_listed_not_only_the_first(frozen):
    live = changed(changed(frozen, "assistant.k", 5), "judge.num_predict", 512)

    assert diff_frozen(frozen, live) == ["assistant.k: 3 -> 5", "judge.num_predict: 256 -> 512"]


def test_no_judge_where_a_judge_was_frozen_is_a_difference(frozen):
    # evaluate --no-judge under a frozen judge would store grades the noise report cannot use.
    assert diff_frozen(frozen, changed(frozen, "judge", None))


def test_dataset_path_alone_is_not_a_difference(frozen):
    # D:/x/d.jsonl and d.jsonl hold the same items; the hashes decide.
    assert diff_frozen(frozen, changed(frozen, "dataset.path", "elsewhere/d.jsonl")) == []


def test_per_item_date_is_not_frozen_as_an_assistant_setting(frozen):
    # `today` is set per item from the dataset; it is covered by dataset.as_of and generation_key.
    assert "today" not in frozen["assistant"]


def test_frozen_file_is_never_overwritten(tmp_path, frozen):
    path = tmp_path / "frozen" / "p4.json"
    write_frozen(frozen, path)

    with pytest.raises(FrozenConfigError, match="not overwritten"):
        write_frozen(changed(frozen, "assistant.k", 5), path)

    loaded, sha = load_frozen(path)
    assert loaded == json.loads(json.dumps(frozen)) and len(sha) == 64


def test_porcelain_splits_tracked_changes_from_untracked_files():
    status = " M evals/stats.py\nM  evals/judge.py\n D old.py\n?? notes.md\n?? scratch/\n"

    assert parse_porcelain(status) == (["evals/stats.py", "evals/judge.py", "old.py"],
                                       ["notes.md", "scratch/"])


def test_tracked_change_refuses_a_frozen_run_and_untracked_only_warns():
    # Evaluator or parsing code is not hashed into the config; a tracked edit could change grades.
    with pytest.raises(FrozenConfigError, match="evals/evaluators.py"):
        check_tree(" M evals/evaluators.py\n?? notes.md\n")

    assert check_tree("?? notes.md\n") == ["notes.md"]
    assert check_tree("") == []
