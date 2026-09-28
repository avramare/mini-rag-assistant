"""What a run was measured WITH (the instrument) vs what was measured (the system under test).

The regression gate (Phase 5) compares a candidate run with the baseline only when the instrument
is the same: the same questions, dates, grading facts, documents, users, judge and evaluator code.
A different instrument would move the numbers without the system changing, so the gate refuses;
the baseline has to be re-evaluated with the new instrument first. The system under test (prompt,
models, k, num_ctx, code commit) may differ: that difference is the experiment, and the gate lists
it.

`--frozen` (Phase 4) is stricter and separate: it pins BOTH groups and a clean tree, for noise runs.

Hashes are over parsed content or text with normalized line endings, so a checkout with CRLF
(another git worktree on Windows) hashes the same as one with LF.
"""

import hashlib
import inspect
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evals.frozen import parse_porcelain
from evals.results import RunResults
from mini_rag.assistant import Answer, parse_answer
from mini_rag.config import PROJECT_ROOT
from mini_rag.documents import Document
from mini_rag.users import User

# The code that turns saved answers into verdicts. `stats`/`gate` are not in it: they recompute
# both sides of a comparison from stored verdicts at gate time, so they cannot make them disagree.
EVALUATOR_FILES = ("evals/evaluators.py", "evals/judge.py", "evals/dataset.py")
# parse_answer/Answer belong to BOTH groups: the app uses them to read the model's reply, and
# `citations_valid`/`schema_valid` use them to re-check it. Only their source is hashed, not the
# rest of assistant.py (the prompt is system under test).
EVALUATOR_OBJECTS = (parse_answer, Answer)
# Git paths whose change since a run means its instrument differs from today's files.
INSTRUMENT_PATHS = ("data", *EVALUATOR_FILES, "src/mini_rag/assistant.py")
# Assistant config keys that are instrument, not system: corpus_sha256 mixes the documents with
# the embedding model, so the documents are hashed on their own (`docs_sha256`) instead.
NOT_SYSTEM_KEYS = frozenset({"corpus_sha256"})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def docs_sha256(docs: Iterable[Document]) -> str:
    """Documents as parsed (id, title, access, body, effective); no embedding model."""
    return _sha([d.model_dump(mode="json") for d in sorted(docs, key=lambda d: d.id)])


def users_sha256(users: dict[str, User]) -> str:
    """Clearances decide what a leak is (`citations_valid`). Sorted: a frozenset's order changes
    between processes (string hashing is randomized)."""
    return _sha({name: sorted(a.value for a in u.clearance) for name, u in users.items()})


def evaluator_sources(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """name -> sha256 of the source, line endings normalized."""
    out = {}
    for rel in EVALUATOR_FILES:
        text = (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        out[rel] = hashlib.sha256(text.encode()).hexdigest()
    for obj in EVALUATOR_OBJECTS:
        text = inspect.getsource(obj).replace("\r\n", "\n")
        out[f"{obj.__module__}.{obj.__qualname__}"] = hashlib.sha256(text.encode()).hexdigest()
    return out


def evaluator_sha256(root: Path = PROJECT_ROOT) -> str:
    return _sha(evaluator_sources(root))


def git_info() -> dict[str, Any]:
    """Commit and whether TRACKED files differ from it. Untracked files (a notes file, a prompt
    draft) do not make the tree dirty, the same rule as the frozen check. None = unknown."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(parse_porcelain(status)[0])}


def sticky_dirty(old: bool | None, new: bool | None) -> bool | None:
    """A run resumed across sessions is dirty if any session was; unknown if any was unknown."""
    if old is True or new is True:
        return True
    if old is None or new is None:
        return None
    return False


def changed_since(commit: str, paths: Iterable[str] = INSTRUMENT_PATHS) -> list[str]:
    """Instrument files that differ between `commit` and the working tree: tracked changes
    (committed or not) and new untracked files (a new doc changes the corpus)."""
    paths = list(paths)
    diff = subprocess.run(["git", "diff", "--name-only", commit, "--", *paths],
                          capture_output=True, text=True, check=True).stdout
    status = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                            capture_output=True, text=True, check=True).stdout
    return sorted(set(diff.split()) | set(parse_porcelain(status)[1]))


def instrument_of(run: RunResults, key: str,
                  fallback: dict[str, str] | None = None) -> dict[str, Any]:
    """The instrument of `run` graded by the evaluation `key`. Values the run did not record
    (runs before Phase 5) come from `fallback`, else stay None, which never equals a hash."""
    recorded = {**(fallback or {}), **(run.config.get("instrument") or {}),
                **(run.evaluations[key].instrument or {})}
    evaluation = run.evaluations[key]
    return {
        "as_of": run.config["as_of"],
        "dataset": {"generation_key": run.config["dataset"]["generation_key"],
                    "grading_sha256": evaluation.dataset_sha256},
        "docs_sha256": recorded.get("docs_sha256"),
        "users_sha256": recorded.get("users_sha256"),
        "judge_key": key,
        "judge": evaluation.judge,
        "evaluator_sha256": recorded.get("evaluator_sha256"),
    }


def system_of(run: RunResults) -> dict[str, Any]:
    c = run.config
    return {
        **{k: v for k, v in c["assistant"].items() if k not in NOT_SYSTEM_KEYS},
        "model_digests": dict(sorted((c.get("model_digests") or {}).items())),
        "git_commit": c.get("git_commit"),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, sub in value.items():
            out |= flatten(sub, f"{prefix}.{key}" if prefix else key)
        return out
    return {prefix: value}


def differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Every differing leaf as "key: old -> new", sorted."""
    a, b = flatten(old), flatten(new)
    return [f"{key}: {a.get(key)!r} -> {b.get(key)!r}"
            for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]
