"""Frozen config for a noise measurement (Phase 4): everything that must stay the same across it.

`run_experiment freeze` writes the file from what the current settings and Ollama would use; Marko
reviews and commits it. `generate|evaluate|rejudge --frozen <file>` then refuse to run when any
field differs, listing every difference. It is not a baseline and holds no thresholds.

Code that is not hashed (evaluators, answer parsing) is covered by the git check instead: tracked
changes refuse, untracked files only warn (a new notes file must not stop a night run).
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from evals.dataset import Dataset

# The assistant settings that change answers; `today` is per item (dataset as_of).
ASSISTANT_KEYS = ("gen_model", "embed_model", "k", "num_ctx", "system_prompt_sha256",
                  "corpus_sha256")


class FrozenConfigError(RuntimeError):
    pass


def frozen_view(dataset: Dataset, assistant: dict[str, Any],
                model_digests: dict[str, str | None],
                judge: dict[str, Any] | None) -> dict[str, Any]:
    """`dataset` is the whole file, never a filtered subset: a filtered dry run over the frozen
    dataset is still the frozen dataset (the filter is recorded in the run config)."""
    return {
        "dataset": {"path": dataset.path.as_posix(), "sha256": dataset.sha256,
                    "generation_key": dataset.generation_key(),
                    "as_of": dataset.as_of.isoformat()},
        "assistant": {key: assistant[key] for key in ASSISTANT_KEYS},
        "model_digests": dict(sorted(model_digests.items())),
        "judge": judge,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            out |= _flatten(sub, f"{prefix}.{key}" if prefix else key)
        return out
    return {prefix: value}


# Where the dataset file sits is not what it holds; the hashes are.
IGNORED = frozenset({"dataset.path"})


def diff_frozen(frozen: dict[str, Any], live: dict[str, Any], *,
                sections: tuple[str, ...] = ("dataset", "assistant", "model_digests",
                                             "judge")) -> list[str]:
    """Every field that differs, as "key: frozen -> live". `sections` limits the check to what a
    command uses: generate does not load the judge, but still checks its digest is the frozen one
    so that a night run does not fail hours later at evaluate."""
    want = {k: v for s in sections for k, v in _flatten({s: frozen.get(s)}).items()}
    have = {k: v for s in sections for k, v in _flatten({s: live.get(s)}).items()}
    return [f"{key}: {want.get(key)!r} -> {have.get(key)!r}"
            for key in sorted(set(want) | set(have))
            if key not in IGNORED and want.get(key) != have.get(key)]


def load_frozen(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_frozen(view: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FrozenConfigError(f"{path} exists; a frozen config is not overwritten, delete it "
                                "first if you mean to re-freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(view, indent=2) + "\n", encoding="utf-8")


def check_frozen(frozen: dict[str, Any], live: dict[str, Any], path: Path,
                 sections: tuple[str, ...] = ("dataset", "assistant", "model_digests",
                                              "judge")) -> None:
    diffs = diff_frozen(frozen, live, sections=sections)
    if diffs:
        raise FrozenConfigError(f"config differs from frozen {path}:\n  " + "\n  ".join(diffs))


def parse_porcelain(output: str) -> tuple[list[str], list[str]]:
    """`git status --porcelain` -> (tracked changes, untracked files). `??` marks untracked;
    every other status (modified, staged, deleted, renamed) is a tracked change."""
    tracked, untracked = [], []
    for line in output.splitlines():
        if not line.strip():
            continue
        (untracked if line.startswith("??") else tracked).append(line[3:])
    return tracked, untracked


def check_tree(status: str | None = None) -> list[str]:
    """Refuse tracked changes; return untracked files for the caller to warn about.
    `status` is injectable for tests; by default git is asked."""
    if status is None:
        try:
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FrozenConfigError(f"cannot read git status ({exc}); a frozen run needs a "
                                    "known code state") from exc
    tracked, untracked = parse_porcelain(status)
    if tracked:
        raise FrozenConfigError("tracked files changed; commit or stash them before a frozen "
                                "run:\n  " + "\n  ".join(tracked))
    return untracked
