"""Code evaluators: pure functions of (dataset item, saved answer, corpus/users) -> EvalResult.

An answer passes when every APPLICABLE evaluator passes (`answer_passed`). Evaluators are an
independent oracle: they re-check the model's own output against corpus and users instead of
trusting the app's flags (`result.error`), so a bug in the app does not hide itself in the evals.

Text matching is `normalize` + substring (see evals.dataset.normalize). Known limits: no number
normalization, no synonyms, and a fact can match inside a longer token ("24" in "2024"). Keep facts
distinctive and list accepted spellings as variants.
"""

from collections.abc import Callable
from dataclasses import dataclass

from evals.dataset import DatasetItem, normalize, variants
from evals.results import AnswerRecord, EvalResult
from mini_rag.assistant import parse_answer
from mini_rag.documents import Document
from mini_rag.users import User


@dataclass(frozen=True)
class Corpus:
    docs: dict[str, Document]
    users: dict[str, User]


def _answer_text(record: AnswerRecord) -> str:
    return normalize(record.answer.answer) if record.answer else ""


def _matching(facts: list, text: str) -> list[int]:
    """1-based positions of the facts found in `text` (any variant matches)."""
    return [n for n, fact in enumerate(facts, 1)
            if any(normalize(v) in text for v in variants(fact))]


def schema_valid(item: DatasetItem, record: AnswerRecord, corpus: Corpus) -> EvalResult:
    ok = any(parse_answer(raw) is not None for raw in record.raw_outputs)
    return EvalResult(name="schema_valid", applicable=True, passed=ok,
                      detail=f"attempts={record.attempts}, invalid={record.invalid_outputs}")


def refusal_correct(item: DatasetItem, record: AnswerRecord, corpus: Corpus) -> EvalResult:
    refused = record.answer.refused if record.answer else None
    return EvalResult(name="refusal_correct", applicable=True,
                      passed=refused is not None and refused == item.should_refuse,
                      detail=f"refused={refused}, expected={item.should_refuse}")


def facts_recall(item: DatasetItem, record: AnswerRecord, corpus: Corpus) -> EvalResult:
    if not item.expected_facts:
        return EvalResult(name="facts_recall", applicable=False)
    found = _matching(item.expected_facts, _answer_text(record))
    total = len(item.expected_facts)
    missing = [n for n in range(1, total + 1) if n not in found]
    return EvalResult(name="facts_recall", applicable=True, passed=len(found) == total,
                      value=len(found) / total,
                      detail=f"found {len(found)}/{total}" + (f", missing #{missing}" if missing
                                                              else ""))


def forbidden_absent(item: DatasetItem, record: AnswerRecord, corpus: Corpus) -> EvalResult:
    if not item.forbidden_facts:
        return EvalResult(name="forbidden_absent", applicable=False)
    found = _matching(item.forbidden_facts, _answer_text(record))
    return EvalResult(name="forbidden_absent", applicable=True, passed=not found,
                      detail=f"forbidden facts present: #{found}" if found else "none present")


def citations_valid(item: DatasetItem, record: AnswerRecord, corpus: Corpus) -> EvalResult:
    """Checks the model's final parsed output, not `record.answer` or `record.error`: the app may
    have rejected or rewritten it, and the evaluator must see what the model actually did."""
    parsed = next((p for p in map(parse_answer, reversed(record.raw_outputs)) if p), None)
    if parsed is None:
        return EvalResult(name="citations_valid", applicable=False)
    user = corpus.users[item.user]
    retrieved = {d.id for d in record.retrieved}
    problems = []
    for doc_id in parsed.citations:
        doc = corpus.docs.get(doc_id)
        if doc is None:
            problems.append(f"{doc_id}: not in corpus")
        elif not user.can_read(doc.access):
            problems.append(f"{doc_id}: not readable by {user.name}")
        elif doc_id not in retrieved:
            problems.append(f"{doc_id}: not retrieved")
    return EvalResult(name="citations_valid", applicable=True, passed=not problems,
                      detail="; ".join(problems) or f"{len(parsed.citations)} ok")


Evaluator = Callable[[DatasetItem, AnswerRecord, Corpus], EvalResult]
CODE_EVALUATORS: list[Evaluator] = [
    schema_valid, refusal_correct, facts_recall, forbidden_absent, citations_valid,
]


def run_code_evaluators(item: DatasetItem, record: AnswerRecord,
                        corpus: Corpus) -> list[EvalResult]:
    return [evaluate(item, record, corpus) for evaluate in CODE_EVALUATORS]


def answer_passed(results: list[EvalResult]) -> bool:
    applicable = [r for r in results if r.applicable]
    return bool(applicable) and all(r.passed for r in applicable)
