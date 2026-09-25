"""Report and compare eval runs.

Sample size rule (see PLAN.md Phase 4): repeats of one question are not independent samples.
An item's pass rate is the mean over its repeats; overall and per-category rates are means over
ITEMS, and n is the number of items. Wilson intervals and pass->fail flips are added in Phase 4.

A results file keeps one evaluation per judge prompt version. Both commands use the latest one;
`report --judge <sha prefix>` picks another.

Usage:
    uv run python -m evals.stats report results/<run>.json [--judge <sha prefix>]
    uv run python -m evals.stats compare results/<baseline>.json results/<candidate>.json
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evals.dataset import SAFETY_CATEGORIES
from evals.results import Evaluation, RunResults, load


class NotComparableError(ValueError):
    pass


def finished(run: RunResults, judge: str | None = None) -> tuple[str, Evaluation]:
    """Rates over an unfinished evaluation would silently cover only some of the answers."""
    key, evaluation = run.evaluation(judge)
    if evaluation.finished_at is None:
        raise ValueError(f"evaluation by judge {key} is unfinished ({len(evaluation.answers)} "
                         f"of {len(run.answers)} answers); finish it with `evaluate --resume`")
    return key, evaluation


def item_pass_rates(run: RunResults, judge: str | None = None) -> dict[str, float]:
    per_item: dict[str, list[bool]] = defaultdict(list)
    for ev in finished(run, judge)[1].answers:
        per_item[ev.item_id].append(ev.passed)
    return {item: sum(passes) / len(passes) for item, passes in per_item.items()}


def category_rates(run: RunResults, judge: str | None = None) -> dict[str, tuple[float, int]]:
    """category -> (mean of item pass rates, number of items)."""
    category = {a.item_id: a.category for a in run.answers}
    by_category: dict[str, list[float]] = defaultdict(list)
    for item, rate in item_pass_rates(run, judge).items():
        by_category[category[item]].append(rate)
    return {c: (sum(r) / len(r), len(r)) for c, r in sorted(by_category.items())}


def overall_rate(run: RunResults, judge: str | None = None) -> tuple[float, int]:
    rates = list(item_pass_rates(run, judge).values())
    return sum(rates) / len(rates), len(rates)


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def judge_label(key: str, evaluation: Evaluation) -> str:
    judge = evaluation.judge
    return f"judge {judge['model']} prompt {key}" if judge else "judge none"


def report(run: RunResults, judge: str | None = None) -> str:
    c, answers = run.config, run.answers
    key, evaluation = finished(run, judge)
    others = sorted(k for k in run.evaluations if k != key)
    lines = [
        f"Run {c['name']}  dataset {c['dataset']['name']} ({c['dataset']['sha256'][:12]})  "
        f"as_of {c['as_of']}  repeats {c['repeats']}",
        f"gen {c['assistant']['gen_model']}  embed {c['assistant']['embed_model']}  "
        f"k {c['assistant']['k']}  num_ctx {c['assistant']['num_ctx']}  "
        f"prompt {c['assistant']['system_prompt_sha256'][:12]}",
        f"{judge_label(key, evaluation)}  evaluated {evaluation.evaluated_at:%Y-%m-%d %H:%M} UTC"
        + (f"  (also stored: {', '.join(others)}; pick with --judge)" if others else ""),
        "",
    ]

    rate, n = overall_rate(run, key)
    lines += [f"Pass rate (item-level, n = {n} items): {_pct(rate)}", "",
              f"{'category':<18}{'items':>6}{'pass rate':>11}"]
    for category, (rate, n) in category_rates(run, key).items():
        flag = "  safety" if category in SAFETY_CATEGORIES else ""
        lines.append(f"{category:<18}{n:>6}{_pct(rate):>11}{flag}")

    applicable, failed = Counter(), Counter()
    diagnostic: set[str] = set()
    judge_errors = judge_timeouts = 0
    for ev in evaluation.answers:
        for r in ev.results:
            if r.applicable:
                applicable[r.name] += 1
                failed[r.name] += not r.passed
                judge_errors += r.detail.startswith("judge_error")
                judge_timeouts += r.attempt_errors.count("timeout")
                if not r.gating:
                    diagnostic.add(r.name)
    lines += ["", f"{'evaluator':<20}{'applicable':>11}{'failed':>8}"]
    lines += [f"{name:<20}{applicable[name]:>11}{failed[name]:>8}"
              + ("  diagnostic, not in pass rate" if name in diagnostic else "")
              for name in applicable]
    if evaluation.judge is not None:
        # A timeout is retried once; only a second failure becomes a judge_error.
        lines.append(f"judge errors (counted as fails) {judge_errors}  "
                     f"judge timeouts (incl. retried) {judge_timeouts}")

    refusals = Counter(a.refusal_reason or "-" for a in answers)
    errors = Counter(a.error or "-" for a in answers)
    durations = [d for a in answers for d in a.durations_ms if d is not None]
    p50, p95 = np.percentile(durations, [50, 95]) if durations else (float("nan"),) * 2
    lines += [
        "",
        f"answers {len(answers)}  refusal_reason {dict(refusals)}  error {dict(errors)}",
        f"retried {sum(a.attempts > 1 for a in answers)}  "
        f"invalid outputs {sum(a.invalid_outputs for a in answers)}  "
        f"truncation_risk {sum(a.truncation_risk for a in answers)}",
        f"generation duration p50 {p50 / 1000:.1f}s  p95 {p95 / 1000:.1f}s  "
        f"({len(durations)} calls, linear interpolation)",
    ]

    categories = {a.item_id: a.category for a in answers}
    failing = defaultdict(Counter)
    # item -> [repeats where retrieval missed an expected doc, repeats where it was checked]
    retrieval: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ev in evaluation.answers:
        for r in ev.results:
            if r.applicable and not r.passed and r.gating:
                failing[ev.item_id][r.name] += 1
            if r.applicable and r.name == "retrieval_recall":
                retrieval[ev.item_id][0] += not r.passed
                retrieval[ev.item_id][1] += 1
    rates = item_pass_rates(run, key)
    if failing:
        # The retrieval note separates the two failure sources: a missed doc is a retrieval
        # failure; all expected docs in the context means generation did not use them.
        lines += ["", "Failing items (pass rate, failed evaluator x repeats, retrieval):"]
        for item in sorted(failing, key=lambda i: (categories[i], i)):
            reasons = ", ".join(f"{name} x{count}" for name, count in failing[item].items())
            missed, checked = retrieval.get(item, (0, 0))
            where = ("retrieval not checked" if not checked else
                     f"retrieval missed x{missed}" if missed else "retrieval ok")
            lines.append(f"  {item:<11}{categories[item]:<18}{_pct(rates[item])}  {reasons}"
                         f"  | {where}")
    return "\n".join(lines)


def check_comparable(a: RunResults, b: RunResults) -> list[str]:
    """Raises when the runs measure different things; returns the config differences that ARE the
    experiment (prompt, model, k, ...), so the reader sees what changed. Uses each run's latest
    evaluation."""
    if a.config["as_of"] != b.config["as_of"]:
        raise NotComparableError(f"different as_of: {a.config['as_of']} vs {b.config['as_of']}; "
                                 "the correct answer to versioning items differs")
    if a.config["dataset"]["generation_key"] != b.config["dataset"]["generation_key"]:
        raise NotComparableError("different dataset items (questions, users or dates)")
    if not a.evaluations or not b.evaluations:
        raise NotComparableError("both runs need an evaluation")
    (key_a, ev_a), (key_b, ev_b) = finished(a), finished(b)
    if ev_a.dataset_sha256 != ev_b.dataset_sha256:
        raise NotComparableError("graded against different dataset facts; re-evaluate both")
    diffs = [f"{key}: {a.config['assistant'][key]} -> {b.config['assistant'][key]}"
             for key in a.config["assistant"]
             if a.config["assistant"][key] != b.config["assistant"].get(key)]
    if (ev_a.judge or {}) != (ev_b.judge or {}):
        diffs.append(f"judge: {judge_label(key_a, ev_a)} -> {judge_label(key_b, ev_b)}")
    return diffs


def compare(a: RunResults, b: RunResults) -> str:
    diffs = check_comparable(a, b)
    lines = [f"{a.config['name']} ({judge_label(*a.evaluation())}) -> "
             f"{b.config['name']} ({judge_label(*b.evaluation())})",
             "config changes: " + ("; ".join(diffs) if diffs else "none (noise check)"), "",
             f"{'category':<18}{'items':>6}{'base':>9}{'cand':>9}{'delta':>9}"]
    ca, cb = category_rates(a), category_rates(b)
    rows = [("overall", overall_rate(a), overall_rate(b))]
    rows += [(c, ca[c], cb[c]) for c in ca]
    for name, (ra, n), (rb, _) in rows:
        lines.append(f"{name:<18}{n:>6}{_pct(ra):>9}{_pct(rb):>9}{100 * (rb - ra):>+8.1f}")
    lines.append("\nNo intervals yet (Phase 4): do not read small deltas as real changes.")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    try:
        if len(args) == 2 and args[0] == "report":
            print(report(load(Path(args[1]))))
        elif len(args) == 4 and args[0] == "report" and args[2] == "--judge":
            print(report(load(Path(args[1])), args[3]))
        elif len(args) == 3 and args[0] == "compare":
            print(compare(load(Path(args[1])), load(Path(args[2]))))
        else:
            print(__doc__.split("Usage:")[1])
            return 2
    except NotComparableError as exc:
        print(f"[fail] not comparable: {exc}")
        return 1
    except ValueError as exc:  # no evaluation, or an unknown --judge version
        print(f"[fail] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
