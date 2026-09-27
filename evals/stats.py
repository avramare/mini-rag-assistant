"""Report and compare eval runs.

Sample size rule (see PLAN.md Phase 4): repeats of one question are not independent samples.
An item's pass rate is the mean over its repeats; overall and per-category rates are means over
ITEMS, and n is the number of items. Wilson intervals and pass->fail flips are added in Phase 4.

A results file keeps one evaluation per judge prompt version. Both commands use the latest one;
`report --judge <sha prefix>` picks another.

Usage:
    uv run python -m evals.stats report results/<run>.json [--judge <sha prefix>]
    uv run python -m evals.stats compare results/<baseline>.json results/<candidate>.json
    uv run python -m evals.stats noise results/<run>.json [--judge <sha prefix>]
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


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95 % with z = 1.96) for a rate of `successes` out of `n`.

    Unlike the textbook p +- z*sqrt(p(1-p)/n), it stays inside [0, 1] and does not collapse to a
    zero-width interval at 0 % or 100 %, which matters with n = 30 items and rates near 100 %.
    `successes` may be fractional: for an item-level rate it is the sum of item pass rates, and
    each item counts as one trial. That is conservative: a mean of fractions varies less than
    p(1-p)/n, so the interval is, if anything, too wide."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap(deltas: list[float], n_resamples: int = 10_000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-item deltas and its 95 % percentile bootstrap interval.

    Resamples ITEMS with replacement, keeping each item's pair (base, candidate) together. Pairing
    matters because items differ in difficulty far more than a prompt change moves them; the
    per-item delta cancels that. Fixed seed: the same file always prints the same interval."""
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, (n_resamples, values.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def judge_label(key: str, evaluation: Evaluation) -> str:
    judge = evaluation.judge
    return f"judge {judge['model']} prompt {key}" if judge else "judge none"


def judge_cost(evaluation: Evaluation) -> str:
    """Per judged answer: wall clock over all its attempts, and tokens of the reply used."""
    judged = [r for ev in evaluation.answers for r in ev.results
              if r.name == "judge_faithfulness" and r.applicable]
    durations = [r.duration_ms for r in judged if r.duration_ms is not None]
    tokens = [r.output_tokens for r in judged if r.output_tokens is not None]
    if not durations:
        return "judge duration not recorded (evaluated before it was)"
    p50, p95 = np.percentile(durations, [50, 95])
    line = (f"judge duration p50 {p50 / 1000:.1f}s  p95 {p95 / 1000:.1f}s  "
            f"({len(durations)} answers, incl. retries)")
    if tokens:
        t50, t95 = np.percentile(tokens, [50, 95])
        line += f"  output tokens p50 {t50:.0f}  p95 {t95:.0f}  max {max(tokens)}"
    return line


# A model already in memory loads in well under a second; more means Ollama (re)loaded it, e.g.
# after its idle timeout. Worth seeing in a night run: the reload is cost, not model speed.
RELOAD_MS = 1000.0


def time_split(label: str, prompt_ms: list[float | None], reply_ms: list[float | None],
               load_ms: list[float | None]) -> list[str]:
    """Where a model's time went, from Ollama's own durations: reading the prompt vs writing the
    reply. On CPU the prompt can dominate; that decides whether a shorter reply saves anything."""
    prompt = [x for x in prompt_ms if x is not None]
    reply = [x for x in reply_ms if x is not None]
    if not prompt or not reply:
        return [f"{label} time split not recorded (run before it was)"]
    share = sum(prompt) / (sum(prompt) + sum(reply))
    lines = []
    for name, values in (("prompt eval", prompt), ("reply", reply)):
        p50, p95 = np.percentile(values, [50, 95])
        lines.append(f"{label} {name:<11} p50 {p50 / 1000:5.1f}s  p95 {p95 / 1000:5.1f}s")
    reloads = sum(x > RELOAD_MS for x in load_ms if x is not None)
    lines.append(f"{label} prompt eval share of model time {_pct(share).strip()}  "
                 f"model loads > {RELOAD_MS / 1000:g}s: {reloads}")
    return lines


def filter_label(run_filter: dict) -> str:
    return "  ".join(f"{k} {','.join(v)}" for k, v in run_filter.items() if v)


def report(run: RunResults, judge: str | None = None) -> str:
    c, answers = run.config, run.answers
    key, evaluation = finished(run, judge)
    others = sorted(k for k in run.evaluations if k != key)
    generated, evaluated = c["dataset"]["sha256"][:12], evaluation.dataset_sha256[:12]
    lines = [
        f"Run {c['name']}  as_of {c['as_of']}  repeats {c['repeats']}",
        # Two hashes: editing facts or expected docs after generation re-grades the saved
        # answers (DECISIONS #22), so they can be graded against another file than generated from.
        f"dataset {c['dataset']['name']}  generated with {generated}  evaluated with {evaluated}"
        + ("" if generated == evaluated else "  (grading fields changed after generation)"),
        f"gen {c['assistant']['gen_model']}  embed {c['assistant']['embed_model']}  "
        f"k {c['assistant']['k']}  num_ctx {c['assistant']['num_ctx']}  "
        f"prompt {c['assistant']['system_prompt_sha256'][:12]}",
        *([f"FILTERED dev run: {filter_label(c['filter'])}; not comparable with full runs"]
          if c.get("filter") else []),
        *([f"frozen config {c['frozen']['path']} ({c['frozen']['sha256'][:12]})"]
          if c.get("frozen") else []),
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
    judge_errors = judge_timeouts = judge_truncated = 0
    for ev in evaluation.answers:
        for r in ev.results:
            if r.applicable:
                applicable[r.name] += 1
                failed[r.name] += not r.passed
                judge_errors += r.detail.startswith("judge_error")
                judge_timeouts += r.attempt_errors.count("timeout")
                judge_truncated += r.attempt_errors.count("truncated")
                if not r.gating:
                    diagnostic.add(r.name)
    lines += ["", f"{'evaluator':<20}{'applicable':>11}{'failed':>8}"]
    lines += [f"{name:<20}{applicable[name]:>11}{failed[name]:>8}"
              + ("  diagnostic, not in pass rate" if name in diagnostic else "")
              for name in applicable]
    if evaluation.judge is not None:
        # A timeout is retried once; only a second failure becomes a judge_error.
        lines.append(f"judge errors (counted as fails) {judge_errors}  "
                     f"judge timeouts (incl. retried) {judge_timeouts}  "
                     f"cut off at output cap (incl. retried) {judge_truncated}")
        lines.append(judge_cost(evaluation))
        judged = [r for ev in evaluation.answers for r in ev.results
                  if r.name == "judge_faithfulness" and r.applicable]
        lines += time_split("judge", [r.prompt_eval_ms for r in judged],
                            [r.eval_ms for r in judged], [r.load_ms for r in judged])

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
        *time_split("generation", [x for a in answers for x in a.prompt_eval_ms],
                    [x for a in answers for x in a.eval_ms],
                    [x for a in answers for x in a.load_ms]),
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
    if a.config.get("filter") != b.config.get("filter"):
        # Checked before the item hash so the message says why: a dev subset is not the dataset.
        label = [filter_label(r.config["filter"]) if r.config.get("filter") else "full run"
                 for r in (a, b)]
        raise NotComparableError(f"different item filters: {label[0]} vs {label[1]}; "
                                 "a filtered dev run covers other items than a full run")
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
    lines.append("(categories: deltas only; 2-8 items per category are too few for an interval)")

    for label, run in (("base", a), ("cand", b)):
        rate, n = overall_rate(run)
        low, high = wilson(rate * n, n)
        lines.append(f"{label} item-level {_pct(rate).strip()}  Wilson 95% "
                     f"[{_pct(low).strip()}, {_pct(high).strip()}]  (n = {n} items)")

    # Paired: the same items in both runs, so each item is compared with itself.
    rates_a, rates_b = item_pass_rates(a), item_pass_rates(b)
    categories = {r.item_id: r.category for r in a.answers}
    deltas = {item: rates_b[item] - rates_a[item] for item in sorted(rates_a)}
    mean, low, high = paired_bootstrap(list(deltas.values()))
    verdict = ("interval includes 0: no evidence of a change" if low <= 0 <= high else
               "interval excludes 0: the change is larger than item sampling explains")
    lines += ["", f"Paired over {len(deltas)} items: mean delta {100 * mean:+.1f} pts, 95% "
              f"bootstrap [{100 * low:+.1f}, {100 * high:+.1f}] (10000 resamples, seed 0); "
              + verdict]
    moved = sorted((d, item) for item, d in deltas.items() if d != 0)
    if moved:
        lines.append("Items that moved (worst first):")
        for delta, item in moved:
            flip = ("  pass->fail" if rates_a[item] == 1 and rates_b[item] < 1 else
                    "  fail->pass" if rates_a[item] < 1 and rates_b[item] == 1 else "")
            lines.append(f"  {item:<11}{categories[item]:<18}{_pct(rates_a[item])} -> "
                         f"{_pct(rates_b[item])}  {100 * delta:+.1f}{flip}")
    else:
        lines.append("No item moved.")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    try:
        if len(args) == 2 and args[0] == "report":
            print(report(load(Path(args[1]))))
        elif len(args) == 4 and args[0] == "report" and args[2] == "--judge":
            print(report(load(Path(args[1])), args[3]))
        elif args[:1] == ["noise"] and len(args) in (2, 4) and args[2:3] in ([], ["--judge"]):
            from evals.noise import noise  # imports this module; imported late to avoid a cycle
            print(noise(load(Path(args[1])), args[3] if len(args) == 4 else None))
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
