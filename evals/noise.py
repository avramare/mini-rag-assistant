"""Noise report for one run of the same config with R repeats (Phase 4): `stats noise <run>`.

Two noise sources, measured separately:
- generation: the same question answered R times (each repeat judged once, so this spread also
  contains judge noise);
- judge: one repeat's answers re-judged K times by the same judge (`run_experiment rejudge`).

The sample size is the number of ITEMS, not items x repeats (PLAN Phase 4): repeats of one
question are not independent. Intervals are over items.
"""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from evals.dataset import SAFETY_CATEGORIES
from evals.evaluators import is_safety_failure
from evals.results import AnswerEvaluation, RunResults
from evals.stats import _pct, finished, judge_label, paired_bootstrap, wilson

JUDGE = "judge_faithfulness"


def pass_matrix(run: RunResults, judge: str | None = None) -> dict[str, list[bool]]:
    """item -> pass/fail per repeat, in repeat order."""
    by_item: dict[str, dict[int, bool]] = defaultdict(dict)
    for ev in finished(run, judge)[1].answers:
        by_item[ev.item_id][ev.repeat] = ev.passed
    return {item: [reps[r] for r in sorted(reps)] for item, reps in sorted(by_item.items())}


def pattern(passes: list[bool]) -> str:
    return "".join("P" if p else "F" for p in passes)


def group_rate(passes: dict[str, list[bool]], group: tuple[int, ...]) -> float:
    """Item-level pass rate over the repeats in `group` (0-based indexes)."""
    return float(np.mean([np.mean([p[r] for r in group]) for p in passes.values()]))


def split_delta(passes: dict[str, list[bool]], a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """|rate(a) - rate(b)|. Absolute: a split into two groups has no order, so which side is
    "baseline" is arbitrary and the sign of the difference means nothing."""
    return abs(group_rate(passes, a) - group_rate(passes, b))


def splits(repeats: int, size: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every unordered pair of disjoint repeat groups of `size`: {a, b} counted once."""
    out = []
    for a in combinations(range(repeats), size):
        rest = [r for r in range(repeats) if r not in a]
        for b in combinations(rest, size):
            if a < b:  # the same pair with the sides swapped is not another split
                out.append((a, b))
    return out


def pairwise_flip_rate(passes: dict[str, list[bool]]) -> float:
    """Mean over repeat pairs of the share of items whose verdict differs between the two."""
    repeats = len(next(iter(passes.values())))
    pairs = list(combinations(range(repeats), 2))
    if not pairs:
        return float("nan")
    return float(np.mean([np.mean([p[a] != p[b] for p in passes.values()]) for a, b in pairs]))


def _interval(low: float, high: float) -> str:
    return f"[{_pct(low).strip()}, {_pct(high).strip()}]"


def generation_noise(run: RunResults, key: str) -> list[str]:
    evaluation = run.evaluations[key]
    passes = pass_matrix(run, key)
    n, repeats = len(passes), run.config["repeats"]
    records = {(a.item_id, a.repeat): a for a in run.answers}
    categories = {a.item_id: a.category for a in run.answers}
    questions = {a.item_id: a.question for a in run.answers}
    graded: dict[tuple[str, int], AnswerEvaluation] = {
        (ev.item_id, ev.repeat): ev for ev in evaluation.answers}

    per_repeat = [group_rate(passes, (r,)) for r in range(repeats)]
    rates = {item: float(np.mean(p)) for item, p in passes.items()}
    overall = float(np.mean(list(rates.values())))
    unstable = [i for i, r in rates.items() if 0 < r < 1]
    lines = [
        f"GENERATION NOISE  n = {n} items x {repeats} repeats; the {n * repeats} answers are "
        "not independent samples, intervals are over items",
        "",
        "Pass rate per repeat (each = one Langfuse dataset run): "
        + "  ".join(f"r{r + 1} {_pct(x).strip()}" for r, x in enumerate(per_repeat)),
        f"  mean {_pct(np.mean(per_repeat)).strip()}  "
        + (f"SD {100 * np.std(per_repeat, ddof=1):.1f} pts  " if repeats > 1 else "SD n/a  ")
        +         f"min {_pct(min(per_repeat)).strip()}  max {_pct(max(per_repeat)).strip()}  "
        f"range {100 * (max(per_repeat) - min(per_repeat)):.1f} pts",
        f"Item-level pass rate {_pct(overall).strip()}  Wilson 95% "
        f"{_interval(*wilson(overall * n, n))}  (n = {n} items)",
        f"Items: stable pass {sum(r == 1 for r in rates.values())}  "
        f"stable fail {sum(r == 0 for r in rates.values())}  unstable {len(unstable)}  "
        f"(stable = no other verdict seen in {repeats} repeats, not 'never')",
        (f"Pairwise flip rate: {_pct(pairwise_flip_rate(passes)).strip()} of items change "
         "verdict between two repeats, on average" if repeats > 1
         else "Pairwise flip rate: n/a with 1 repeat"),
        "",
        f"{'category':<18}{'items':>6}{'pass rate':>11}  {'Wilson 95%':<17}{'unstable':>9}",
    ]
    by_category: dict[str, list[str]] = defaultdict(list)
    for item in passes:
        by_category[categories[item]].append(item)
    for category, items in sorted(by_category.items()):
        rate = float(np.mean([rates[i] for i in items]))
        flag = "  safety" if category in SAFETY_CATEGORIES else ""
        lines.append(f"{category:<18}{len(items):>6}{_pct(rate):>11}  "
                     f"{_interval(*wilson(rate * len(items), len(items))):<17}"
                     f"{sum(i in unstable for i in items):>9}{flag}")

    def failed_evaluators(item: str) -> str:
        counts = Counter(r.name for rep in range(1, repeats + 1)
                         for r in graded[(item, rep)].results
                         if r.applicable and r.gating and not r.passed)
        return ", ".join(f"{name} x{c}" for name, c in sorted(counts.items())) or "-"

    def outcomes(item: str) -> str:
        """Per repeat: what the user saw (answered, a refusal and why, or an error)."""
        out = []
        for rep in range(1, repeats + 1):
            a = records[(item, rep)]
            out.append(a.error or (f"refused:{a.refusal_reason}" if a.refusal_reason
                                   else "answered"))
        return " ".join(out)

    versioning = [i for i in passes if categories[i] == "versioning"]
    if versioning:
        lines += ["", "Versioning items (forbidden_absent failed = said a value not in force):"]
        for item in versioning:
            said_other = sum(not r.passed for rep in range(1, repeats + 1)
                             for r in graded[(item, rep)].results
                             if r.name == "forbidden_absent" and r.applicable)
            lines.append(f"  {item:<11}{pattern(passes[item])}  {_pct(rates[item]).strip():>6}  "
                         f"not-in-force value x{said_other}/{repeats}  "
                         f"as_of {records[(item, 1)].as_of}  {questions[item]}")

    if unstable:
        lines += ["", "Unstable items (0 < pass rate < 1): pattern per repeat, failed "
                  "evaluators, outcome per repeat"]
        for item in sorted(unstable, key=lambda i: (categories[i], i)):
            lines.append(f"  {item:<11}{categories[item]:<18}{pattern(passes[item])}  "
                         f"{_pct(rates[item]).strip():>6}  {failed_evaluators(item)}  "
                         f"| {outcomes(item)}")

    safety = [(item, rep, r.name) for (item, rep), ev in sorted(graded.items())
              for r in ev.results if is_safety_failure(r)]
    per_rep = Counter(rep for _, rep, _ in safety)
    lines += ["", "Safety evaluators with no config change (forbidden_absent, citations_valid, "
              "refusal_correct where a refusal was expected):",
              "  failures per repeat: " + "  ".join(f"r{r} {per_rep[r]}"
                                                    for r in range(1, repeats + 1))]
    if safety:
        lines.append("  ! Phase 5 zero tolerance would fail on noise alone: "
                     + ", ".join(f"{i} r{rep} {name}" for i, rep, name in safety))
    else:
        lines.append(f"  none in {repeats} repeats")
    return lines


def split_distribution(run: RunResults, key: str) -> list[str]:
    """Input for the Phase 5 noise margin: how far the item-level rate of g repeats moves against
    another g repeats of the SAME config."""
    passes = pass_matrix(run, key)
    repeats = run.config["repeats"]
    lines = ["", "NO-CHANGE SPLITS  |item-level rate(group A) - rate(group B)| over every split of "
             "the repeats into two disjoint groups of g (unordered, so no sign)",
             f"{'g':>3}{'splits':>8}{'median':>9}{'p95':>9}{'max':>9}  (percentage points)"]
    if repeats < 2:
        return lines + ["  needs at least 2 repeats"]
    for size in range(1, repeats // 2 + 1):
        deltas = [split_delta(passes, a, b) for a, b in splits(repeats, size)]
        median, p95 = np.percentile(deltas, [50, 95])
        lines.append(f"{size:>3}{len(deltas):>8}{100 * median:>9.1f}{100 * p95:>9.1f}"
                     f"{100 * max(deltas):>9.1f}")
    half = repeats // 2
    first, second = tuple(range(half)), tuple(range(half, 2 * half))
    deltas = [float(np.mean([p[r] for r in second]) - np.mean([p[r] for r in first]))
              for p in passes.values()]
    mean, low, high = paired_bootstrap(deltas)
    lines.append(f"Paired bootstrap, no change: r1-{half} vs r{half + 1}-{2 * half}: mean delta "
                 f"{100 * mean:+.1f} pts, 95% [{100 * low:+.1f}, {100 * high:+.1f}] "
                 f"(n = {len(deltas)} items, 10000 resamples, seed 0)")
    lines.append("'worst drop' for Phase 5 = max; with few splits it is a rough estimate")
    return lines


def judge_noise(run: RunResults, key: str) -> list[str]:
    samples = [s for s in run.judge_samples.get(key, []) if s.finished_at is not None]
    unfinished = len(run.judge_samples.get(key, [])) - len(samples)
    if not samples:
        return ["", "JUDGE NOISE  no finished re-judge samples; run "
                "`run_experiment rejudge --name <run> --samples K`"]
    repeat = samples[0].repeat
    graded = {(ev.item_id, ev.repeat): ev for ev in run.evaluations[key].answers
              if ev.repeat == repeat}
    sampled = defaultdict(list)
    for s in samples:
        for a in s.answers:
            sampled[(a.item_id, a.repeat)].append(a.result)

    flips, disagreement, spread, decisive = [], [], 0, 0
    judged = 0
    for answer, ev in sorted(graded.items()):
        original = next(r for r in ev.results if r.name == JUDGE)
        if not original.applicable:
            continue
        judged += 1
        results = [original, *sampled[answer]]
        verdicts = [bool(r.passed) for r in results]
        scores = [r.value for r in results]
        pairs = list(combinations(verdicts, 2))
        disagreement.append(np.mean([a != b for a, b in pairs]))
        spread += len(set(scores)) > 1
        if len(set(verdicts)) > 1:
            others_pass = all(r.passed for r in ev.results
                              if r.applicable and r.gating and r.name != JUDGE)
            decisive += others_pass
            flips.append((answer, scores, others_pass))
    k = len(samples)
    lines = ["", f"JUDGE NOISE  repeat r{repeat} re-judged {k}x by the same judge; verdicts "
             f"per answer = evaluation + {k} samples = {k + 1}"
             + (f"  ({unfinished} unfinished sample(s) ignored)" if unfinished else "")]
    if not judged:
        return lines + ["  no judged answers in this repeat"]
    lines += [
        f"Answers whose verdict flips: {len(flips)} of {judged} judged "
        f"({_pct(len(flips) / judged).strip()}, Wilson 95% "
        f"{_interval(*wilson(len(flips), judged))}, n = {judged} judged answers)",
        f"Mean pairwise verdict disagreement: {_pct(float(np.mean(disagreement))).strip()}  "
        f"answers with more than one distinct score: {spread}",
        f"Flips that change item_pass (all other gating evaluators passed): {decisive}",
    ]
    for (item, rep), scores, others_pass in flips:
        # Scores only: a reason can quote a restricted answer.
        lines.append(f"  {item:<11}r{rep}  scores {' '.join(f'{s:g}' if s is not None else 'err'
                                                         for s in scores)}"
                     + ("  item_pass flips" if others_pass else ""))
    return lines


def noise(run: RunResults, judge: str | None = None) -> str:
    key, evaluation = finished(run, judge)
    lines = [f"Run {run.config['name']}  {judge_label(key, evaluation)}  "
             f"repeats {run.config['repeats']}"]
    if run.config.get("frozen"):
        lines.append(f"frozen config {run.config['frozen']['path']} "
                     f"({run.config['frozen']['sha256'][:12]})")
    if run.config.get("filter"):
        lines.append("FILTERED dev run: numbers cover the selected items only")
    lines.append("")
    lines += generation_noise(run, key)
    lines += split_distribution(run, key)
    lines += judge_noise(run, key)
    return "\n".join(lines)
