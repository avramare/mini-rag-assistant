"""Regression gate (Phase 5): is a candidate run worse than the committed baseline?

    uv run python -m evals.stats gate evals/baseline.json results/<candidate>.json   # exit 0/1
    uv run python -m evals.gate baseline results/<run>.json --out evals/baseline.json

`baseline.json` is built once from a noise run and committed only with Marko's approval; the gate
reads it and never writes it. It is self-contained (per-item rates, noise numbers, provenance), so
the gate does not need the baseline's results file.

A candidate is refused (exit 1, nothing judged) unless it was measured with the same INSTRUMENT
as the baseline (evals.instrument): same questions, dates, grading facts, documents, users, judge
and evaluator code, not filtered, a finished evaluation, a clean tree, and exactly
`candidate_repeats` repeats. The SYSTEM under test may differ; the report starts with what did.

Rules, all evaluated and reported even when an earlier one fails:
1. Safety: any safety failure (`is_safety_failure`, by failure type) in any candidate repeat.
2. Overall: mean over items of (candidate item rate - baseline item rate) below -margin. The
   paired bootstrap interval is shown as information: with n = 30 items it is wide, and the
   margin was set from measured no-change noise instead (DECISIONS 38, 40).
3. Stable-pass regression: an item that passed every baseline repeat and failed every candidate
   repeat. Catches one broken item that the overall mean (1 item = 3.3 pts) would hide.
4. Per category and every item that dropped: reported, never gating (2-8 items per category).
"""

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from evals.evaluators import is_safety_failure
from evals.instrument import (
    changed_since,
    differences,
    docs_sha256,
    evaluator_sha256,
    git_info,
    instrument_of,
    system_of,
    users_sha256,
)
from evals.noise import judge_flips, no_change_bootstrap, pass_matrix, pattern, split_table
from evals.results import RunResults, load
from evals.stats import _pct, category_rates, finished, paired_bootstrap, wilson

PROVISIONAL = "provisional"
GATE_PARAMS = ("candidate_repeats", "overall_margin_pts")
# Comparing a mean of fractions with -margin: a delta that equals the margin passes, whatever
# the float rounding of 1/3 steps.
EPS_PTS = 1e-9


class GateRefused(ValueError):
    """The candidate (or the baseline file) cannot be judged; every reason is listed."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("\n  ".join(["gate refused:", *reasons]))
        self.reasons = reasons


def _round(x: float) -> float:
    return round(float(x), 6)


def pattern_rate(passes: str) -> float:
    """Exact item rate from its stored pattern ("PPFPFP"); the stored `rate` is rounded for
    reading, and a rounded rate would move a delta that sits exactly on the margin."""
    return passes.count("P") / len(passes)


# --- baseline ---------------------------------------------------------------------------------

def build_baseline(run: RunResults, *, fallback_instrument: dict[str, str],
                   changed: list[str], note: str) -> dict[str, Any]:
    """`fallback_instrument`: docs/users/evaluator hashes of today's files, used for what the run
    did not record (runs before Phase 5). Only valid if none of those files changed since the run
    was generated: `changed` lists the ones that did, and any refuses."""
    reasons = []
    if changed:
        reasons.append("instrument files changed since the run's commit "
                       f"{run.config.get('git_commit')}: {', '.join(changed)}; today's hashes "
                       "would not describe the run")
    if run.config.get("filter"):
        reasons.append("a filtered dev run cannot be a baseline")
    if run.config.get("git_dirty") is not False:
        reasons.append(f"run generated on a dirty or unknown tree "
                       f"(git_dirty {run.config.get('git_dirty')})")
    if reasons:
        raise GateRefused(reasons)
    key, _ = finished(run)
    passes = pass_matrix(run, key)
    categories = {a.item_id: a.category for a in run.answers}
    rates = {item: float(np.mean(p)) for item, p in passes.items()}
    n = len(rates)
    overall = float(np.mean(list(rates.values())))
    per_repeat = [float(np.mean([p[r] for p in passes.values()]))
                  for r in range(run.config["repeats"])]
    half, mean, low, high = no_change_bootstrap(passes)
    judge = judge_flips(run, key)
    safety = sum(is_safety_failure(r, categories[ev.item_id])
                 for ev in run.evaluations[key].answers for r in ev.results)
    c = run.config
    return {
        "status": PROVISIONAL,
        "status_note": note,
        "approved_by": None,
        "gate": {name: None for name in GATE_PARAMS},
        "provenance": {"run": c["name"], "created_at": c["created_at"],
                       "git_commit": c.get("git_commit"), "frozen": c.get("frozen"),
                       "repeats": c["repeats"], "built_at": datetime.now(UTC).isoformat(
                           timespec="seconds")},
        "instrument": instrument_of(run, key, fallback_instrument),
        "system": system_of(run),
        "rates": {
            "overall": {"rate": _round(overall), "n_items": n,
                        "wilson_95": [_round(x) for x in wilson(overall * n, n)]},
            "categories": {cat: {"rate": _round(rate), "n_items": count}
                           for cat, (rate, count) in category_rates(run, key).items()},
            "items": {item: {"category": categories[item], "rate": _round(rates[item]),
                             "pattern": pattern(passes[item])} for item in passes},
        },
        "noise": {
            "per_repeat_rates": [_round(x) for x in per_repeat],
            "sd_pts": _round(100 * np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else None,
            "range_pts": _round(100 * (max(per_repeat) - min(per_repeat))),
            "no_change_splits": [{k: _round(v) if isinstance(v, float) else v
                                  for k, v in row.items()} for row in split_table(passes)],
            "no_change_bootstrap": {"halves": f"r1-{half} vs r{half + 1}-{2 * half}",
                                    "mean_pts": _round(100 * mean),
                                    "ci95_pts": [_round(100 * low), _round(100 * high)]},
            "judge": None if judge is None else {
                "repeat": judge.repeat, "samples": judge.samples, "judged": judge.judged,
                "flipped": len(judge.flips)},
            "safety_failures": safety,
        },
    }


# --- gate -------------------------------------------------------------------------------------

_HASH = re.compile(r"^(sha256:)?[0-9a-f]{40,64}$")


def _short(value: Any) -> Any:
    """Hashes cut to 12 characters for the report; everything else unchanged."""
    if isinstance(value, dict):
        return {k: _short(v) for k, v in value.items()}
    if isinstance(value, str) and _HASH.match(value):
        return value[:19] if value.startswith("sha256:") else value[:12]
    return value


def check_candidate(baseline: dict[str, Any], run: RunResults) -> str:
    """Every reason the candidate cannot be judged against the baseline; returns the evaluation
    key to use (the baseline's judge version)."""
    params = baseline.get("gate") or {}
    missing = [p for p in GATE_PARAMS if params.get(p) is None]
    if missing:
        raise GateRefused([f"baseline gate parameters not set: {', '.join(missing)} (Marko "
                           "fills them in baseline.json)"])
    key = baseline["instrument"]["judge_key"]
    if key not in run.evaluations:
        raise GateRefused([f"candidate has no evaluation by the baseline judge {key} "
                           f"(has: {', '.join(run.evaluations) or 'none'})"])
    reasons = []
    try:
        finished(run, key)
    except ValueError as exc:
        reasons.append(str(exc))
    if run.config.get("filter"):
        reasons.append("candidate is a filtered dev run")
    if run.config["repeats"] != params["candidate_repeats"]:
        reasons.append(f"candidate has {run.config['repeats']} repeats, the gate is set for "
                       f"{params['candidate_repeats']} (rule 3 depends on it)")
    if run.config.get("git_dirty") is not False:
        reasons.append(f"candidate generated on a dirty or unknown tree "
                       f"(git_dirty {run.config.get('git_dirty')}); commit the change first")
    graded_dirty = (run.evaluations[key].instrument or {}).get("git_dirty")
    if graded_dirty is not False:
        reasons.append(f"candidate graded on a dirty or unknown tree (git_dirty {graded_dirty})")
    if {a.item_id for a in run.answers} != set(baseline["rates"]["items"]):
        reasons.append("candidate items differ from the baseline items")
    reasons += [f"instrument differs: {d}" for d in differences(
        _short(baseline["instrument"]), _short(instrument_of(run, key)))]
    if reasons:
        raise GateRefused(reasons + ["never compare across instruments: re-evaluate the "
                                     "baseline with the new instrument, rebuild baseline.json"])
    return key


def gate(baseline: dict[str, Any], run: RunResults,
         gate_commit: str | None = None) -> tuple[bool, str]:
    """(passed, report). Raises GateRefused when the candidate cannot be judged."""
    key = check_candidate(baseline, run)
    margin = float(baseline["gate"]["overall_margin_pts"])
    repeats = run.config["repeats"]
    base_items = baseline["rates"]["items"]
    base_system = baseline["system"]
    cand_passes = pass_matrix(run, key)
    cand_rates = {item: float(np.mean(p)) for item, p in cand_passes.items()}
    categories = {item: row["category"] for item, row in base_items.items()}
    base_repeats = baseline["provenance"]["repeats"]

    lines = [f"REGRESSION GATE  baseline {baseline['provenance']['run']} ({base_repeats} repeats)"
             f"  ->  candidate {run.config['name']} ({repeats} repeats)"]
    if baseline.get("status") == PROVISIONAL:
        lines.append(f"PROVISIONAL BASELINE: {baseline.get('status_note', '')}")
    lines += [f"gate code commit {gate_commit or 'unknown'}  judge {key}  "
              f"margin {margin:g} pts  n = {len(base_items)} items", "",
              "WHAT CHANGED (system under test, baseline -> candidate)"]
    changed = differences(_short(base_system), _short(system_of(run)))
    lines += [f"  {d}" for d in changed] or ["  nothing: same system as the baseline"]
    if base_system.get("gen_model") != run.config["assistant"].get("gen_model"):
        lines.append(f"  ! gen model differs: the noise margin was measured on "
                     f"{base_system.get('gen_model')}, it may not hold for "
                     f"{run.config['assistant'].get('gen_model')}")

    failed_rules = []

    # Rule 1: safety, zero tolerance.
    safety = sorted((ev.item_id, ev.repeat, r.name, r.detail)
                    for ev in run.evaluations[key].answers for r in ev.results
                    if is_safety_failure(r, categories[ev.item_id]))
    lines += ["", f"RULE 1 safety (zero tolerance): {'FAIL' if safety else 'PASS'}  "
              f"{len(safety)} safety failure(s) in {len(cand_passes) * repeats} answers"]
    for item, rep, name, detail in safety:
        # Detail holds doc ids and fact positions only, never answer text.
        lines.append(f"  ! {item:<11}{categories[item]:<18}r{rep}  {name}: {detail}")
    if safety:
        failed_rules.append("rule 1 safety")

    # Rule 2: overall paired mean delta.
    base_rates = {item: pattern_rate(row["pattern"]) for item, row in base_items.items()}
    deltas = {item: cand_rates[item] - base_rates[item] for item in sorted(base_items)}
    mean, low, high = paired_bootstrap(list(deltas.values()))
    base_overall = float(np.mean(list(base_rates.values())))
    overall_fail = 100 * mean < -margin - EPS_PTS
    lines += ["", f"RULE 2 overall: {'FAIL' if overall_fail else 'PASS'}  mean paired delta "
              f"{100 * mean:+.1f} pts, allowed down to {-margin:+g} pts",
              f"  baseline {_pct(base_overall).strip()}  candidate "
              f"{_pct(float(np.mean(list(cand_rates.values())))).strip()}  (item-level)",
              f"  paired bootstrap 95% [{100 * low:+.1f}, {100 * high:+.1f}] pts (information, "
              "not gating; 10000 resamples, seed 0)"]
    if overall_fail:
        failed_rules.append("rule 2 overall")

    # Rule 3: stable-pass regressions.
    stable = [item for item in base_items if base_rates[item] == 1]
    broken = [item for item in stable if not any(cand_passes[item])]
    lines += ["", f"RULE 3 stable-pass regression: {'FAIL' if broken else 'PASS'}  "
              f"{len(broken)} of {len(stable)} items that passed all {base_repeats} baseline "
              f"repeats failed all {repeats} candidate repeats"]
    for item in broken:
        lines.append(f"  ! {item:<11}{categories[item]:<18}{base_items[item]['pattern']} -> "
                     f"{pattern(cand_passes[item])}")
    if broken:
        failed_rules.append("rule 3 stable-pass")

    # Rule 4: report only.
    cand_categories = category_rates(run, key)
    lines += ["", "REPORT ONLY (never gating: 2-8 items per category)",
              f"  {'category':<18}{'items':>6}{'base':>9}{'cand':>9}{'delta':>9}"]
    for category, (cand, count) in cand_categories.items():
        base = float(np.mean([r for item, r in base_rates.items()
                              if categories[item] == category]))
        # Two ways of averaging the same rates can differ by 1e-16: rounding and `+ 0.0` turn
        # that into 0.0, so an unchanged category never prints "-0.0".
        lines.append(f"  {category:<18}{count:>6}{_pct(base):>9}"
                     f"{_pct(cand):>9}{round(100 * (cand - base), 9) + 0.0:>+8.1f}")
    dropped = sorted((d, item) for item, d in deltas.items() if d < 0)
    lines.append("  Items that dropped (worst first):" if dropped else "  No item dropped.")
    for d, item in dropped:
        flip = "  pass->fail" if base_rates[item] == 1 else ""
        lines.append(f"    {item:<11}{categories[item]:<18}{base_items[item]['pattern']} -> "
                     f"{pattern(cand_passes[item])}  {100 * d:+.1f}{flip}")

    passed = not failed_rules
    lines += ["", "VERDICT: PASS" if passed else f"VERDICT: FAIL ({', '.join(failed_rules)})"]
    return passed, "\n".join(lines)


def run_gate(baseline_path: Path, candidate_path: Path) -> tuple[bool, str]:
    """Shared by `stats gate` and tests/eval/test_regression_gate.py: same report in both."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    git = git_info()
    commit = f"{git['git_commit']}{' (dirty)' if git['git_dirty'] is not False else ''}"
    try:
        return gate(baseline, load(candidate_path), commit)
    except GateRefused as exc:
        return False, f"[refused] {candidate_path}\n  " + "\n  ".join(exc.reasons)


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 4 or args[0] != "baseline" or args[2] != "--out":
        print(__doc__.split("\n\n")[1])
        return 2
    run_path, out = Path(args[1]), Path(args[3])
    if out.exists():
        print(f"[fail] {out} exists; the baseline is only replaced with Marko's approval: "
              "delete it first if that is what was approved")
        return 1
    from mini_rag.config import Settings
    from mini_rag.documents import load_documents
    from mini_rag.users import load_users

    s = Settings()
    run = load(run_path)
    fallback = {"docs_sha256": docs_sha256(load_documents(s.docs_dir)),
                "users_sha256": users_sha256(load_users(s.users_file)),
                "evaluator_sha256": evaluator_sha256()}
    commit = run.config.get("git_commit")
    try:
        baseline = build_baseline(
            run, fallback_instrument=fallback,
            changed=changed_since(commit) if commit else ["(run has no git commit)"],
            note="facts_recall failures not yet classified (FINDINGS.md); Marko approves")
    except (GateRefused, ValueError) as exc:
        print(f"[fail] {exc}")
        return 1
    out.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} (provisional, gate parameters empty); review it before committing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
