"""Test data and helpers shared across test files (import these; fixtures live in conftest.py)."""

import json
from datetime import date
from pathlib import Path

# The restricted doc deliberately shares vocabulary with the public ones ("budget", "project"),
# so a broken access filter would rank it highly and the leak would show up.
FIXTURE_DOCS = {
    "orion-budget.md": ("orion-budget", "Orion budget", "restricted",
                        "The Orion project budget is 4.2 million euros. Codename BLUEHERON."),
    "orion-overview.md": ("orion-overview", "Orion overview", "public",
                          "Project Orion builds a route planner for delivery vans."),
    "holiday-policy.md": ("holiday-policy", "Holiday policy", "public",
                          "Employees get 27 days of paid holiday per year.", "2026-06-01"),
    "budget-process.md": ("budget-process", "Budget process", "public",
                          "Every project budget is reviewed each quarter by finance."),
}
RESTRICTED_SECRET = "BLUEHERON"
# Tests never read the real clock. A past date, so it never equals the run date and a wall-clock
# bug can't hide behind a coincidence. Before orion-budget-v2 takes effect; tests that need a
# later date set their own.
FIXED_TODAY = date(2026, 7, 15)


def write_dataset(path: Path, items: list[dict], name: str = "test",
                  as_of: str = "2026-07-15") -> Path:
    lines = [{"dataset": {"name": name, "as_of": as_of}}, *items]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def dataset_item(item_id: str, user: str, question: str, *, category: str = "factual",
                 expected: list | None = None, forbidden: list | None = None,
                 should_refuse: bool = False, as_of: str | None = None) -> dict:
    item = {"id": item_id, "category": category, "user": user, "question": question,
            "expected_facts": expected or [], "forbidden_facts": forbidden or [],
            "should_refuse": should_refuse}
    if as_of:
        item["as_of"] = as_of
    return item


def write_doc(directory: Path, filename: str, doc_id: str, title: str, access: str, body: str,
              effective: str | None = None):
    extra = f"effective: {effective}\n" if effective else ""
    (directory / filename).write_text(
        f"---\nid: {doc_id}\ntitle: {title}\naccess: {access}\n{extra}---\n{body}\n",
        encoding="utf-8",
    )
