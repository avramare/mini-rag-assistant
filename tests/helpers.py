"""Test data and helpers shared across test files (import these; fixtures live in conftest.py)."""

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
FIXED_TODAY = date(2026, 9, 24)  # tests never read the real clock


def write_doc(directory: Path, filename: str, doc_id: str, title: str, access: str, body: str,
              effective: str | None = None):
    extra = f"effective: {effective}\n" if effective else ""
    (directory / filename).write_text(
        f"---\nid: {doc_id}\ntitle: {title}\naccess: {access}\n{extra}---\n{body}\n",
        encoding="utf-8",
    )
