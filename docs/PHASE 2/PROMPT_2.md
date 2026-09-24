Phase 3 plan approved with these changes.

First, as a small separate commit, close the Phase 2 review findings:
1) add a lead question that retrieves only public docs as a control (mask by retrieved docs, not by user),
2) pin LANGFUSE_SAMPLE_RATE / LANGFUSE_TRACING_ENABLED in the test fixture,
3) test the fail-closed default (span opened before retrieval must be masked),
plus the test where a restricted call fails validation on both attempts.

Decisions:
A) Do not upload facts. B) facts_recall passes at 1.0; each fact may be a string or a list of accepted variants
(any variant matches). C) OK. D) The dataset has a fixed as_of date (dataset-level default, optional per-item override).
The runner uses it instead of the real date, records it, and stats compare refuses runs with different as_of.

Report: besides pass rates, show refusal_reason distribution, errors, retries, truncation_risk count,
and p50/p95 of generation duration.

Draft dataset: you may write evals/dataset.draft.jsonl (never evals/dataset.jsonl). About 30 items:
factual 8, multi_doc 5, unanswerable 5, restricted_probe 6 (3 analyst/lead pairs), injection 2, versioning 4
(Orion asked with as_of before and after v2's effective date, plus both policy pairs).
Every expected and forbidden fact must appear verbatim in the corpus; check it in code. Paraphrase questions,
do not copy sentences from the docs. Add a DECISIONS entry that this set is generated and will be compared
with my hand-written set.

Then implement the plan. Run the smoke test on the draft dataset with 1 repeat and show me the report.