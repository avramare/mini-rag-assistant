Commit Phase 3 now as "feat: phase 3 dataset, two-pass runner, evaluators", before any judge change.

Then, as separate commits:
1) Judge fix, by protocol:
   - Write the prediction in DECISIONS.md first: ver-02 and ver-03 flip to pass, nothing else changes.
   - Give the judge the as_of date and the rule "use the latest version in force on as_of".
   - Add a negative control test: an answer with the not-yet-effective Orion figure for an as_of before v2
     must score below the pass threshold.
   - Re-run only `evaluate` on 20260925-smoke-draft. Report every item that changed, both directions,
     and whether the prediction held.
2) Versioning items: put the not-yet-effective value into forbidden_facts, so a code evaluator catches
   the wrong version independently of the judge.
3) Add expected_docs to the dataset schema and a retrieval_recall evaluator. Fill expected_docs in the draft set.
   Re-evaluate and tell me for fact-01, fact-05 and restr-03b whether the missing half was a retrieval
   or a generation failure.
4) Note in PLAN.md Phase 5: zero tolerance applies to safety evaluators (forbidden_absent, citations_valid,
   refusal_correct where should_refuse), not to every evaluator on safety-category items.
5) Note the empty-answer case (ver-04) in STATUS.md as a contract change to measure after Phase 4. Do not change it now.