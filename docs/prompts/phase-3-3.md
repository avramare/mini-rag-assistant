Next steps, each with a prediction in DECISIONS.md first where the judge changes:

1) Judge v4: compute each document's status for as_of (in force / superseded / not yet in force) in an
   evaluator-side function, independent of the app code, and label documents with it in the judge prompt.
   Unit-test the function with same-year and different-year dates. Also add a control with as_of in 2028
   to test the hypothesis that the judge fails only on same-year date comparisons (run it on v3 first).

2) Judge cost: limit the reason to at most 3 sentences and cap output tokens (num_predict). Report
   judge p50/p95 duration before and after, and confirm all controls still behave the same.

3) Runner: add --items and --category filters for quick development runs. Filtered runs are marked
   in the results config and cannot be compared with full runs.

Do not start Phase 4.