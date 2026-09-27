Next, as separate commits, in this order:

1) Judge v3, by protocol:
   - Write a new prediction in DECISIONS.md first (committed alone), including what should happen to ver-02.
   - Put `reason` before `score` in the judge's JSON schema.
   - Add an after-v2 control pair (as_of after v2: the new figure passes, the superseded figure fails).
   - Re-run evaluate on 20260925-smoke-draft and report every change against the prediction.

2) Keep evaluation history:
   - Include the judge prompt sha in score_id and key local evaluations by it, so the same judge version
     overwrites itself and a new version is stored alongside the old one.
   - The report shows which judge version it is using.

3) Make evaluate robust before any long run:
   - Save after every evaluated answer and support --resume.
   - A timeout becomes judge_error after one retry.
   - Make the Ollama read timeout configurable and count timeouts in the report.
   - Test it with FakeLLM raising a timeout mid-run.

4) Report header shows both dataset hashes (generation and evaluation).