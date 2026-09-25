"""Score ids decide what Langfuse overwrites: a wrong id either duplicates scores on every re-run
or lets a new judge version silently replace the old one's grades."""

from evals.publish import publish_scores
from evals.results import AnswerRecord, EvalResult, RetrievedDoc


class RecordingPublisher:
    def __init__(self) -> None:
        self.scores: dict[str, tuple[str, dict | None]] = {}  # name -> (score_id, metadata)

    def score(self, trace_id: str, name: str, value: float, comment: str, score_id: str,
              metadata: dict[str, str] | None) -> None:
        self.scores[name] = (score_id, metadata)


RECORD = AnswerRecord(
    item_id="x", repeat=1, category="factual", user="analyst", question="Q?", as_of="2026-09-01",
    answer=None, raw_outputs=[], retrieved=[RetrievedDoc(id="doc", access="public")], error=None,
    refusal_reason=None, attempts=1, invalid_outputs=0, prompt_tokens=[None], durations_ms=[None],
    truncation_risk=False, trace_id="a" * 32)
RESULTS = [EvalResult(name="facts_recall", applicable=True, passed=True, value=1.0),
           EvalResult(name="judge_faithfulness", applicable=True, passed=True, value=5)]


def published(judge_key: str) -> dict[str, tuple[str, dict | None]]:
    publisher = RecordingPublisher()
    publish_scores(publisher, RECORD, RESULTS, passed=True, judge_key=judge_key)
    return publisher.scores


def test_same_judge_version_reuses_score_ids_so_a_rerun_overwrites():
    assert published("aaa") == published("aaa")


def test_new_judge_version_gets_new_ids_only_for_judge_dependent_scores():
    old, new = published("aaa"), published("bbb")

    assert old["judge_faithfulness"][0] != new["judge_faithfulness"][0]
    assert old["item_pass"][0] != new["item_pass"][0]  # item_pass includes the judge verdict
    # facts_recall does not depend on the judge: one score per trace, not one per judge version.
    assert old["facts_recall"][0] == new["facts_recall"][0]
    assert new["judge_faithfulness"][1] == {"judge": "bbb"}
    assert new["facts_recall"][1] is None
