from typing import Any

from dashboard.cases import find_cases
from tests.eval.refusal import is_refusal


def _item(**overrides: Any) -> dict[str, Any]:
    item = {"question": "q", "category": "answerable", "route": "reason", "answer": "Do X [1].",
            "context": ["c1"], "retrieved_ref_hits": 1, "scores": {}}
    return {**item, **overrides}


def test_identical_good_answers_are_not_cases():
    assert find_cases([_item()], [_item()]) == []


def test_off_corpus_answer_without_refusal_is_a_case_and_refusal_is_not():
    a = _item(category="unanswerable", answer="Sudden numbness of the face [1].", retrieved_ref_hits=0)
    b = _item(category="unanswerable", answer="The provided information does not include the signs.",
              retrieved_ref_hits=0)

    [case] = find_cases([a], [b])

    assert "A answered an off-corpus question" in case["why"]
    assert "B answered" not in case["why"]


def test_citation_past_context_gold_miss_and_score_gap():
    a = _item(answer="Do X [3].", scores={"faithfulness": 0.9})
    b = _item(retrieved_ref_hits=0, scores={"faithfulness": 0.2})

    [case] = find_cases([a], [b])

    assert "A: cites [3] but saw 1 chunks" in case["why"]
    assert "gold chunk retrieved only by A" in case["why"]
    assert "faithfulness: A 0.90, B 0.20" in case["why"]


def test_refusal_without_citation_is_not_a_citation_problem():
    refusal = _item(category="unanswerable", answer="The provided information does not include the signs.")

    assert find_cases([refusal], [refusal]) == []


def test_is_refusal_reads_hedged_refusals():
    assert is_refusal("The maximum daily dose of ibuprofen is not directly provided in the given information.")
    assert not is_refusal("Apply direct pressure for 10 minutes [1].")
