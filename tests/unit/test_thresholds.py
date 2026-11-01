import json

from tests.eval import thresholds


def test_missing_metric_in_a_section_that_ran_is_a_violation(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({
        "deterministic.retrieval.hit@6": {"min": 0.8},
        "deterministic.routing.false_answer_rate": {"max": 0.2},
        "llm.generation.faithfulness": {"min": 0.85},
    }))
    monkeypatch.setattr(thresholds, "THRESHOLDS_PATH", path)
    report = {"deterministic": {"retrieval": {"hit@5": 0.9}, "routing": {"false_answer_rate": 0.3}}}

    assert thresholds.check_thresholds(report) == [
        "deterministic.retrieval.hit@6 = missing",
        "deterministic.routing.false_answer_rate = 0.3 > max 0.2",
    ]  # llm section never ran (--skip-llm), so its gate is skipped


def test_metric_with_no_rows_to_score_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"deterministic.routing.false_answer_rate": {"max": 0.2}}))
    monkeypatch.setattr(thresholds, "THRESHOLDS_PATH", path)

    # --limit run with no unanswerable rows: the metric exists but has no value
    assert thresholds.check_thresholds({"deterministic": {"routing": {"false_answer_rate": None}}}) == []
