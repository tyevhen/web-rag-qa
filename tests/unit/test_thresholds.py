import json

from tests.eval import thresholds


def test_missing_metric_in_a_section_that_ran_is_a_violation(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({
        "deterministic.retrieval.hit@6": {"min": 0.8},
        "deterministic.routing.off_corpus_answered_rate": {"max": 0.2},
        "llm.generation.faithfulness": {"min": 0.85},
    }))
    monkeypatch.setattr(thresholds, "THRESHOLDS_PATH", path)
    report = {"deterministic": {"retrieval": {"hit@5": 0.9}, "routing": {"off_corpus_answered_rate": 0.3}}}

    assert thresholds.check_thresholds(report) == [
        "deterministic.retrieval.hit@6 = missing",
        "deterministic.routing.off_corpus_answered_rate = 0.3 > max 0.2",
    ]  # llm section never ran (--skip-llm), so its gate is skipped


def test_metric_with_no_rows_to_score_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"deterministic.routing.off_corpus_answered_rate": {"max": 0.2}}))
    monkeypatch.setattr(thresholds, "THRESHOLDS_PATH", path)

    # --limit run with no unanswerable rows: the metric exists but has no value
    assert thresholds.check_thresholds({"deterministic": {"routing": {"off_corpus_answered_rate": None}}}) == []


def test_floors_replace_the_target_for_their_golden_set(tmp_path, monkeypatch):
    path, floors = tmp_path / "thresholds.json", tmp_path / "floors.json"
    path.write_text(json.dumps({"deterministic.routing.rate": {"max": 0.05}, "deterministic.routing.other": {"max": 0.1}}))
    floors.write_text(json.dumps({"golden_et.jsonl": {"deterministic.routing.rate": {"max": 0.19}}}))
    monkeypatch.setattr(thresholds, "THRESHOLDS_PATH", path)
    monkeypatch.setattr(thresholds, "FLOORS_PATH", floors)
    run = {"deterministic": {"routing": {"rate": 0.15, "other": 0.2}}}

    assert thresholds.check_thresholds({**run, "meta": {"golden": "golden_et.jsonl"}}) == [
        "deterministic.routing.other = 0.2 > max 0.1"]  # floor passes, gates it doesn't replace still apply
    assert thresholds.check_thresholds({**run, "meta": {"golden": "golden.jsonl"}}) == [
        "deterministic.routing.rate = 0.15 > max 0.05", "deterministic.routing.other = 0.2 > max 0.1"]
    assert thresholds.check_thresholds(run, thresholds.target())[0].startswith("deterministic.routing.rate")
