from typing import Any

import pandas as pd

from dashboard.analysis.runs import broken, default_pair, judge_summary, serving_by_day, serving_line, threshold_rows


def _run(generator: str = "g", error_rate: float = 0.0, **metrics: Any) -> dict[str, Any]:
    return {"meta": {"generator": generator},
            "deterministic": {"operational": {"error_rate": error_rate}, "retrieval": {"dense": metrics}}}


def test_broken_at_half_errors():
    assert broken(_run(error_rate=0.5))
    assert not broken(_run(error_rate=0.49))


def test_threshold_rows_pass_fail_and_not_measured():
    thresholds = {"deterministic.retrieval.dense.mrr": {"min": 0.5},
                  "deterministic.retrieval.dense.hit": {"max": 0.2},
                  "deterministic.retrieval.dense.absent": {"min": 0.1}}
    target = {**thresholds, "deterministic.retrieval.dense.mrr": {"min": 0.8}}
    rows = threshold_rows(_run(mrr=0.6, hit=0.3), thresholds, target)
    assert [r["status"] for r in rows] == ["pass", "FAIL", "not measured"]
    assert [r["limit"] for r in rows] == ["min 0.5", "max 0.2", "min 0.1"]
    assert [r["target"] for r in rows] == ["min 0.8", "", ""]  # shown only where a floor replaced it


def test_judge_summary_counts_wins_and_ties():
    def q(name: str, score: float | None) -> dict[str, Any]:
        return {"question": name, "scores": {"faithfulness": score}}

    paired = [(q("1", 0.2), q("1", 0.9)), (q("2", 0.5), q("2", 0.5)), (q("3", 0.8), q("3", 0.1)),
              (q("4", None), q("4", 0.7))]
    [row] = judge_summary(paired)
    assert (row["questions"], row["B better"], row["tie"], row["A better"]) == (3, 1, 1, 1)


def test_default_pair_takes_latest_run_of_two_newest_generators():
    runs = [_run("x"), _run("x"), _run("y"), _run("z")]
    assert default_pair(runs) == (0, 2)
    assert default_pair([_run("x"), _run("x")]) == (0, 1)


def _call(node: str, ttft_ms: int, tpot_ms: float | None = 20.0) -> dict[str, Any]:
    return {"node": node, "start_ms": 1000, "ttft_ms": ttft_ms, "total_ms": ttft_ms + 500,
            "prompt_tokens": 900, "output_tokens": 26, "tpot_ms": tpot_ms}


def test_serving_numbers_come_from_the_last_reason_call():
    log = pd.DataFrame({"day": ["d1", "d1", "d2"],
                        "llm_calls": [[_call("decompose", 9000), _call("reason", 500)], [_call("reason", 1500)], []]})
    by_day = serving_by_day(log)
    assert list(by_day.index) == ["d1"] and by_day.loc["d1", "ttft p50"] == 1.0
    assert serving_line([_call("reason", 500, None)]) == "first answer token at 1.5 s · prompt 900 tokens"
    assert serving_line([]) == ""
