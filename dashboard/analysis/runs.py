"""Pure functions over eval-run reports and query_log frames. No Streamlit, so they can be unit tested."""
from typing import Any

import pandas as pd

from tests.eval.thresholds import lookup as lookup_any

BROKEN = 0.5  # runs where at least half the rows errored (e.g. rate limits) say nothing about quality


def lookup(report: dict[str, Any], dotted: str) -> float | None:
    value = lookup_any(report, dotted)
    return value if isinstance(value, (int, float)) else None


def error_rate(report: dict[str, Any]) -> float:
    return lookup(report, "deterministic.operational.error_rate") or 0.0


def broken(report: dict[str, Any]) -> bool:
    return error_rate(report) >= BROKEN


def threshold_rows(run: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for metric, bounds in thresholds.items():
        value = lookup(run, metric)
        ok = value is None or bounds.get("min", float("-inf")) <= value <= bounds.get("max", float("inf"))
        rows.append({
            "metric": metric,
            "value": value,
            "limit": f"min {bounds['min']}" if "min" in bounds else f"max {bounds['max']}",
            "status": "not measured" if value is None else ("pass" if ok else "FAIL"),
        })
    return rows


def trend(runs: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    """One row per run, oldest first, one column per thresholded metric."""
    return pd.DataFrame(
        [{"run": r["meta"]["timestamp"], **{m.split(".", 1)[1]: lookup(r, m) for m in thresholds}} for r in reversed(runs)]
    ).set_index("run")


def default_pair(runs: list[dict[str, Any]]) -> tuple[int, int]:
    """The latest run of each of the two most recent generators (runs are newest first); needs at least 2 runs."""
    generators = list(dict.fromkeys(r["meta"].get("generator") for r in runs))
    a = next(i for i, r in enumerate(runs) if r["meta"].get("generator") == generators[0])
    if len(generators) < 2:
        return a, 1
    return a, next(i for i, r in enumerate(runs) if r["meta"].get("generator") == generators[1])


def numbers(run: dict[str, Any]) -> pd.Series:
    flat = pd.json_normalize({k: run[k] for k in ("deterministic", "llm") if k in run}).iloc[0]
    return pd.Series(pd.to_numeric(flat, errors="coerce"), index=flat.index).dropna()


def paired_scores(a_items: list[dict[str, Any]], b_items: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Questions both runs judged, as (A item, B item)."""
    b_by_q = {q["question"]: q for q in b_items}
    return [(q, b_by_q[q["question"]]) for q in a_items if q.get("scores") and b_by_q.get(q["question"], {}).get("scores")]


def score_names(paired: list[tuple[dict, dict]]) -> list[str]:
    return sorted({m for qa, qb in paired for m in qa["scores"].keys() & qb["scores"].keys()})


def judge_summary(paired: list[tuple[dict, dict]]) -> list[dict[str, Any]]:
    summary = []
    for m in score_names(paired):
        pairs = [(qa["scores"].get(m), qb["scores"].get(m)) for qa, qb in paired]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if pairs:
            summary.append({"metric": m, "questions": len(pairs),
                            "mean A": sum(x for x, _ in pairs) / len(pairs),
                            "mean B": sum(y for _, y in pairs) / len(pairs),
                            "B better": sum(y > x for x, y in pairs), "tie": sum(y == x for x, y in pairs),
                            "A better": sum(y < x for x, y in pairs)})
    return summary


def production_stats(log: pd.DataFrame) -> dict[str, str]:
    """Headline numbers for a non-empty query_log frame."""
    latency = log["latency_ms"] / 1000
    return {
        "Requests": str(len(log)),
        "Clarify rate": f"{(log['route'] == 'clarify').mean():.0%}",
        "Latency p50 / p95": f"{latency.quantile(0.5):.1f}s / {latency.quantile(0.95):.1f}s",
        "Retry rate": f"{(log['retry_count'] > 0).mean():.0%}",
    }
