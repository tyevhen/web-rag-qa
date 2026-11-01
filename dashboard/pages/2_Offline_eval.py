import json

import pandas as pd
import streamlit as st

from dashboard.data import load_result, result_files
from dashboard.views import run_label
from tests.eval.thresholds import THRESHOLDS_PATH, check_thresholds, lookup as lookup_any

st.set_page_config(page_title="Offline eval", layout="wide")
st.title("Offline evaluation")
st.caption("Runs of `tests.eval.run_eval` over the golden set. Metrics are grouped by pipeline stage.")

files = result_files()
if not files:
    st.info("No eval runs yet. Run `uv run python -m tests.eval.run_eval --skip-llm`.")
    st.stop()

runs = [load_result(f) for f in files]
thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS_PATH.read_text())


def lookup(report: dict, dotted: str) -> float | None:
    value = lookup_any(report, dotted)
    return value if isinstance(value, (int, float)) else None


_BROKEN = 0.5  # runs where at least half the rows errored (e.g. rate limits) say nothing about quality


def broken(report: dict) -> bool:
    return (lookup(report, "deterministic.operational.error_rate") or 0.0) >= _BROKEN


st.subheader("Trend across runs")
valid = [r for r in runs if not broken(r)]
if len(runs) > len(valid):
    st.caption(f"{len(runs) - len(valid)} run(s) left out: at least {_BROKEN:.0%} of their rows errored.")
if len(valid) < 2:
    st.info("A trend needs at least 2 valid runs.")
else:
    trend = pd.DataFrame(
        [{"run": r["meta"]["timestamp"], **{m.split(".", 1)[1]: lookup(r, m) for m in thresholds}} for r in reversed(valid)]
    ).set_index("run")
    st.line_chart(trend.dropna(axis=1, how="all"))

labels = [run_label(r) + (" · mostly errors" if broken(r) else "") for r in runs]
default = next((i for i, r in enumerate(runs) if not broken(r)), 0)
run = runs[st.selectbox("Run", range(len(runs)), index=default, format_func=lambda i: labels[i])]

violations = check_thresholds(run)
errors = lookup(run, "deterministic.operational.error_rate") or 0.0
if broken(run):
    st.error(f"{errors:.0%} of rows errored. The metrics below describe only the rows that ran.")
st.metric("Threshold violations", len(violations))

st.subheader("Metrics vs thresholds")
table = []
for metric, bounds in thresholds.items():
    value = lookup(run, metric)
    ok = value is None or (value >= bounds.get("min", float("-inf")) and value <= bounds.get("max", float("inf")))
    table.append({
        "metric": metric,
        "value": value,
        "limit": f"min {bounds['min']}" if "min" in bounds else f"max {bounds['max']}",
        "status": "not measured" if value is None else ("pass" if ok else "FAIL"),
    })
st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("**Retrieval variants**: does each component earn its place?")
    retrieval = run["deterministic"]["retrieval"]
    st.bar_chart(pd.DataFrame(retrieval).T[["hit@6", "mrr"]], stack=False, sort=False)
with right:
    st.markdown("**Grader threshold sweep**: precision and recall of the relevance filter")
    sweep = pd.DataFrame(run["deterministic"]["grader_threshold_sweep"]).T
    sweep.index = [float(i.split("=")[1]) for i in sweep.index]
    st.line_chart(sweep)

if "llm" in run:
    st.subheader("LLM-judged metrics (Ragas)")
    st.dataframe(pd.json_normalize(run["llm"]).T.rename(columns={0: "value"}), width="stretch")

st.subheader("Per question")
questions = pd.DataFrame(run["questions"])
questions["route"] = questions["route"].replace("", "error")
f1, f2 = st.columns(2)
categories = f1.multiselect("Category", sorted(questions["category"].unique()))
routes = f2.multiselect("Route", sorted(questions["route"].unique()))
mask = pd.Series(True, index=questions.index)
if categories:
    mask &= questions["category"].isin(categories)
if routes:
    mask &= questions["route"].isin(routes)
shown = questions[mask]
st.dataframe(
    shown[["category", "route", "retry_count", "retrieved_ref_hits", "graded_ref_hits", "latency_s", "question", "answer", "error"]],
    width="stretch",
    hide_index=True,
)
