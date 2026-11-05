import pandas as pd
import streamlit as st

from dashboard.analysis.runs import BROKEN, broken, error_rate, threshold_rows, trend
from dashboard.components.conversation import run_label
from dashboard.data.results import load_runs, load_thresholds
from tests.eval.thresholds import check_thresholds

st.title("Offline evaluation")
st.caption("Runs of `tests.eval.run_eval` over the golden set. Metrics are grouped by pipeline stage.")

runs = load_runs()
if not runs:
    st.info("No eval runs yet. Run `uv run python -m tests.eval.run_eval --skip-llm`.")
    st.stop()

thresholds = load_thresholds()

st.subheader("Trend across runs")
valid = [r for r in runs if not broken(r)]
if len(runs) > len(valid):
    st.caption(f"{len(runs) - len(valid)} run(s) left out: at least {BROKEN:.0%} of their rows errored.")
if len(valid) < 2:
    st.info("A trend needs at least 2 valid runs.")
else:
    st.line_chart(trend(valid, thresholds).dropna(axis=1, how="all"))

labels = [run_label(r) + (" · mostly errors" if broken(r) else "") for r in runs]
default = next((i for i, r in enumerate(runs) if not broken(r)), 0)
run = runs[st.selectbox("Run", range(len(runs)), index=default, format_func=lambda i: labels[i])]

violations = check_thresholds(run)
if broken(run):
    st.error(f"{error_rate(run):.0%} of rows errored. The metrics below describe only the rows that ran.")
st.metric("Threshold violations", len(violations))

st.subheader("Metrics vs thresholds")
st.dataframe(pd.DataFrame(threshold_rows(run, thresholds)), width="stretch", hide_index=True)

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
