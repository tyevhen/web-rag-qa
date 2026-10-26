import pandas as pd
import streamlit as st

from dashboard.data import load_eval_scores, load_query_log
from evaluation.alerts import ALERT_THRESHOLDS

st.set_page_config(page_title="Production", layout="wide")
st.title("Production")
st.caption("Live traffic from `query_log`, and nightly judge scores from `query_eval` (sampled by the worker cron).")

days = st.slider("Window (days)", min_value=1, max_value=90, value=14)
log = load_query_log(days)

if log.empty:
    st.info("No /ask traffic in this window.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", len(log))
    c2.metric("Clarify rate", f"{(log['route'] == 'clarify').mean():.0%}")
    c3.metric("Latency p50 / p95", f"{log['latency_ms'].quantile(0.5) / 1000:.1f}s / {log['latency_ms'].quantile(0.95) / 1000:.1f}s")
    c4.metric("Retry rate", f"{(log['retry_count'] > 0).mean():.0%}")

    log["day"] = pd.to_datetime(log["created_at"]).dt.date
    st.subheader("Requests per day by route")
    st.bar_chart(log.pivot_table(index="day", columns="route", values="question", aggfunc="count", fill_value=0))
    st.subheader("Latency per day (seconds)")
    latency = pd.DataFrame(log.groupby("day")["latency_ms"].agg(p50=lambda s: s.quantile(0.5), p95=lambda s: s.quantile(0.95)))
    st.line_chart(latency / 1000)

st.subheader("Answer quality (sampled, LLM-judged)")
scores = load_eval_scores(days)
if scores.empty:
    st.info("No scored answers yet. The worker scores a sample nightly (score_recent_task), or trigger it by hand.")
    st.stop()

scores["day"] = pd.to_datetime(scores["evaluated_at"]).dt.date
daily = scores.pivot_table(index="day", columns="metric", values="score", aggfunc="mean")
cols = st.columns(len(ALERT_THRESHOLDS))
for col, (metric, threshold) in zip(cols, ALERT_THRESHOLDS.items()):
    mean = scores.loc[scores["metric"] == metric, "score"].mean()
    col.metric(f"{metric} (alert < {threshold})", "n/a" if pd.isna(mean) else f"{mean:.2f}",
               delta=None if pd.isna(mean) else f"{mean - threshold:+.2f} vs alert", delta_color="normal")
st.line_chart(daily)

st.subheader("Lowest-scoring answers")
worst = scores.dropna(subset=["score"]).sort_values("score").head(10)
st.dataframe(worst[["evaluated_at", "metric", "score", "question", "answer", "judge_model"]], width="stretch", hide_index=True)
