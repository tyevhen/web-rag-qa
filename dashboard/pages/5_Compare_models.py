from typing import Any

import pandas as pd
import streamlit as st

from dashboard.cases import find_cases
from dashboard.data import load_result, result_files
from dashboard.views import context_chunks, conversation, run_label

st.set_page_config(page_title="Compare models", layout="wide")
st.title("Compare models")
st.caption("Two eval runs side by side, usually two generator models on the same golden set: "
           "metrics, per-question judge scores, and the questions where they differ.")

runs = [load_result(f) for f in result_files()]
if len(runs) < 2:
    st.info("Needs two eval runs. Run `uv run python -m tests.eval.run_eval` once per model.")
    st.stop()

# Default: the latest run of each of the two most recent generators (runs are newest first)
generators = list(dict.fromkeys(r["meta"].get("generator") for r in runs))
default_a = next(i for i, r in enumerate(runs) if r["meta"].get("generator") == generators[0])
default_b = next((i for i, r in enumerate(runs) if r["meta"].get("generator") == generators[1]), 1) \
    if len(generators) > 1 else 1
c1, c2 = st.columns(2)
a_i = c1.selectbox("Run A", range(len(runs)), index=default_a, format_func=lambda i: run_label(runs[i]))
b_i = c2.selectbox("Run B", range(len(runs)), index=default_b, format_func=lambda i: run_label(runs[i]))
a, b = runs[a_i], runs[b_i]


def numbers(run: dict[str, Any]) -> pd.Series:
    flat = pd.json_normalize({k: run[k] for k in ("deterministic", "llm") if k in run}).iloc[0]
    return pd.Series(pd.to_numeric(flat, errors="coerce"), index=flat.index).dropna()


st.subheader("Metrics")
metrics = pd.DataFrame({"A": numbers(a), "B": numbers(b)})
metrics["Δ (B − A)"] = metrics["B"] - metrics["A"]
st.dataframe(metrics, width="stretch")

st.subheader("Judge scores per question")
a_items, b_items = a["questions"], b["questions"]
b_by_q = {q["question"]: q for q in b_items}
paired = [(q, b_by_q[q["question"]]) for q in a_items if q.get("scores") and b_by_q.get(q["question"], {}).get("scores")]
if not paired:
    st.caption("Needs per-question judge scores in both runs: run `run_eval` without `--skip-llm`.")
else:
    names = sorted({m for qa, qb in paired for m in qa["scores"].keys() & qb["scores"].keys()})
    summary = []
    for m in names:
        pairs = [(qa["scores"].get(m), qb["scores"].get(m)) for qa, qb in paired]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if pairs:
            summary.append({"metric": m, "questions": len(pairs),
                            "mean A": sum(x for x, _ in pairs) / len(pairs),
                            "mean B": sum(y for _, y in pairs) / len(pairs),
                            "B better": sum(y > x for x, y in pairs), "tie": sum(y == x for x, y in pairs),
                            "A better": sum(y < x for x, y in pairs)})
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    with st.expander("Every question"):
        st.dataframe(pd.DataFrame([{"question": qa["question"],
                                    **{f"{m} {side}": q["scores"].get(m)
                                       for m in names for side, q in (("A", qa), ("B", qb))}}
                                   for qa, qb in paired]), hide_index=True, width="stretch")

st.subheader("Interesting cases")
cases = find_cases(a_items, b_items)
if not cases:
    st.caption("No differences worth flagging between these runs.")
    st.stop()
st.dataframe(pd.DataFrame(cases), hide_index=True, width="stretch")
pick = st.selectbox(f"Read a case ({len(cases)})", range(len(cases)), format_func=lambda i: cases[i]["question"])
case = cases[pick]
st.caption(case["why"])
qa = next(q for q in a_items if q["question"] == case["question"])
qb = b_by_q[case["question"]]
for col, side, item in zip(st.columns(2), ("A", "B"), (qa, qb), strict=True):
    with col:
        conversation(item, f"{side}: {item['route'] or 'error'} · {item['latency_s']:.1f} s · "
                           f"gold retrieved {item['retrieved_ref_hits']}")
        with st.expander("Chunks the reasoner saw"):
            context_chunks(item.get("context"))
st.markdown("**Reference answer**")
st.info(qa["reference"] or "No reference: the assistant should refuse or ask for clarification.")
