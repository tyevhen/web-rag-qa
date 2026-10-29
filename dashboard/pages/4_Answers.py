import pandas as pd
import streamlit as st

from dashboard.data import load_result, result_files
from dashboard.views import context_chunks, conversation, run_label

st.set_page_config(page_title="Answers", layout="wide")
st.title("Answers")
st.caption("Read what the pipeline answered on each golden question, with the conversation history and cited chunks.")

files = result_files()
if not files:
    st.info("No eval runs yet. Run `uv run python -m tests.eval.run_eval --skip-llm`.")
    st.stop()

runs = [load_result(f) for f in files]


def label(i: int) -> str:
    return run_label(runs[i])


c1, c2 = st.columns(2)
run_i = c1.selectbox("Run", range(len(runs)), format_func=label)
run = runs[run_i]
other_i = c2.selectbox("Compare with", [None, *range(len(runs))], format_func=lambda i: "—" if i is None else label(i))
other = {q["question"]: q for q in runs[other_i]["questions"]} if other_i is not None else {}

questions = pd.DataFrame(run["questions"])
questions["route"] = questions["route"].replace("", "error")
f1, f2, f3 = st.columns(3)
categories = f1.multiselect("Category", sorted(questions["category"].unique()))
routes = f2.multiselect("Route", sorted(questions["route"].unique()))
search = f3.text_input("Question contains")
mask = pd.Series(True, index=questions.index)
if categories:
    mask &= questions["category"].isin(categories)
if routes:
    mask &= questions["route"].isin(routes)
if search:
    mask &= questions["question"].str.contains(search, case=False, regex=False)
shown = questions[mask]
if shown.empty:
    st.info("No questions match these filters.")
    st.stop()

q = shown.loc[st.selectbox(f"Question ({len(shown)} shown)", shown.index, format_func=lambda i: questions.at[i, "question"])]
st.markdown(
    f"`{q['category']}` · route **{q['route']}** · retries {q['retry_count']} · "
    f"gold chunks retrieved {q['retrieved_ref_hits']}, kept by grader {q['graded_ref_hits']} · {q['latency_s']:.1f} s"
)
if q["error"]:
    st.error(q["error"])


item = q.to_dict()
if other_i is not None and item["question"] in other:
    left, right = st.columns(2)
    with left:
        conversation(item, label(run_i))
    with right:
        conversation(other[item["question"]], label(other_i))
else:
    conversation(item, "Conversation")

st.markdown("**Reference answer**")
st.info(item["reference"] or "No reference: the assistant should refuse or ask for clarification.")
if item["sub_questions"]:
    st.markdown("**Search queries (decomposer)**")
    st.write(item["sub_questions"])

st.markdown("**Chunks the reasoner saw** — `[n]` in the answer points to chunk n")
context_chunks(item.get("context"))
