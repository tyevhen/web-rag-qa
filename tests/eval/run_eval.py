import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.nodes._llm import get_groq_llm

load_dotenv()

API_BASE = "http://localhost:8000"
QA_PAIRS_PATH = Path("tests/eval/qa_pairs.json")
RESULTS_PATH = Path("tests/eval/results.json")

_QUESTION_GEN_PROMPT = """Generate 20 diverse test questions for a civil preparedness RAG system.
The system contains content from the Estonian Rescue Services website (olevalmis.ee) covering:
air raids, drone threats, evacuation, sheltering, emergency supplies, power outages,
water supply disruption, communication outages, chemical/radiation accidents,
floods, wildfires, heatwaves, storms, and the EE-ALARM public warning system.

Include a mix of:
- Straightforward factual questions
- Multi-step situational questions ("I am in X situation, what do I do?")
- Edge cases and unusual scenarios
- Questions likely to expose gaps or hallucinations

Return a JSON array of strings only, no other text."""

_JUDGE_PROMPT = """Evaluate this RAG system response.

Question: {question}
Answer: {answer}
Citations: {citations}

Score each dimension 1-5 (5 = excellent):
- relevance: does the answer directly address the question asked?
- groundedness: is the answer supported by the cited sources, or does it appear hallucinated?
- completeness: is it a thorough answer, or does it dodge/partially answer?

Set flag=true if any score is <= 2.

Respond with JSON only, no other text:
{{"relevance": <int>, "groundedness": <int>, "completeness": <int>, "flag": <bool>, "reason": "<one sentence>"}}"""


async def _generate_questions(llm) -> list[str]:
    response = await llm.ainvoke([HumanMessage(content=_QUESTION_GEN_PROMPT)])
    return json.loads(str(response.content))


async def _ask(client: httpx.AsyncClient, question: str) -> dict:
    t0 = time.monotonic()
    r = await client.post("/ask", json={"question": question}, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    data["latency_s"] = round(time.monotonic() - t0, 2)
    return data


async def _judge(llm, question: str, answer: str, citations: list[str]) -> dict:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        citations="\n".join(citations) if citations else "none provided",
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return json.loads(str(response.content))


def _print_report(results: list[dict]) -> None:
    scored = [r for r in results if "relevance" in r]
    errors = [r for r in results if "error" in r]
    flagged = [r for r in scored if r["flag"]]

    if scored:
        avg = {
            k: round(sum(r[k] for r in scored) / len(scored), 2)
            for k in ("relevance", "groundedness", "completeness")
        }
    else:
        avg = {}

    print(f"\n{'='*70}")
    print(f"EVAL REPORT  |  {len(results)} questions  |  {len(flagged)} flagged  |  {len(errors)} errors")
    if avg:
        print(
            f"Avg — relevance: {avg['relevance']}  "
            f"groundedness: {avg['groundedness']}  "
            f"completeness: {avg['completeness']}"
        )
    print(f"{'='*70}")

    sorted_results = sorted(
        results,
        key=lambda r: min(r.get("relevance", 0), r.get("groundedness", 0), r.get("completeness", 0)),
    )

    for r in sorted_results:
        if "error" in r:
            print(f"\n[ERROR] {r['question']}")
            print(f"  ! {r['error']}")
            continue

        avg_score = (r["relevance"] + r["groundedness"] + r["completeness"]) / 3
        label = "FLAGGED" if r["flag"] else "ok     "
        print(
            f"\n[{label}] avg={avg_score:.1f}  "
            f"rel={r['relevance']} gnd={r['groundedness']} cmp={r['completeness']}  "
            f"({r['latency_s']}s)"
        )
        print(f"  Q: {r['question']}")
        if r["flag"]:
            print(f"  ! {r['reason']}")


async def main() -> None:
    llm = get_groq_llm()

    questions: list[str] = json.loads(QA_PAIRS_PATH.read_text())
    if not questions:
        print("qa_pairs.json is empty — generating questions from ingested content...")
        questions = await _generate_questions(llm)
        QA_PAIRS_PATH.write_text(json.dumps(questions, indent=2))
        print(f"Generated {len(questions)} questions, saved to {QA_PAIRS_PATH}\n")

    results: list[dict] = []
    async with httpx.AsyncClient(base_url=API_BASE) as client:
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:72]}...")
            try:
                api_resp = await _ask(client, question)
                scores = await _judge(llm, question, api_resp["answer"], api_resp["citations"])
                results.append({
                    "question": question,
                    "answer": api_resp["answer"],
                    "citations": api_resp["citations"],
                    "sub_questions": api_resp["sub_questions"],
                    "latency_s": api_resp["latency_s"],
                    **scores,
                })
            except Exception as exc:
                print(f"  ERROR: {exc}")
                results.append({"question": question, "error": str(exc), "flag": True})

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    _print_report(results)
    print(f"\nFull results → {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
