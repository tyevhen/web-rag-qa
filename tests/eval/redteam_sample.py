"""Stratified red-team sample from the jailbreak / prompt-injection dataset (HF Necent/llm-jailbreak-prompt-injection-dataset).

    uv run python -m tests.eval.redteam_sample --data ../llm-jailbreak-prompt-injection-dataset/data

Writes tests/eval/results/redteam/sample.jsonl (gitignored): one attack per line with its arm and dataset labels.
Needs Postgres with ingested content for the on-topic arm (prompts nearest to the corpus).
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agent.embed import embed_queries, get_encoder
from db.client import create_pool
from db.repositories.documents import DocumentRepository

OUT = Path(__file__).parent / "results" / "redteam" / "sample.jsonl"
SEED = 7
_COLS = ["prompt", "prompt_type", "source", "language", "prompt_harmful", "prompt_adversarial", "attack_technique"]
# Llama-3 chat template turns, as used by the many-shot rows
_TURN = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\s*(.*?)(?=<\|eot_id\|>|<\|start_header_id\|>|$)", re.S)


def spread(df: pd.DataFrame, n: int, by: str) -> pd.DataFrame:
    """n rows, round-robin across the values of `by` so rare techniques/sources are represented."""
    shuffled = df.sample(frac=1, random_state=SEED)
    order = shuffled.groupby(by, sort=False).cumcount()
    return shuffled.assign(_o=order).sort_values("_o", kind="stable").head(n).drop(columns="_o")


def split_turns(prompt: str) -> tuple[str, list[dict]] | None:
    """Chat-template prompt → (last user message, earlier user/assistant turns as client history)."""
    turns = [(r, c.strip()) for r, c in _TURN.findall(prompt) if r in ("user", "assistant") and c.strip()]
    while turns and turns[-1][0] != "user":
        turns.pop()
    if len(turns) < 2:
        return None
    return turns[-1][1], [{"role": r, "content": c} for r, c in turns[:-1]]


async def _corpus_vectors() -> np.ndarray:
    pool = await create_pool()
    try:
        rows = await DocumentRepository(pool).chunk_embeddings(get_encoder().name)
    finally:
        await pool.close()
    return np.array([np.asarray(r["embedding"], dtype=np.float32) for r in rows])


async def on_topic(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Harmful prompts closest to the corpus: they retrieve real passages and pass the grader."""
    corpus = await _corpus_vectors()
    vecs = np.array(await asyncio.to_thread(embed_queries, candidates.prompt.tolist()), dtype=np.float32)
    best = (vecs @ corpus.T).max(axis=1)  # vectors are normalized → dot = cosine
    return candidates.assign(corpus_cosine=best).nlargest(n, "corpus_cosine")


def arms(df: pd.DataFrame) -> list[pd.DataFrame]:
    en = df.loc[df.language == "en"]
    harmful, adv = en.prompt_harmful == 1, en.prompt_adversarial == 1
    t = en.prompt_type
    return [
        spread(en.loc[(t == "jailbreak") & harmful & adv], 60, "attack_technique").assign(arm="jailbreak"),
        spread(en.loc[(t == "prompt_injection") & adv], 60, "source").assign(arm="prompt_injection"),
        spread(en.loc[t == "obfuscation"], 60, "attack_technique").assign(arm="obfuscation"),
        spread(en.loc[(t == "harmful_behavior") & harmful & ~adv], 60, "source").assign(arm="harmful_behavior"),
        spread(en.loc[(t == "toxicity") & harmful], 60, "source").assign(arm="toxicity"),
        spread(en.loc[(t == "linguistic") & harmful], 40, "source").assign(arm="linguistic"),
        spread(df.loc[(df.language != "en") & ((df.prompt_harmful == 1) | (df.prompt_adversarial == 1))], 40,
               "language").assign(arm="multilingual"),
        spread(en.loc[(en.source == "OR-Bench") & ~harmful & ~adv], 60, "source").assign(arm="benign_control"),
    ]


def records(frame: pd.DataFrame) -> list[dict]:
    out = []
    for r in frame.to_dict("records"):
        history: list[dict] = []
        question = r["prompt"]
        if r["arm"] == "fake_history":
            question, history = split_turns(r["prompt"])  # type: ignore[misc]  # filtered to parseable rows
        meta = {k: (int(r[k]) if k.startswith("prompt_") and k != "prompt_type" else r[k]) for k in _COLS[1:]}
        out.append({"arm": r["arm"], "question": question, "history": history, **meta,
                    "corpus_cosine": round(float(r.get("corpus_cosine", float("nan"))), 3)})
    return out


async def main(data: Path) -> None:
    df = pd.concat([pd.read_parquet(f, columns=_COLS) for f in sorted(data.glob("*.parquet"))], ignore_index=True)
    df = df.dropna(subset=["prompt"])
    picked = arms(df)
    taken = set(pd.concat(picked).index)
    en = df.loc[(df.language == "en") & ~df.index.isin(taken)]
    pool = en.loc[(en.prompt_type == "harmful_behavior") & (en.prompt_harmful == 1) & (en.prompt.str.len() < 1000)]
    picked.append((await on_topic(pool.sample(min(20_000, len(pool)), random_state=SEED), 60)).assign(arm="on_topic"))
    shots = en.loc[en.attack_technique == "many_shot"]
    shots = shots.loc[shots.prompt.map(lambda p: split_turns(p) is not None)]
    picked.append(shots.sample(min(30, len(shots)), random_state=SEED).assign(arm="fake_history"))
    rows = [rec for frame in picked for rec in records(frame)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n" for i, r in enumerate(rows)))
    counts = pd.Series([r["arm"] for r in rows]).value_counts().to_dict()
    print(f"{len(rows)} rows → {OUT}  {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../llm-jailbreak-prompt-injection-dataset/data"))
    asyncio.run(main(parser.parse_args().data))
