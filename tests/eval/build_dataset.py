"""Generate the synthetic part of the golden set from ingested chunks.

    uv run python -m tests.eval.build_dataset [--size 10] [--sample 20]

Kept deliberately cheap for the Groq free tier (~8k tokens/min): one NER call per chunk,
fixed personas, and one call per question. Single-hop questions only; add multi-hop ones by hand.

Review golden.jsonl by hand afterwards: fix wrong references, delete nonsense rows.
Hand-written unanswerable / multi_turn rows already in the file are preserved.
"""
import argparse
import asyncio
import logging

import numpy as np
from langchain_core.documents import Document
from ragas import SingleTurnSample
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms.extractors.llm_based import NERExtractor

from agent.embed import embed_documents
from db.client import create_pool
from db.repository import DocumentRepository
from evaluation.judge import RUN_CONFIG, get_judge_embeddings, get_judge_llm
from tests.eval.dataset import GOLDEN_PATH, GoldenRow, content_hash, load_golden, save_golden

log = logging.getLogger(__name__)

# Fixed, domain-agnostic personas — skips Ragas's LLM-based persona generation (which needs summaries)
_PERSONAS = [
    Persona(name="Newcomer", role_description="Knows nothing about the topic and asks basic, practical questions."),
    Persona(name="Practitioner", role_description="Applies the material in real situations and asks for specific steps."),
    Persona(name="Detail checker", role_description="Asks precise questions about numbers, conditions and exceptions."),
]


def _source_hashes(contexts: list[str], chunks: list[str], chunk_vecs: np.ndarray) -> list[str]:
    """Map each generated reference context back to the chunk it came from."""
    hashes: list[str] = []
    for ctx in contexts:
        norm = " ".join(ctx.split())
        exact = [c for c in chunks if norm and norm in " ".join(c.split())]
        if exact:
            hashes.append(content_hash(exact[0]))
            continue
        # Ragas may re-split or trim a document — fall back to the nearest chunk
        ctx_vec = np.array(embed_documents([ctx])[0])
        hashes.append(content_hash(chunks[int(np.argmax(chunk_vecs @ ctx_vec))]))
    return list(dict.fromkeys(hashes))


async def _sample_chunks(n: int) -> list[str]:
    pool = await create_pool()
    try:
        rows = await DocumentRepository(pool).sample_chunks(n)
    finally:
        await pool.close()
    return [r["content"] for r in rows]


def _generate(chunks: list[str], size: int) -> list[GoldenRow]:
    llm = get_judge_llm()
    generator = TestsetGenerator(llm=llm, embedding_model=get_judge_embeddings(), persona_list=_PERSONAS)
    testset = generator.generate_with_langchain_docs(
        [Document(page_content=c) for c in chunks],
        testset_size=size,
        # Only the entities the single-hop synthesizer needs; default pipeline costs ~4 LLM calls per chunk
        transforms=[NERExtractor(llm=llm)],
        query_distribution=[(SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)],
        run_config=RUN_CONFIG,
    )
    if not isinstance(testset, Testset):
        raise TypeError(f"TestsetGenerator returned {type(testset).__name__}, expected Testset")
    chunk_vecs = np.array(embed_documents(chunks))
    rows: list[GoldenRow] = []
    for sample in testset.samples:
        s = sample.eval_sample
        if not isinstance(s, SingleTurnSample) or not isinstance(s.user_input, str):
            continue
        contexts = s.reference_contexts or []
        rows.append(GoldenRow(
            question=s.user_input,
            category="answerable",
            reference=s.reference or "",
            reference_contexts=contexts,
            reference_context_hashes=_source_hashes(contexts, chunks, chunk_vecs),
            synthesizer=sample.synthesizer_name,
        ))
    return rows


async def main(size: int, sample: int) -> None:
    chunks = await _sample_chunks(sample)
    if not chunks:
        raise SystemExit("No chunks in the database — ingest a site first (POST /ingest/url)")
    generated = await asyncio.to_thread(_generate, chunks, size)
    kept = [r for r in load_golden() if r.category != "answerable"] if GOLDEN_PATH.exists() else []
    save_golden(generated + kept)
    log.info("wrote %d generated + %d hand-written rows to %s", len(generated), len(kept), GOLDEN_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.size, args.sample))
