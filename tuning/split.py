"""Train/validation files for mlx_lm.lora from the exported examples (see tuning/README.md).

    uv run python -m tuning.split --tokenizer "$ROOT/base-4bit" --out "$ROOT/data"

Rows are split by source passage, so a passage the model trained on never scores validation.
Rows longer than --max-tokens (default MAX_TOKENS) are dropped: mlx would cut them at the end, which is the answer.
"""
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

MAX_TOKENS = 3072  # the longest that trains on 16 GB with margin (tuning/README.md)
_SOURCES = [Path("tuning/data/et.jsonl"), Path("tuning/data/en.jsonl")]


def split(rows: list[dict], valid_share: float = 0.05, seed: int = 0) -> tuple[list[dict], list[dict]]:
    chunks = sorted({r["meta"]["source_chunk"] for r in rows})
    valid = set(random.Random(seed).sample(chunks, round(len(chunks) * valid_share)))
    return ([r for r in rows if r["meta"]["source_chunk"] not in valid],
            [r for r in rows if r["meta"]["source_chunk"] in valid])


def main(tokenizer_path: Path, out: Path, max_tokens: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    rows = [json.loads(line) for f in _SOURCES for line in f.read_text().splitlines()]
    # tokenize=True returns a BatchEncoding here, whose len() is 2: count the rendered text instead
    fits = [r for r in rows if len(tokenizer.encode(tokenizer.apply_chat_template(r["messages"], tokenize=False))) <= max_tokens]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "valid"), split(fits)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in part))
        print(f"{name}: {len(part)}")
    print(f"dropped over {max_tokens} tokens: {len(rows) - len(fits)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True, help="the base model folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="4096 on a cloud GPU (Colab)")
    args = parser.parse_args()
    main(args.tokenizer, args.out, args.max_tokens)
