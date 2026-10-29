from agent.embed import count_tokens, get_encoder
from ingestion.chunker import chunk_raw
from ingestion.scraper import RawChunk


def test_every_chunk_fits_the_embedder_window():
    # Under the 1500-char cap but far over 512 tokens: OCR-like numbers and a run-on "word"
    dense = " ".join(f"{i}.{i * 7}" for i in range(150))
    text = f"# Table\n\n{dense}\n\n{'x1' * 700}"
    chunks = chunk_raw([RawChunk(content=text, metadata={})], "markdown_headers", [])

    assert len(chunks) > 1
    assert all(count_tokens(c.content) <= get_encoder().max_tokens for c in chunks)
    assert "".join(c.content for c in chunks).replace(" ", "").replace("\n", "") == \
        text.removeprefix("# Table").replace(" ", "").replace("\n", "")
