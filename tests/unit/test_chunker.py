from agent.embed import ENCODERS, count_tokens
from ingestion.chunker import chunk_raw
from ingestion.scraper import RawChunk


def test_every_chunk_fits_the_embedder_window():
    # Under the 1500-char cap but far over 512 tokens: OCR-like numbers and a run-on "word"
    dense = " ".join(f"{i}.{i * 7}" for i in range(150))
    text = f"# Table\n\n{dense}\n\n{'x1' * 700}"
    chunks = chunk_raw([RawChunk(content=text, metadata={})], "markdown_headers", [])

    assert len(chunks) > 1
    assert all(count_tokens(c.content, ENCODERS["BAAI/bge-small-en-v1.5"]) <= 512 for c in chunks)
    assert "".join(c.content for c in chunks).replace(" ", "").replace("\n", "") == \
        text.removeprefix("# Table").replace(" ", "").replace("\n", "")


def test_chunks_do_not_depend_on_the_search_encoder(monkeypatch):
    text = "# A\n\n" + " ".join(f"word{i}" for i in range(600))
    before = chunk_raw([RawChunk(content=text, metadata={})], "markdown_headers", [])
    monkeypatch.setattr("agent.embed.get_settings", lambda: type("S", (), {"embedder": "sentence-transformers/all-MiniLM-L6-v2"})())
    after = chunk_raw([RawChunk(content=text, metadata={})], "markdown_headers", [])
    assert [c.content for c in before] == [c.content for c in after]


def test_a_long_table_splits_on_rows_and_repeats_its_header():
    rows = "\n".join(f"|ravim {i}|{i} mg/kg|veenisiseselt aeglaselt|" for i in range(80))
    text = f"# Lisa\n\n|Ravim|Annus|Manustamine|\n|---|---|---|\n{rows}"
    chunks = chunk_raw([RawChunk(content=text, metadata={})], "markdown_headers", [])

    assert len(chunks) > 1
    for c in chunks:
        lines = c.content.splitlines()
        assert lines[:2] == ["|Ravim|Annus|Manustamine|", "|---|---|---|"]
        assert all(line.startswith("|") and line.endswith("|") for line in lines)  # no row cut in half
    assert sum(len(c.content.splitlines()) - 2 for c in chunks) == 80
