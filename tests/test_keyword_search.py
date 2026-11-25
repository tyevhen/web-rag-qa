"""
Integration test for keyword search (migrations/011_keyword_bm25.sql). Requires the migrated database with the
Ole valmis Estonian chunks ingested.
"""
from db.client import create_pool
from db.repositories.documents import DocumentRepository


async def test_estonian_terms_and_search() -> None:
    pool = await create_pool()
    try:
        terms = await pool.fetchval("SELECT tsvector_to_array(keyword_vector('et', 'Mida teha, kui leian drooni ja droonist?'))")
        assert terms == ["droon", "leian"]  # function words dropped, inflected forms share one term
        rows = await DocumentRepository(pool).fulltext_search("Mida teha, kui leian allakukkunud drooni?", 6, "et")
        assert rows and all(r["metadata"]["lang"] == "et" for r in rows)
        assert "droon" in rows[0]["content"].lower()
        assert await DocumentRepository(pool).fulltext_search("ja kui", 6, "et") == []  # only function words
    finally:
        await pool.close()
