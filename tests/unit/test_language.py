from api.schemas import IngestUrlRequest
from ingestion import evaluator
from ingestion.evaluator import evaluate_chunks, filter_pages
from ingestion.scraper import RawChunk, _excluded

_EN = ("In an emergency, keep calm and follow the instructions of the authorities. Prepare drinking water, "
       "food and medicine for at least one week, and keep a battery radio at home so you can hear the news.")
_ET = ("Kriisiolukorras jää rahulikuks ja järgi ametiasutuste juhiseid. Varu vähemalt üheks nädalaks joogivett, "
       "toitu ja ravimeid ning hoia kodus patareidega raadiot, et kuulda uudiseid ka elektrikatkestuse ajal.")


def _page(text: str) -> RawChunk:
    return RawChunk(content=text, metadata={"url": "https://x"})


def test_estonian_ingest_rejects_english_pages_and_tags_the_rest():
    kept = filter_pages([_page(_EN), _page(_ET)], ["et"])
    assert [p.content for p in kept] == [_ET]
    assert kept[0].metadata["lang"] == "et"


def test_default_keeps_english_only():
    kept = filter_pages([_page(_EN), _page(_ET)], IngestUrlRequest(url="https://x").langs)
    assert [(p.content, p.metadata["lang"]) for p in kept] == [(_EN, "en")]
    assert [c.content for c in evaluate_chunks([_page(_EN), _page(_ET)], ["en"])] == [_EN]


def test_undetectable_chunk_is_kept_only_for_estonian(monkeypatch):
    monkeypatch.setattr(evaluator, "detect_lang", lambda text: None)
    assert evaluate_chunks([_page(_ET)], ["et"])
    assert evaluate_chunks([_page(_EN)], ["en"]) == []


def test_excluded_paths_match_whole_segments():
    skip = ["/en", "/et-simple"]
    assert _excluded("https://www.olevalmis.ee/en/seeking-shelter", skip)
    assert _excluded("https://www.olevalmis.ee/en", skip)
    assert not _excluded("https://www.olevalmis.ee/energia", skip)
    assert not _excluded("https://www.olevalmis.ee/varjumine", skip)
