from ingestion.evaluator import evaluate_chunks
from ingestion.scraper import RawChunk


def test_a_table_of_numbers_is_kept_but_bare_numbers_are_not():
    rows = "\n".join(f"|Hingamissagedus (×/min)|{35 - i}|{25 - i}|{20 - i}|{16 - i}|" for i in range(6))
    table = f"|Näitaja|Vastsündinu|1-aastane|8-aastane|Täiskasvanu|\n|---|---|---|---|---|\n{rows}"
    numbers = " ".join(str(n) for n in range(100, 140))
    assert [c.content for c in evaluate_chunks([RawChunk(table, {}), RawChunk(numbers, {})], ["et"])] == [table]
