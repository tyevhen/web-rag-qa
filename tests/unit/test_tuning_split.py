from tuning.split import split


def test_a_source_passage_is_never_on_both_sides():
    rows = [{"meta": {"source_chunk": f"c{i % 40}"}, "n": i} for i in range(120)]  # 3 rows per passage

    train, valid = split(rows, valid_share=0.1)

    assert {r["meta"]["source_chunk"] for r in train}.isdisjoint(r["meta"]["source_chunk"] for r in valid)
    assert len(valid) == 12 and len(train) + len(valid) == 120
