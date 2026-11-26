from ingestion.pdf import assign_pages, clean_page, merge_short
from ingestion.scraper import RawChunk


def test_soft_hyphenated_words_are_joined_from_the_raw_text():
    raw = "tõttu kõige ini\xad\nmesi ja trauma\xadkeskusesse"
    md = "tõttu kõige ini mesi ja trauma keskusesse"
    assert clean_page(md, raw) == "tõttu kõige inimesi ja traumakeskusesse"


def test_layout_noise_is_removed():
    md = "# **T E K E** \n\n<u>ESMAABI VIGASTUSTE LIIKIDE JA PÕHJUSTE JÄRGI</u> **<u>35</u>**\n\nüks<br>kaks <mark>morf **i** n</mark>\n<!-- Start of picture text -->\nFAST<br>Positiivne\n<!-- End of picture text -->\n"
    assert clean_page(md, "anna morfiin").split() == ["üks", "kaks", "morfiin"]


def test_mixed_case_headings_are_upper_cased_and_captions_stay_text():
    md = "# **2. TrAuMA MEHHANisM JA EPidEMioLooGiA**\n\n## Röntgenuuring\n\n# Tabel 6.3. LEMON\n"
    assert clean_page(md, "").splitlines() == [
        "# 2. TRAUMA MEHHANISM JA EPIDEMIOLOOGIA", "", "## Röntgenuuring", "", "**Tabel 6.3. LEMON**"]


def test_each_chunk_gets_the_page_it_starts_on():
    url = "https://x.ee/a.pdf"
    chunks = [RawChunk("<!-- page 2 -->\nesimene\n<!-- page 3 -->\njätkub", {"section": ""}),
              RawChunk("teine lõik lehel 3\n<!-- page 5 -->", {"section": "B"}),
              RawChunk("kolmas", {"section": "C"})]
    out = assign_pages(chunks, url)
    assert [c.metadata["url"] for c in out] == [f"{url}#page=2", f"{url}#page=3", f"{url}#page=5"]
    assert out[0].content == "esimene\njätkub" and "<!--" not in out[1].content


def test_a_short_section_is_kept_by_merging_it_into_the_next():
    long = " ".join(["sõna"] * 30)
    chunks = [RawChunk("Peata kõigepealt väline verejooks.", {"section": "Verejooks", "url": "u#page=4"}),
              RawChunk(long, {"section": "Hingamine", "url": "u#page=5"})]
    [merged] = merge_short(chunks)
    assert merged.content == f"**Verejooks**\nPeata kõigepealt väline verejooks.\n\n{long}"
    assert merged.metadata == {"section": "Hingamine", "url": "u#page=4"}
