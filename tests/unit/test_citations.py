from agent.nodes.formatter import text_fragment
from dashboard.analysis.citations import chunk_urls, link_markers


def test_cited_markers_link_to_their_chunk_and_others_stay_plain():
    urls = chunk_urls([{"n": 2, "chunk_id": "c2"}, "[1] | url: old row"], "http://api")
    assert link_markers("A [2], B [1], C [9].", urls) == "A [[2]](http://api/chunks/c2), B [1], C [9]."


def test_text_fragment_takes_plain_leading_words():
    assert text_fragment("## Boil the water for one-two minutes, then\nmore") == (
        "#:~:text=Boil%20the%20water%20for%20one%2Dtwo%20minutes%2C")


def test_text_fragment_skipped_when_markup_comes_too_early():
    assert text_fragment("**Answers** **What kind") == ""
