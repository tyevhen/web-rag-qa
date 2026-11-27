from tuning.checks import answer_problem, is_pure_refusal, near_duplicate


def test_answer_problem() -> None:
    assert answer_problem("Varu vähemalt 2 liitrit vett inimese kohta ööpäevas [1].", 1, "et") is None
    assert answer_problem("Varu vähemalt 2 liitrit vett inimese kohta ööpäevas [3].", 2, "et").startswith("cites")
    assert answer_problem("Varu **vähemalt 2 liitrit** vett inimese kohta ööpäevas [1].", 1, "et") is None
    assert answer_problem("Vastavalt allikale [1] tuleks varuda 2 liitrit vett.", 1, "et").startswith("style")
    assert answer_problem("Store at least two litres of water per person per day [1].", 1, "et").startswith("language")
    assert answer_problem("Varu vähemalt kaks liitrit vett inimese kohta ööpäevas.", 1, "et") == "answer without citations"
    assert answer_problem("Allikad ei käsitle žguti pealehoidmise aega.", 2, "et") is None  # a refusal needs no [n]


def test_only_a_one_sentence_refusal_skips_the_judge() -> None:
    assert is_pure_refusal("See teave ei kata küsimust laste vigastuste kohta.")
    assert not is_pure_refusal("See teave ei kata küsimust. Elektrilöögi korral helista kohe 112.")
    assert not is_pure_refusal("Helista kohe 112 [1].")


def test_near_duplicate_is_about_wording() -> None:
    golden = ["Mitmeks ajaks peaksin elektrikatkestuseks kodus varusid hoidma?"]
    assert near_duplicate("Mitmeks ajaks peaks elektrikatkestuseks kodus varusid hoidma?", golden)
    assert not near_duplicate("Mida peaks kodus olema, et olla valmis kriisiolukorraks?", golden)
