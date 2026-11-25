from api.errors import _question_of, describe_errors


def test_length_limits_read_as_sizes():
    long_q = {"type": "string_too_long", "loc": ("body", "question"), "input": "x" * 41203,
              "ctx": {"max_length": 2000}, "msg": "String should have at most 2000 characters"}
    many = {"type": "too_long", "loc": ("body", "history"), "input": [{}] * 12,
            "ctx": {"field_type": "List", "max_length": 10, "actual_length": 12}, "msg": "List too long"}
    assert describe_errors([long_q]) == "question: 41,203 chars > 2,000"
    assert describe_errors([many]) == "history: 12 items > 10"


def test_other_errors_keep_pydantic_message():
    extra = {"type": "extra_forbidden", "loc": ("body", "history"), "input": [], "msg": "Extra inputs are not permitted"}
    missing = {"type": "missing", "loc": ("body", "question"), "input": {}, "msg": "Field required"}
    bad_json = {"type": "json_invalid", "loc": ("body", 7), "input": {}, "msg": "JSON decode error",
                "ctx": {"error": "Expecting value"}}
    assert describe_errors([extra]) == "history: Extra inputs are not permitted"
    assert describe_errors([missing, bad_json]) == "question: Field required; 7: JSON decode error"
    assert describe_errors([{"loc": ("body",), "msg": "bad"}]) == "body: bad"


def test_question_from_parsed_or_raw_body():
    assert _question_of({"question": "q"}) == "q"
    assert _question_of('{"question": ') == '{"question": '
    assert _question_of({"question": 5}) == _question_of(None) == ""
