from graph_eval.extraction.prompt import (
    STRUCTURED_RESPONSE_FORMAT,
    build_messages,
    repair_messages,
)


def test_build_messages_is_answer_only():
    msgs = build_messages("THE ANSWER TEXT", "paper_prompt")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "THE ANSWER TEXT"  # only the answer, verbatim
    assert "triple" in msgs[0]["content"].lower()


def test_structured_response_format_is_strict_json_schema():
    assert STRUCTURED_RESPONSE_FORMAT["type"] == "json_schema"
    js = STRUCTURED_RESPONSE_FORMAT["json_schema"]
    assert js["strict"] is True
    items = js["schema"]["properties"]["triples"]["items"]
    assert items["minItems"] == 3 and items["maxItems"] == 3


def test_repair_appends_prior_raw_and_instruction():
    base = build_messages("A", "paper_prompt")
    rep = repair_messages(base, "garbage")
    assert rep[-2] == {"role": "assistant", "content": "garbage"}
    assert rep[-1]["role"] == "user"
