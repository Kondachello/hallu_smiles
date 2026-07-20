import json

from graph_eval.parser import (
    STATUS_EMPTY,
    STATUS_MALFORMED,
    STATUS_OK,
    parse_triples,
)
from graph_eval.types import PARSE_DUPLICATE, PARSE_INVALID, PARSE_OK


def test_accepts_ordered_triples_of_three_nonempty_strings():
    out = parse_triples(json.dumps([["a", "rel", "b"], ["c", "rel2", "d"]]))
    assert out.status == STATUS_OK
    assert len(out.valid_triples) == 2
    assert out.valid_triples[0].as_tuple() == ("a", "rel", "b")
    assert out.invalid_count == 0


def test_malformed_output_keeps_raw_and_reports():
    raw = "not json at all {"
    out = parse_triples(raw)
    assert out.status == STATUS_MALFORMED
    assert out.raw_output == raw
    assert out.triples == ()
    assert out.error


def test_empty_list_is_empty_status():
    out = parse_triples("[]")
    assert out.status == STATUS_EMPTY
    assert out.valid_triples == ()


def test_invalid_items_are_flagged_not_dropped():
    out = parse_triples(json.dumps([["a", "b"], ["x", "y", "z"], ["only-one"]]))
    assert out.status == STATUS_OK
    assert out.invalid_count == 2
    statuses = [t.parse_status for t in out.triples]
    assert statuses == [PARSE_INVALID, PARSE_OK, PARSE_INVALID]
    assert len(out.valid_triples) == 1


def test_duplicates_flagged_with_duplicate_of():
    out = parse_triples(json.dumps([["A", "is", "B"], [" a ", "IS", "b"]]))
    assert [t.parse_status for t in out.triples] == [PARSE_OK, PARSE_DUPLICATE]
    assert out.triples[1].duplicate_of == out.triples[0].triple_id
    # duplicates are not re-verified
    assert len(out.valid_triples) == 1


def test_object_form_with_triples_key():
    out = parse_triples(json.dumps({"triples": [["a", "r", "b"]]}))
    assert out.status == STATUS_OK
    assert len(out.valid_triples) == 1


def test_blank_string_is_invalid():
    out = parse_triples(json.dumps([["a", "  ", "b"]]))
    assert out.invalid_count == 1
    assert out.valid_triples == ()
