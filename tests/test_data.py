"""Unit tests for RAGTruth loading, (C,Q,A) construction, and the Data2txt serializer."""
from src.data import (
    serialize_business_json,
    build_context_query,
    response_label,
)


# --------------------------------------------------------------------------------------
# Data2txt JSON serializer
# --------------------------------------------------------------------------------------
def test_serialize_basic_scalars_and_booleans():
    info = {
        "name": "Deja Vu Cafe IV",
        "city": "Reno",
        "attributes": {"RestaurantsReservations": False, "OutdoorSeating": True},
    }
    out = serialize_business_json(info)
    assert "The business is named Deja Vu Cafe IV." in out
    assert "Deja Vu Cafe IV has city = Reno." in out
    assert "Deja Vu Cafe IV has attributes.RestaurantsReservations = false." in out
    assert "Deja Vu Cafe IV has attributes.OutdoorSeating = true." in out


def test_serialize_null_is_unknown_not_false():
    # RAGTruth null-trap: null must render literally, NOT as 'no'/'false'.
    info = {"name": "X", "attributes": {"Music": None}}
    out = serialize_business_json(info)
    assert "X has attributes.Music = null (unknown)." in out
    assert "= false." not in out
    assert "= no." not in out


def test_serialize_nested_dotted_keys():
    info = {"name": "X", "attributes": {"BusinessParking": {"garage": False, "lot": True}}}
    out = serialize_business_json(info)
    assert "X has attributes.BusinessParking.garage = false." in out
    assert "X has attributes.BusinessParking.lot = true." in out


def test_serialize_reviews_appended_verbatim():
    info = {
        "name": "X",
        "review_info": [
            {"review_stars": 1.0, "review_text": "Rude staff. Very bad!"},
            {"review_stars": 5.0, "review_text": "Nice and clean."},
        ],
    }
    out = serialize_business_json(info)
    assert "Customer reviews:" in out
    assert "Rude staff. Very bad!" in out
    assert "Nice and clean." in out
    # verbatim: the exact review text is preserved
    assert out.rstrip().endswith("Nice and clean.")


def test_serialize_is_deterministic():
    info = {"name": "X", "a": 1, "b": {"c": True, "d": None}}
    assert serialize_business_json(info) == serialize_business_json(info)


# --------------------------------------------------------------------------------------
# (C, Q, A) construction
# --------------------------------------------------------------------------------------
def test_build_cqa_qa():
    src = {
        "task_type": "QA",
        "source_info": {"question": "how to boil water", "passages": "passage 1: heat it."},
    }
    c, q = build_context_query(src)
    assert c == "passage 1: heat it."
    assert q == "how to boil water"


def test_build_cqa_data2txt_has_empty_query():
    src = {"task_type": "Data2txt", "source_info": {"name": "Subway", "city": "SB"}}
    c, q = build_context_query(src)
    assert q is None  # Data2txt -> G_q = empty
    assert "Subway has city = SB." in c


def test_build_cqa_summary_has_empty_query():
    src = {"task_type": "Summary", "source_info": "An article about cats."}
    c, q = build_context_query(src)
    assert q is None
    assert c == "An article about cats."


# --------------------------------------------------------------------------------------
# Response-level labels
# --------------------------------------------------------------------------------------
def test_label_positive_when_span_present():
    rec = {"labels": [{"start": 1, "end": 2, "label_type": "Evident Conflict"}]}
    y, types = response_label(rec)
    assert y == 1 and types == ["Evident Conflict"]


def test_label_negative_when_no_spans():
    y, types = response_label({"labels": []})
    assert y == 0 and types == []


def test_label_implicit_true_counts_by_default():
    rec = {"labels": [{"label_type": "Subtle Baseless Info", "implicit_true": True}]}
    y, _ = response_label(rec, exclude_implicit_true=False)
    assert y == 1


def test_label_implicit_true_can_be_excluded():
    rec = {"labels": [{"label_type": "Subtle Baseless Info", "implicit_true": True}]}
    y, _ = response_label(rec, exclude_implicit_true=True)
    assert y == 0


def test_label_due_to_null_always_counts():
    rec = {"labels": [{"label_type": "Evident Baseless Info", "due_to_null": True}]}
    y, _ = response_label(rec, exclude_implicit_true=True)
    assert y == 1  # due_to_null is not excludable
