"""Unit tests for normalization, entity match() (exact/substring/embedding + stopword guard),
and relation align() (direction, relation-similarity)."""
from types import SimpleNamespace

import numpy as np

from src.matching import DictEmbedder, RefGraph, normalize


def mcfg(**over):
    base = dict(
        entity_sim_threshold=0.90,
        relation_sim_threshold=0.75,
        allow_substring_match=True,
        direction_sensitive_edges=True,
        inverse_edge_match=False,
        min_substring_chars=2,
        stopwords=["it", "the", "a", "is", "of"],
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------------------
def test_normalize_lowercase_and_strip_punct():
    assert normalize("  The Cat.  ") == "cat"
    # surrounding punctuation is stripped; internal punctuation is preserved (per spec).
    assert normalize("'Paris'") == "paris"
    assert normalize("Westfield Properties, LLC") == "westfield properties, llc"


def test_normalize_drops_leading_article_only():
    assert normalize("A dog") == "dog"
    assert normalize("the United States") == "united states"
    # internal 'the' is preserved
    assert normalize("king of the hill") == "king of the hill"


def test_normalize_collapses_whitespace():
    assert normalize("new    york\tcity") == "new york city"


# --------------------------------------------------------------------------------------
# entity match: exact
# --------------------------------------------------------------------------------------
def test_match_exact():
    rg = RefGraph(["Paris", "France"], [], mcfg(), embedder=None)
    m = rg.match_entity("paris")
    assert m.matched and m.method == "exact"


def test_no_match_unrelated():
    rg = RefGraph(["Paris", "France"], [], mcfg(), embedder=None)
    assert not rg.match_entity("berlin").matched


# --------------------------------------------------------------------------------------
# entity match: substring (with guards)
# --------------------------------------------------------------------------------------
def test_match_substring_llc_variants():
    rg = RefGraph(["Westfield Properties"], [], mcfg(), embedder=None)
    m = rg.match_entity("Westfield Properties LLC")
    assert m.matched and m.method == "substring"


def test_substring_stopword_guard_blocks_trivial():
    # 'it' is a stopword and must not substring-match into a longer ref entity.
    rg = RefGraph(["it is the law"], [], mcfg(), embedder=None)
    assert not rg.match_entity("it").matched


def test_substring_min_chars_guard():
    # 'ab' IS a token-boundary substring of 'ab cdef', but is blocked by min_substring_chars=3.
    rg = RefGraph(["ab cdef"], [], mcfg(min_substring_chars=3, stopwords=[]), embedder=None)
    assert not rg.match_entity("ab").matched
    # with min_substring_chars=2 it would match
    rg2 = RefGraph(["ab cdef"], [], mcfg(min_substring_chars=2, stopwords=[]), embedder=None)
    assert rg2.match_entity("ab").matched


def test_substring_requires_token_boundary():
    # 'cat' should not match 'category' (not a whole-token boundary match).
    rg = RefGraph(["category"], [], mcfg(), embedder=None)
    assert not rg.match_entity("cat").matched


# --------------------------------------------------------------------------------------
# entity match: embedding
# --------------------------------------------------------------------------------------
def test_match_embedding_path():
    # 'new york city' and 'nyc' are neither equal nor substrings -> embedding path.
    emb = DictEmbedder({"new york city": [1, 0, 0], "nyc": [1, 0, 0]}, dim=3)
    rg = RefGraph(["nyc"], [], mcfg(entity_sim_threshold=0.90), embedder=emb)
    m = rg.match_entity("new york city")
    assert m.matched and m.method == "embedding" and m.ref == "nyc"


def test_embedding_below_threshold_no_match():
    emb = DictEmbedder({"cat": [1, 0], "dog": [0, 1]}, dim=2)  # orthogonal -> cos 0
    rg = RefGraph(["dog"], [], mcfg(entity_sim_threshold=0.90), embedder=emb)
    assert not rg.match_entity("cat").matched


# --------------------------------------------------------------------------------------
# relation align: direction + relation similarity
# --------------------------------------------------------------------------------------
def test_align_exact_forward():
    rg = RefGraph(["france", "paris"], [("france", "has capital", "paris")],
                  mcfg(), embedder=None)
    a = rg.align_relation(("France", "has capital", "Paris"))
    assert a.matched and a.method == "forward"


def test_align_direction_sensitive_blocks_swap():
    rg = RefGraph(["france", "paris"], [("france", "has capital", "paris")],
                  mcfg(direction_sensitive_edges=True, inverse_edge_match=False), embedder=None)
    # swapped subject/object must NOT align when direction-sensitive
    a = rg.align_relation(("Paris", "has capital", "France"))
    assert not a.matched


def test_align_inverse_allowed_when_configured():
    rg = RefGraph(["france", "paris"], [("france", "has capital", "paris")],
                  mcfg(inverse_edge_match=True), embedder=None)
    a = rg.align_relation(("Paris", "has capital", "France"))
    assert a.matched and a.method == "inverse"


def test_align_relation_similarity_paraphrase():
    # 'pays rent to' ~ 'is obligated to pay' via embeddings (cos=1 here); endpoints exact.
    emb = DictEmbedder(
        {"pays rent to": [1, 0], "is obligated to pay": [1, 0]}, dim=2
    )
    rg = RefGraph(["tenant", "landlord"], [("tenant", "is obligated to pay", "landlord")],
                  mcfg(relation_sim_threshold=0.75), embedder=emb)
    a = rg.align_relation(("tenant", "pays rent to", "landlord"))
    assert a.matched and a.method == "forward"


def test_align_relation_dissimilar_predicate_fails():
    emb = DictEmbedder({"loves": [1, 0], "destroys": [0, 1]}, dim=2)
    rg = RefGraph(["a", "b"], [("a", "destroys", "b")],
                  mcfg(relation_sim_threshold=0.75), embedder=emb)
    a = rg.align_relation(("a", "loves", "b"))
    assert not a.matched
