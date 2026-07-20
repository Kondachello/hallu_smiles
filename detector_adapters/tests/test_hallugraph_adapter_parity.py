"""Parity: the adapter reproduces run.build_rows numbers exactly (fake offline path)."""
import run
from detector_adapters.hallugraph_adapter import HalluGraphAdapter
from graph_eval.types import STATUS_FAILED, STATUS_OK, DetectionInput
from src.config import load_config
from src.extract import UsageLogger
from src.metrics import score_response

CTX = "Alice knows Bob. Bob lives in Paris."
QUERY = "Where does Bob live?"
RESP = "Bob lives in Paris."
ALPHA = 0.7


def _fixture(tmp_path):
    cfg = load_config("config.yaml")
    cfg.cache_dir = str(tmp_path / "kg")
    cfg._data["cache_dir"] = cfg.cache_dir  # noqa: SLF001
    usage = UsageLogger(tmp_path / "usage.jsonl")
    extractor = run.get_extractor(cfg, True, usage)
    embedder = run.get_embedder(cfg, True)
    return cfg, extractor, embedder


def test_adapter_matches_run_build_rows(tmp_path):
    cfg, extractor, embedder = _fixture(tmp_path)
    # Baseline: exactly what run.score_all + run.build_rows compute (strict mode).
    gc, gq = extractor.extract_reference(CTX, QUERY)
    ga = extractor.extract(RESP, kind="response")
    refgraph = run.build_refgraph(cfg, gc, gq, embedder, None, None)
    sr = score_response(ga, refgraph, gc, gq, context=CTX, query=QUERY)
    record = {
        "response_id": "r1", "source_id": "s1", "task": "QA", "gen_model": "m",
        "split": "test", "y": 1, "context_len": len(CTX), "score": sr.to_dict(),
    }
    row = run.build_rows([record], ALPHA, ALPHA, "strict")[0]

    adapter = HalluGraphAdapter(cfg, extractor, embedder, alpha=ALPHA)
    res = adapter.predict(DetectionInput("r1", "s1", CTX, RESP, query=QUERY))

    assert res.status == STATUS_OK
    assert res.method == "hallugraph"
    assert res.raw_score == row["H_strict"]
    assert res.components["EG"] == row["EG"]
    assert res.components["RP_strict"] == row["RP_strict"]
    assert res.components["CFI"] == row["CFI_strict"]


def test_adapter_extraction_failure_is_failed_state(tmp_path):
    cfg, _extractor, embedder = _fixture(tmp_path)

    class Raises:
        def extract_reference(self, context, query):
            raise RuntimeError("boom")

        def extract(self, response, kind="response"):
            raise RuntimeError("boom")

    adapter = HalluGraphAdapter(cfg, Raises(), embedder, alpha=ALPHA)
    res = adapter.predict(DetectionInput("r1", "s1", CTX, RESP, query=QUERY))
    assert res.status == STATUS_FAILED
    assert res.raw_score is None
    assert res.failure["stage"] == "extraction"


def test_adapter_exposes_eg_rp_for_reweighting(tmp_path):
    cfg, extractor, embedder = _fixture(tmp_path)
    adapter = HalluGraphAdapter(cfg, extractor, embedder, alpha=ALPHA)
    res = adapter.predict(DetectionInput("r1", "s1", CTX, RESP, query=QUERY))
    assert "EG" in res.components and "RP_strict" in res.components
    if res.status == STATUS_OK and res.components["RP_strict"] is not None:
        eg, rp = res.components["EG"], res.components["RP_strict"]
        assert res.components["CFI"] == ALPHA * eg + (1 - ALPHA) * rp
