"""Factories that connect the generic runner to the existing detector implementations."""
from __future__ import annotations

from pathlib import Path

from ._compat import ensure_graph_eval_importable
from .contracts import DetectorProtocol


def build_grapheval_fake() -> DetectorProtocol:
    """Build the real GraphEval facade over its deterministic offline backends."""
    ensure_graph_eval_importable()
    from graph_eval.detector import GraphEvalDetector
    from graph_eval.extraction.fake import FakeExtractor
    from graph_eval.nli.fake import FakeNLI

    detector = GraphEvalDetector(FakeExtractor(), FakeNLI())
    detector.method_name = "grapheval"  # type: ignore[attr-defined]
    detector.variant_name = "fake_offline_v1"  # type: ignore[attr-defined]
    return detector


def build_hallugraph_fake(config_path: str | Path = "config.yaml") -> DetectorProtocol:
    """Build the real HalluGraph adapter with FakeKGGen/DictEmbedder.

    This creates no network calls.  It is a framework integration aid; it is not a
    scientific live run and must remain labelled ``fake_offline_v1`` in artifacts.
    """
    import run
    from detector_adapters.hallugraph_adapter import HalluGraphAdapter
    from src.config import load_config
    from src.extract import UsageLogger

    cfg = load_config(config_path)
    usage = UsageLogger(None)
    adapter = HalluGraphAdapter(
        cfg,
        run.get_extractor(cfg, fake=True, usage=usage),
        run.get_embedder(cfg, fake=True),
        alpha=0.7,
        relation_mode="strict",
    )
    adapter.method_name = "hallugraph"  # type: ignore[attr-defined]
    adapter.variant_name = "fake_offline_v1"  # type: ignore[attr-defined]
    return adapter


def build_real_detectors(*, hallugraph_config: str | Path, grapheval_config: dict, gateway_manifest_sha256: str | None) -> dict[str, DetectorProtocol]:
    """Construct real backends only when the caller explicitly supplies live config.

    The function never reads credentials itself; the existing detector factories receive
    only environment-variable names from their validated configurations.
    """
    ensure_graph_eval_importable()
    import run
    from detector_adapters.hallugraph_adapter import HalluGraphAdapter
    from graph_eval.config import from_dict
    from graph_eval.detector import GraphEvalDetector
    from graph_eval.factory import build_extractor, build_nli
    from src.config import load_config
    from src.extract import UsageLogger

    graph_cfg = from_dict(grapheval_config)
    graph_detector = GraphEvalDetector(
        build_extractor(graph_cfg, manifest_sha256=gateway_manifest_sha256),
        build_nli(graph_cfg),
        paper_threshold=graph_cfg.nli.paper_threshold,
        aggregation=graph_cfg.nli.aggregation,
    )
    graph_detector.method_name = "grapheval"  # type: ignore[attr-defined]
    graph_detector.variant_name = "configured_live_v1"  # type: ignore[attr-defined]

    hallu_cfg = load_config(hallugraph_config)
    usage = UsageLogger(None)
    hallu_detector = HalluGraphAdapter(
        hallu_cfg,
        run.get_extractor(hallu_cfg, fake=False, usage=usage),
        run.get_embedder(hallu_cfg, fake=False),
        alpha=float(hallu_cfg.metrics.alpha or 0.7),
        relation_mode="strict",
    )
    hallu_detector.method_name = "hallugraph"  # type: ignore[attr-defined]
    hallu_detector.variant_name = "configured_live_v1"  # type: ignore[attr-defined]
    return {"hallugraph": hallu_detector, "grapheval": graph_detector}
