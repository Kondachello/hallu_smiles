"""Factories that connect the generic runner to the existing detector implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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


def build_controlled_shared_kggen_fake(
    config_path: str | Path = "config.yaml", *, cache_mode: str = "read_write",
    cache_root: str | Path | None = None, cache_sources: Iterable[Any] = (),
) -> tuple[dict[str, DetectorProtocol], Any]:
    """Offline integration fixture for the controlled shared-KGGen track only."""
    ensure_graph_eval_importable()
    import run
    from detector_adapters.hallugraph_adapter import HalluGraphAdapter
    from detector_adapters.shared_kggen_grapheval import SharedKGGenGraphEvalExtractor
    from graph_eval.detector import GraphEvalDetector
    from graph_eval.nli.fake import FakeNLI
    from src.config import load_config
    from src.extract import UsageLogger

    from .shared_graphs import SharedKGExtractorProxy, SharedKGGraphProvider

    cfg = load_config(config_path)
    if cache_root is not None:
        cfg._data["cache_dir"] = str(Path(cache_root))
        cfg.cache_dir = cfg._data["cache_dir"]
    provider = SharedKGGraphProvider(
        run.get_extractor(cfg, fake=True, usage=UsageLogger(None), cache_only=cache_mode == "cache_only"),
        sources=tuple(cache_sources), cache_mode=cache_mode,
    )
    proxy = SharedKGExtractorProxy(provider)
    hallu = HalluGraphAdapter(cfg, proxy, run.get_embedder(cfg, fake=True), alpha=0.7, relation_mode="strict")
    hallu.method_name = "hallugraph"  # type: ignore[attr-defined]
    hallu.variant_name = "shared_kggen_fake_offline_v1"  # type: ignore[attr-defined]
    hallu.shared_graph_ref = proxy.response_reference  # type: ignore[attr-defined]
    graph_extractor = SharedKGGenGraphEvalExtractor(provider)
    graph = GraphEvalDetector(graph_extractor, FakeNLI())
    graph.method_name = "grapheval"  # type: ignore[attr-defined]
    graph.variant_name = "shared_kggen_fake_offline_v1"  # type: ignore[attr-defined]
    graph.shared_graph_ref = graph_extractor.response_reference  # type: ignore[attr-defined]
    return {"hallugraph": hallu, "grapheval": graph}, provider


def build_real_detectors(
    *,
    hallugraph_config: str | Path,
    grapheval_config: dict,
    gateway_manifest_sha256: str | None,
    hallugraph_usage_path: str | Path | None = None,
) -> dict[str, DetectorProtocol]:
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
    usage = UsageLogger(hallugraph_usage_path)
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


def _configure_shared_cache(cfg: Any, *, cache_sources: Iterable[Any], cache_mode: str) -> None:
    """Apply framework cache policy without changing the source YAML on disk."""
    read_dirs = [] if cache_mode == "live_fresh" else [str(source.root) for source in cache_sources]
    cfg._data["cache_read_dirs"] = read_dirs
    cfg.cache_read_dirs = read_dirs


def build_controlled_shared_kggen_detectors(
    *,
    hallugraph_config: str | Path,
    grapheval_config: dict,
    gateway_manifest_sha256: str | None,
    cache_sources: Iterable[Any] = (),
    cache_mode: str = "read_through",
    hallugraph_usage_path: str | Path | None = None,
) -> tuple[dict[str, DetectorProtocol], Any]:
    """Build the controlled track with one shared KGGen response graph.

    The caller passes the returned provider to ``run_paired(...,
    shared_graph_provider=provider)``.  This explicit pairing prevents a caller
    from accidentally labelling an independently extracted GraphEval result as
    controlled.
    """
    ensure_graph_eval_importable()
    import run
    from detector_adapters.hallugraph_adapter import HalluGraphAdapter
    from detector_adapters.shared_kggen_grapheval import SharedKGGenGraphEvalExtractor
    from graph_eval.config import from_dict
    from graph_eval.detector import GraphEvalDetector
    from graph_eval.factory import build_extractor, build_nli
    from src.config import load_config
    from src.extract import UsageLogger

    if grapheval_config.get("extractor", {}).get("backend") != "shared_kggen":
        raise ValueError("controlled_shared_kggen_response_v1 requires grapheval.extractor.backend='shared_kggen'")
    sources = tuple(cache_sources)
    hallu_cfg = load_config(hallugraph_config)
    _configure_shared_cache(hallu_cfg, cache_sources=sources, cache_mode=cache_mode)
    usage = UsageLogger(hallugraph_usage_path)
    cache_only = cache_mode == "cache_only"
    base_extractor = run.get_extractor(hallu_cfg, fake=False, usage=usage, cache_only=cache_only)

    from .shared_graphs import SharedKGExtractorProxy, SharedKGGraphProvider

    provider = SharedKGGraphProvider(base_extractor, sources=sources, cache_mode=cache_mode)
    shared_proxy = SharedKGExtractorProxy(provider)
    hallu_detector = HalluGraphAdapter(
        hallu_cfg, shared_proxy, run.get_embedder(hallu_cfg, fake=False, cache_only=cache_only),
        alpha=float(hallu_cfg.metrics.alpha or 0.7), relation_mode="strict",
    )
    hallu_detector.method_name = "hallugraph"  # type: ignore[attr-defined]
    hallu_detector.variant_name = "shared_kggen_strict_v1"  # type: ignore[attr-defined]
    hallu_detector.shared_graph_ref = shared_proxy.response_reference  # type: ignore[attr-defined]

    graph_cfg = from_dict(grapheval_config)
    shared_extractor = SharedKGGenGraphEvalExtractor(provider)
    graph_detector = GraphEvalDetector(
        build_extractor(graph_cfg, manifest_sha256=gateway_manifest_sha256, injected_extractor=shared_extractor),
        build_nli(graph_cfg), paper_threshold=graph_cfg.nli.paper_threshold,
        aggregation=graph_cfg.nli.aggregation,
    )
    graph_detector.method_name = "grapheval"  # type: ignore[attr-defined]
    graph_detector.variant_name = "shared_kggen_hhem_v1"  # type: ignore[attr-defined]
    graph_detector.shared_graph_ref = shared_extractor.response_reference  # type: ignore[attr-defined]
    return {"hallugraph": hallu_detector, "grapheval": graph_detector}, provider
