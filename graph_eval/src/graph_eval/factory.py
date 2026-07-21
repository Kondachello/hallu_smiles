"""Build detector backends from config.  Never imports torch/openai at module load."""
from __future__ import annotations

from .cache import JsonCache
from .config import GraphEvalConfig
from .extraction.cached import CachedExtractor, extraction_identity
from .extraction.fake import FakeExtractor
from .nli.cached import CachedNLI
from .nli.fake import FakeNLI
from .verbalize import VERBALIZER_VERSION


def _nli_identity(cfg: GraphEvalConfig) -> dict:
    nli = cfg.nli
    return {
        "nli_model": nli.model,
        "nli_model_revision": nli.revision,
        "nli_model_label": nli.model_label,
        "evidence_policy": nli.evidence_policy,
        "verbalizer_version": VERBALIZER_VERSION,
    }


def build_nli(cfg: GraphEvalConfig, *, cache: bool = True):
    """Return an NLIModel for ``cfg.nli.backend`` (fake | hhem), cache-wrapped."""
    backend = cfg.nli.backend
    if backend == "fake":
        inner = FakeNLI()
        identity = {**_nli_identity(cfg), "nli_model": "fake"}
    elif backend == "hhem":
        from .nli.hhem import HHEMNLIModel  # imports nothing heavy at construction

        inner = HHEMNLIModel(cfg.nli)
        identity = _nli_identity(cfg)
    else:  # pragma: no cover - guarded by config.validate()
        raise ValueError(f"unknown nli.backend: {backend!r}")

    if not cache:
        return inner
    store = JsonCache(cfg.cache_dir, cfg.nli.cache_namespace, cache_only=cfg.cache_only)
    return CachedNLI(inner, store, identity=identity)


def build_extractor(
    cfg: GraphEvalConfig, *, client=None, manifest_sha256: str | None = None, cache: bool = True,
    injected_extractor=None,
):
    """Return an Extractor for ``cfg.extractor.backend``, cache-wrapped when owned here.

    ``shared_kggen`` is deliberately injection-only: its graph belongs to the
    experiment-level shared preprocessing stage, so GraphEval must neither create a
    second LLM client nor wrap the common cache in its private cache namespace.
    """
    backend = cfg.extractor.backend
    if backend == "fake":
        inner = FakeExtractor()
        identity = {**extraction_identity(cfg.extractor, manifest_sha256), "model": "fake"}
    elif backend == "gateway":
        from .extraction.gateway import GatewayExtractor  # lazy: no openai at load

        inner = GatewayExtractor(cfg.extractor, client=client, manifest_sha256=manifest_sha256)
        identity = extraction_identity(cfg.extractor, manifest_sha256)
    elif backend == "vllm":  # pragma: no cover - plan section 7.4, not yet implemented
        raise NotImplementedError("local vLLM extractor is a separate gated config (plan 7.4)")
    elif backend == "shared_kggen":
        if injected_extractor is None:
            raise ValueError("extractor.backend='shared_kggen' requires injected_extractor")
        return injected_extractor
    else:  # pragma: no cover - guarded by config.validate()
        raise ValueError(f"unknown extractor.backend: {backend!r}")

    if not cache:
        return inner
    store = JsonCache(cfg.cache_dir, cfg.extractor.cache_namespace, cache_only=cfg.cache_only)
    return CachedExtractor(inner, store, identity=identity)
