"""Build detector backends from config.  Never imports torch/openai at module load."""
from __future__ import annotations

from .cache import JsonCache
from .config import GraphEvalConfig
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
