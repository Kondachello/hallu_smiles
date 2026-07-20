"""Per-response caching wrapper around any Extractor.

Cache key (plan section 8) = extraction identity (prompt profile+version, schema
version, output mode, logical model, gateway manifest SHA-256, decoding params) +
the canonical response text.  A warm ``cache_only`` replay makes zero extractor
calls; a miss raises via :class:`~graph_eval.cache.CacheOnlyMissError`.
"""
from __future__ import annotations

from ..cache import JsonCache, make_key
from .base import ExtractionOutput
from .prompt import PROMPT_VERSION, SCHEMA_VERSION


def extraction_identity(extractor_cfg, manifest_sha256: str | None) -> dict:
    return {
        "prompt_profile": extractor_cfg.prompt_profile,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "output_mode": extractor_cfg.output_mode,
        "model": extractor_cfg.model,
        "gateway_manifest_sha256": manifest_sha256,
        "temperature": extractor_cfg.temperature,
        "max_tokens": extractor_cfg.max_tokens,
    }


class CachedExtractor:
    def __init__(self, inner, cache: JsonCache, *, identity: dict):
        self.inner = inner
        self.cache = cache
        self.identity = dict(identity)
        self.prompt_profile = getattr(inner, "prompt_profile", "unknown")

    def extract(self, response_text: str) -> ExtractionOutput:
        key = make_key(self.identity, response_text)
        cached = self.cache.get(key)  # raises in cache_only when absent
        if cached is not None:
            return ExtractionOutput(
                raw_output=cached["raw_output"],
                usage={"extractor_calls": 0, "extraction_cache_hit": True},
            )
        out = self.inner.extract(response_text)
        self.cache.put(key, {"raw_output": out.raw_output})
        return ExtractionOutput(
            raw_output=out.raw_output,
            usage={**out.usage, "extraction_cache_hit": False},
        )
