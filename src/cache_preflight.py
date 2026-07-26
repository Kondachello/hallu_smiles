"""Deterministic, zero-network validation of a KG cache before a replay.

The cache key deliberately contains an LLM identity.  A DataSphere image rebuild
can change its *runtime* fingerprint without changing KGGen, the model, gateway,
or extraction contract.  When a recorded historical cache lineage is used, this
module proves that every graph needed by the fixed manifest is actually present
and structurally valid before the expensive scoring Job starts.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .cache import CacheOnlyMissError
from .data import Instance, unique_sources
from .extract import KGExtractor


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Return the canonical hash retained in the preflight diagnostic."""
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_kg_cache(
    cfg: Any,
    instances: Iterable[Instance],
    *,
    excluded_source_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Check each non-empty C/Q/A graph lookup using cache-only KGExtractor.

    No backend is constructed and no cache file is written.  The returned
    report is deliberately concrete enough to diagnose a wrong checkpoint,
    gateway identity, or extraction-contract drift without waiting for a
    threaded extraction stage to fail on its first miss.
    """
    rows = list(instances)
    excluded = {str(source_id) for source_id in excluded_source_ids}
    known_sources = set(unique_sources(rows))
    unknown_exclusions = sorted(excluded - known_sources)
    if unknown_exclusions:
        raise ValueError(
            "explicitly excluded source_id(s) are absent from the fixed manifest: "
            + ", ".join(unknown_exclusions)
        )
    analysis_rows = [row for row in rows if row.source_id not in excluded]
    extractor = KGExtractor(cfg, cache_only=True)
    checks: list[tuple[str, str, str, str]] = []
    for source_id, inst in sorted(unique_sources(analysis_rows).items()):
        checks.append(("context", source_id, "", inst.context))
        checks.append(("query", source_id, "", inst.query))
    for inst in sorted(analysis_rows, key=lambda item: (item.split, item.source_id, item.response_id)):
        checks.append(("response", inst.source_id, inst.response_id, inst.response))

    seen_keys: set[str] = set()
    misses: list[dict[str, str]] = []
    skipped_empty = 0
    for kind, source_id, response_id, text in checks:
        normalised = (text or "").strip()
        if not normalised:
            skipped_empty += 1
            continue
        key = extractor._cache_key(normalised)
        seen_keys.add(key)
        try:
            extractor.extract(normalised, kind=f"preflight_{kind}")
        except CacheOnlyMissError as exc:
            misses.append({
                "kind": kind,
                "source_id": source_id,
                "response_id": response_id,
                "cache_key": key,
                "expected_path": str(exc.path),
            })

    return {
        "protocol": "hallu-kg-cache-preflight-v1",
        "status": "ready" if not misses else "missing",
        "responses": len(rows),
        "sources": len(unique_sources(rows)),
        "analysis_responses": len(analysis_rows),
        "analysis_sources": len(unique_sources(analysis_rows)),
        "excluded_source_ids": sorted(excluded),
        "excluded_response_count": len(rows) - len(analysis_rows),
        "graph_slots": len(checks),
        "empty_graph_slots": skipped_empty,
        "unique_nonempty_cache_keys": len(seen_keys),
        "missing_count": len(misses),
        # Keep the artifact compact even if a completely wrong cache namespace
        # is supplied; the count and deterministic first keys remain enough to
        # identify the failure.
        "missing": misses[:50],
    }
