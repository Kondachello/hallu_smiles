"""Auditable KGGen graph sharing and read-only cache inspection.

This module deliberately sits in the experiment framework.  GraphEval remains
independent from HalluGraph; it receives an ordinary ``Extractor`` adapter while
this module owns the cross-method provenance contract.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.extract import (
    CACHE_KEY_SCHEMA_CURRENT,
    CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY,
    Graph,
)

from .artifacts import canonical_json, sha256_bytes

CACHE_PROTOCOL = "hallu-kg-cache-v2"


class CacheIntegrityError(ValueError):
    """A cache entry cannot be safely reused."""


class CachePreflightError(RuntimeError):
    """A cache-only run has incomplete or incompatible graph coverage."""


@dataclass(frozen=True)
class GraphCacheSource:
    """An immutable source of previously extracted KGGen graphs."""

    source_id: str
    root: Path
    read_only: bool = True
    lineage_manifest: Path | None = None
    priority: int = 0
    cache_key_compatibility: tuple[str, ...] = ()


@dataclass(frozen=True)
class SharedGraphArtifact:
    graph_id: str
    role: str
    cache_key: str | None
    cache_key_schema: str | None
    graph_sha256: str
    input_sha256: str
    graph: Graph
    source_id: str
    cache_hit: bool
    extraction_identity: Mapping[str, Any]

    def reference(self) -> dict[str, str]:
        return {
            "shared_graph_id": self.graph_id,
            "shared_graph_sha256": self.graph_sha256,
            "shared_graph_cache_key": self.cache_key or "",
            "shared_graph_cache_key_schema": self.cache_key_schema or "",
            "shared_graph_source": self.source_id,
        }

    def record(self) -> dict[str, Any]:
        return {
            **self.reference(),
            "role": self.role,
            "input_sha256": self.input_sha256,
            "cache_hit": self.cache_hit,
            "entities": sorted(self.graph.entities),
            "relations": sorted([list(row) for row in self.graph.relations]),
            "extraction_identity": dict(self.extraction_identity),
        }


def _graph_from_envelope(path: Path, *, expected_key: str | None = None) -> tuple[str, Graph, str]:
    """Validate one exact ``hallu-kg-cache-v2`` envelope without silent fallback."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CacheIntegrityError(f"invalid JSON cache entry: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != {"protocol", "cache_key", "graph", "graph_sha256"}:
        raise CacheIntegrityError(f"invalid cache envelope shape: {path}")
    key = raw.get("cache_key")
    if not isinstance(key, str) or (expected_key is not None and key != expected_key):
        raise CacheIntegrityError(f"cache key mismatch: {path}")
    if raw.get("protocol") != CACHE_PROTOCOL or path.stem != key:
        raise CacheIntegrityError(f"cache protocol or filename mismatch: {path}")
    payload = raw.get("graph")
    if not isinstance(payload, dict) or set(payload) != {"entities", "relations"}:
        raise CacheIntegrityError(f"invalid graph payload: {path}")
    digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if raw.get("graph_sha256") != digest:
        raise CacheIntegrityError(f"graph SHA-256 mismatch: {path}")
    graph = Graph.from_dict(payload)
    if graph.to_dict() != payload:
        raise CacheIntegrityError(f"graph payload does not round-trip: {path}")
    return key, graph, digest


def inspect_cache_sources(sources: Iterable[GraphCacheSource]) -> dict[str, Any]:
    """Inspect all entries and surface conflicts before a detector is called."""
    entries: dict[str, list[dict[str, Any]]] = {}
    invalid: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: (-item.priority, item.source_id)):
        root = Path(source.root)
        count = 0
        for path in sorted(root.rglob("*.json")) if root.exists() else []:
            try:
                key, _graph, digest = _graph_from_envelope(path)
                entries.setdefault(key, []).append(
                    {"source_id": source.source_id, "path": str(path), "graph_sha256": digest}
                )
                count += 1
            except CacheIntegrityError as exc:
                invalid.append({"source_id": source.source_id, "path": str(path), "error": str(exc)})
        source_rows.append(
            {
                "source_id": source.source_id,
                "root": str(root),
                "read_only": source.read_only,
                "lineage_manifest": str(source.lineage_manifest) if source.lineage_manifest else None,
                "cache_key_compatibility": list(source.cache_key_compatibility),
                "entries_valid": count,
                "exists": root.exists(),
            }
        )
    conflicts = [
        {"cache_key": key, "entries": rows}
        for key, rows in sorted(entries.items())
        if len({row["graph_sha256"] for row in rows}) > 1
    ]
    return {
        "cache_protocol": CACHE_PROTOCOL,
        "sources": source_rows,
        "unique_cache_keys": len(entries),
        "invalid_entries": invalid,
        "conflicts": conflicts,
        "valid": not invalid and not conflicts,
    }


class SharedKGGraphProvider:
    """Materialize each KGGen graph once and retain its reproducible provenance."""

    def __init__(
        self,
        extractor: Any,
        *,
        sources: Iterable[GraphCacheSource] = (),
        cache_mode: str = "read_through",
        writable_source_id: str = "current_run",
    ):
        if cache_mode not in {"cache_only", "read_through", "read_write", "live_fresh", "inspect_only"}:
            raise ValueError(f"unsupported shared graph cache mode: {cache_mode!r}")
        self.extractor = extractor
        self.sources = tuple(sorted(sources, key=lambda item: (-item.priority, item.source_id)))
        for source in self.sources:
            if not source.read_only and source.cache_key_compatibility:
                raise ValueError("legacy cache-key compatibility requires a read-only source")
            unsupported = set(source.cache_key_compatibility) - {CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY}
            if unsupported:
                raise ValueError(f"unsupported source cache-key compatibility: {sorted(unsupported)!r}")
        self.cache_mode = cache_mode
        self.writable_source_id = writable_source_id
        self._artifacts: dict[str, SharedGraphArtifact] = {}
        self._resolutions: list[dict[str, Any]] = []
        # Guards the in-memory memo + resolution log so record-level parallel
        # typing cannot race (RAGTruth responses share source docs -> same key).
        self._lock = threading.Lock()

    def inspection(self) -> dict[str, Any]:
        return inspect_cache_sources(self.sources)

    def _identity(self) -> dict[str, Any]:
        return {
            "extractor_protocol": "kggen-0.4-shared-response-v1",
            "cache_key_version": 11,
            "llm": getattr(self.extractor, "model", None),
            "runtime_fingerprint": getattr(getattr(self.extractor, "cfg", None).llm, "runtime_fingerprint", None)
            if getattr(self.extractor, "cfg", None) is not None else None,
            "max_tokens": getattr(self.extractor, "max_tokens", None),
            "chunk_chars": getattr(self.extractor, "chunk_chars", None),
            "cluster": getattr(self.extractor, "cluster", None),
        }

    def _source_key_candidates(
        self, source: GraphCacheSource, text: str, current_key: str
    ) -> Iterable[tuple[str, str]]:
        yield CACHE_KEY_SCHEMA_CURRENT, current_key
        for schema in source.cache_key_compatibility:
            legacy_key = self.extractor.cache_key_for_schema(text, schema=schema)
            if legacy_key != current_key:
                yield schema, legacy_key

    def _cached_source(
        self, text: str, current_key: str
    ) -> tuple[str, str, str, Graph, str] | None:
        """Resolve a graph by current key, then source-declared legacy keys only."""
        roots: list[tuple[GraphCacheSource, Path]] = [
            (GraphCacheSource(self.writable_source_id, Path(self.extractor.cache_dir)), Path(self.extractor.cache_dir))
        ]
        if self.cache_mode != "live_fresh":
            roots.extend((source, source.root) for source in self.sources)
        for source, root in roots:
            for schema, key in self._source_key_candidates(source, text, current_key):
                path = root / f"{key}.json"
                if not path.exists():
                    continue
                try:
                    _, graph, digest = _graph_from_envelope(path, expected_key=key)
                except CacheIntegrityError:
                    # KGExtractor itself treats invalid entries as misses.  Here it is a
                    # preflight/audit failure rather than an opportunity for silent reuse.
                    raise
                return source.source_id, key, schema, graph, digest
        return None

    def materialize(self, text: str, *, role: str) -> SharedGraphArtifact:
        # Cache resolution is fast (cache-only reads); serializing it is cheap and
        # keeps the shared memo/resolution log consistent under concurrent records.
        # The heavy typing LLM work (build_source_registry) runs outside this lock.
        with self._lock:
            return self._materialize_impl(text, role=role)

    def _materialize_impl(self, text: str, *, role: str) -> SharedGraphArtifact:
        normalized = (text or "").strip()
        input_sha = sha256_bytes(normalized.encode("utf-8"))
        key = None if not normalized else self.extractor._cache_key(normalized)
        memo_key = f"{role}:{key or input_sha}"
        if memo_key in self._artifacts:
            return self._artifacts[memo_key]

        cached = self._cached_source(normalized, key) if key else None
        was_cached = cached is not None
        if self.cache_mode == "inspect_only":
            raise RuntimeError("inspect_only cache mode does not materialize graphs")
        if cached is None and self.cache_mode == "cache_only" and normalized:
            from src.cache import CacheOnlyMissError

            raise CacheOnlyMissError(role, key, Path(self.extractor.cache_dir) / f"{key}.json")

        if cached is not None:
            # Do not delegate an external read-through hit back to KGExtractor:
            # it only knows its own configured roots, while this provider owns
            # ordered immutable sources and has already validated the envelope.
            _source_id, resolved_key, _key_schema, graph, _digest = cached
            if key is not None:
                self.extractor.usage.record_call(role, resolved_key, 0.0, cached=True)
        else:
            graph = self.extractor.extract(normalized, kind=role)
            if key:
                cached = self._cached_source(normalized, key)
        if cached is None:
            # Empty text is intentionally represented as an empty graph and has no cache key.
            graph_payload = graph.to_dict()
            graph_sha = sha256_bytes(canonical_json(graph_payload).encode("utf-8"))
            source_id, resolved_key, key_schema, cache_hit = "empty_input", None, None, True
        else:
            source_id, resolved_key, key_schema, cached_graph, graph_sha = cached
            if cached_graph.to_dict() != graph.to_dict():
                raise CacheIntegrityError(
                    f"extractor/cache disagreement for requested={key}, resolved={resolved_key}"
                )
            cache_hit = was_cached
        graph_id = sha256_bytes(canonical_json({"cache_key": resolved_key, "graph_sha256": graph_sha}).encode("utf-8"))
        artifact = SharedGraphArtifact(
            graph_id=graph_id,
            role=role,
            cache_key=resolved_key,
            cache_key_schema=key_schema,
            graph_sha256=graph_sha,
            input_sha256=input_sha,
            graph=graph,
            source_id=source_id,
            cache_hit=cache_hit,
            extraction_identity=self._identity(),
        )
        self._artifacts[memo_key] = artifact
        self._resolutions.append(artifact.record())
        return artifact

    def prepare_response(self, item: Any) -> SharedGraphArtifact:
        return self.materialize(item.response, role="response")

    def extract_reference(self, context: str, query: str | None) -> tuple[Graph, Graph]:
        # Reference (context + query) graphs pulled from the same cache-only
        # sources. Mirrors SharedKGExtractorProxy.extract_reference so a detector
        # handed the provider directly (e.g. TypedVertexDetector) can resolve
        # both the response and the reference graphs without the proxy.
        return (
            self.materialize(context, role="context").graph,
            self.materialize(query or "", role="query").graph,
        )

    def response_reference(self, item: Any) -> dict[str, str]:
        artifact = self.prepare_response(item)
        return artifact.reference()

    def artifact_records(self) -> list[dict[str, Any]]:
        return [artifact.record() for _, artifact in sorted(self._artifacts.items())]

    def resolution_records(self) -> list[dict[str, Any]]:
        return list(self._resolutions)

    def preflight(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        roles: tuple[str, ...] = ("response",),
        require_complete: bool | None = None,
    ) -> dict[str, Any]:
        """Report coverage without constructing a model backend or modifying cache."""
        rows: list[dict[str, Any]] = []
        for record in records:
            texts = {"response": record.get("response_raw", ""), "context": record.get("context_raw", ""), "query": record.get("query_raw", "")}
            for role in roles:
                normalized = str(texts[role] or "").strip()
                key = None if not normalized else self.extractor._cache_key(normalized)
                try:
                    found = self._cached_source(normalized, key) if key else (
                        "empty_input", None, None, Graph.empty(),
                        sha256_bytes(canonical_json(Graph.empty().to_dict()).encode("utf-8")),
                    )
                    rows.append({
                        "response_id": str(record.get("response_id")), "role": role,
                        "requested_cache_key": key,
                        "cache_key": found[1] if found else key,
                        "cache_key_schema": found[2] if found else CACHE_KEY_SCHEMA_CURRENT,
                        "status": "compatible_hit" if found else "miss",
                        "source_id": found[0] if found else None,
                    })
                except CacheIntegrityError as exc:
                    rows.append({"response_id": str(record.get("response_id")), "role": role, "requested_cache_key": key, "cache_key": key, "status": "corrupt", "error": str(exc)})
        report = {"cache_mode": self.cache_mode, "rows": rows, "hits": sum(row["status"] == "compatible_hit" for row in rows), "misses": sum(row["status"] == "miss" for row in rows), "valid": not any(row["status"] == "corrupt" for row in rows)}
        # A caller selecting one replayable record from a larger historical cache
        # deliberately asks for a coverage report before it chooses that record.
        # Keep the default strict for cache_only callers, but let that inspection be
        # explicitly non-fatal.  Materialization remains strictly cache_only.
        strict = self.cache_mode == "cache_only" if require_complete is None else require_complete
        if strict and (report["misses"] or not report["valid"]):
            raise CachePreflightError("cache_only preflight requires a valid graph for every expected role")
        return report


class SharedKGExtractorProxy:
    """Drop-in HalluGraph extractor proxy backed by :class:`SharedKGGraphProvider`."""

    def __init__(self, provider: SharedKGGraphProvider):
        self.provider = provider
        self.last_by_role: dict[str, SharedGraphArtifact] = {}

    def extract(self, text: str, kind: str = "graph") -> Graph:
        artifact = self.provider.materialize(text, role=kind)
        self.last_by_role[kind] = artifact
        return artifact.graph

    def extract_reference(self, context: str, query: str | None) -> tuple[Graph, Graph]:
        return self.extract(context, kind="context"), self.extract(query or "", kind="query")

    def response_reference(self, item: Any) -> dict[str, str]:
        return self.provider.response_reference(item)


__all__ = [
    "CACHE_PROTOCOL", "CacheIntegrityError", "CachePreflightError", "GraphCacheSource",
    "SharedGraphArtifact", "SharedKGExtractorProxy", "SharedKGGraphProvider", "inspect_cache_sources",
]
