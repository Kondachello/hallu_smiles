"""KGGen extraction wrapper: disk cache, retries, chunking, cost logging.

Design notes / adaptation to the real kg-gen API (documented in README):
  - The §3 snippet shows ``kg.generate(input_data=text, cluster=...)`` and a graph with
    ``.entities`` (set[str]) and ``.relations`` (set[tuple]). The installed kg-gen matches
    this and additionally exposes ``.edges`` and ``*_clusters``; we only need entities +
    relations.
  - Long-context chunking is native: ``kg.generate(chunk_size=..., cluster=True)`` chunks,
    aggregates across chunks, and runs one clustering pass -- exactly the behaviour §3 asks
    for. We therefore map ``extraction.context_chunk_chars`` -> ``chunk_size`` rather than
    merging chunks by hand.
  - The backend is injectable (``backend=``) so tests / offline smoke runs can pass a
    ``FakeExtractor`` without importing kg-gen or hitting an API.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .api_runtime import (
    CacheOnlyMissError,
    config_value,
    configure_dspy_lm,
    exception_status_code,
    install_kggen_relation_contract,
    llm_runtime_fingerprint,
    provider_options,
    strict_json_object_adapter,
)


# --------------------------------------------------------------------------------------
# Graph container (decouples the rest of the pipeline from kg-gen's object)
# --------------------------------------------------------------------------------------
@dataclass
class Graph:
    entities: set[str] = field(default_factory=set)
    relations: set[tuple[str, str, str]] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": sorted(self.entities),
            "relations": sorted([list(r) for r in self.relations]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graph":
        return cls(
            entities={str(e) for e in d.get("entities", [])},
            relations={tuple(str(x) for x in r) for r in d.get("relations", []) if len(r) == 3},
        )

    @classmethod
    def empty(cls) -> "Graph":
        return cls(set(), set())

    def union(self, other: "Graph") -> "Graph":
        return Graph(self.entities | other.entities, self.relations | other.relations)


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Cost / usage logging (best-effort; token accounting depends on LiteLLM being reachable)
# --------------------------------------------------------------------------------------
class UsageLogger:
    def __init__(
        self,
        path: str | Path | None,
        provider_calls_path: str | Path | None = None,
    ):
        self.path = Path(path) if path else None
        self.provider_calls_path = (
            Path(provider_calls_path)
            if provider_calls_path
            else (self.path.with_name("provider_calls.jsonl") if self.path else None)
        )
        self.calls = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.provider_calls = 0
        self.provider_successes = 0
        self.provider_failures = 0
        self.provider_contract_errors = 0
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.provider_calls_path:
            self.provider_calls_path.parent.mkdir(parents=True, exist_ok=True)

    def try_hook_litellm(self) -> None:
        """Compatibility no-op; calls are recorded at the exact request boundary.

        A process-global LiteLLM callback can accidentally include unrelated requests
        and receives prompt-bearing kwargs.  The API runtime instead wraps only the
        KGGen LM and relation verifier instances and writes a fixed allowlist.
        """

    def record_call(self, kind: str, cache_key: str, seconds: float, cached: bool) -> None:
        with self._lock:
            self.total_requests += 1
            self.cache_hits += int(cached)
            self.calls += 0 if cached else 1
            if not self.path:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind, "cache_key": cache_key, "seconds": round(seconds, 4),
                    "cached": cached, "cum_calls": self.calls,
                    "cum_requests": self.total_requests, "cum_cache_hits": self.cache_hits,
                    "cum_cost_usd": round(self.cost, 6),
                    "cum_prompt_tokens": self.prompt_tokens,
                    "cum_completion_tokens": self.completion_tokens,
                }) + "\n")

    def record_provider_call(
        self,
        *,
        outcome: str,
        seconds: float,
        response: Any = None,
        error: BaseException | None = None,
        retry_index: int = 0,
    ) -> None:
        """Write one redacted provider-attempt record with no prompts or secrets."""
        if outcome not in {"success", "failure", "contract_error"}:
            raise ValueError(f"unsupported provider outcome {outcome!r}")
        usage = _value(response, "usage")
        prompt_tokens = int(_value(usage, "prompt_tokens") or 0)
        completion_tokens = int(_value(usage, "completion_tokens") or 0)
        total_tokens = int(_value(usage, "total_tokens") or 0)
        request_id = _value(response, "id")
        if request_id is None and error is not None:
            request_id = getattr(error, "request_id", None)
        status = exception_status_code(error) if error is not None else None
        if status is None and response is not None:
            status = 200
        record = {
            "outcome": outcome,
            "request_id": str(request_id) if request_id is not None else None,
            "latency_s": round(float(seconds), 6),
            "http_status": status,
            "retry_index": max(0, int(retry_index)),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "error_type": type(error).__name__ if error is not None else None,
        }
        with self._lock:
            self.provider_calls += 1
            self.provider_successes += int(outcome == "success")
            self.provider_failures += int(outcome == "failure")
            self.provider_contract_errors += int(outcome == "contract_error")
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if self.provider_calls_path:
                with open(self.provider_calls_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "requests_total": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.total_requests, 6)
            if self.total_requests else 0.0,
            "estimated_cost_usd": round(self.cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider_calls": self.provider_calls,
            "provider_successes": self.provider_successes,
            "provider_failures": self.provider_failures,
            "provider_contract_errors": self.provider_contract_errors,
        }


def _value(value: Any, name: str) -> Any:
    if value is None:
        return None
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


# --------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------
class KGExtractor:
    def __init__(
        self,
        cfg,
        backend: Any | None = None,
        usage: UsageLogger | None = None,
        *,
        cache_only: bool = False,
    ):
        self.cfg = cfg
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        # KGGen 0.4 defaults to 16k output tokens for GPT-5-family models.
        # Keep that upstream default for the main config unless a caller
        # explicitly supplies a narrower cap (the micro demo does).
        self.max_tokens = (
            cfg.llm.get("max_tokens", None)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "max_tokens", None)
        )
        self.cluster = cfg.extraction.cluster
        self.chunk_chars = cfg.extraction.context_chunk_chars
        self.runtime_options = (
            provider_options(cfg.llm)
            if backend is None
            else {
                "response_format": {"type": "json_object"},
                "extra_body": {"enable_thinking": False},
                "timeout": float(config_value(cfg.llm, "request_timeout_s", 180)),
            }
        )
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_only = bool(cache_only)
        if not self.cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend  # if None, lazily construct KGGen on first use
        self._dspy_adapter = None
        self.backend_fingerprint = (
            "kggen-0.4"
            if backend is None
            else f"{type(backend).__module__}.{type(backend).__qualname__}"
        )
        self.usage = usage or UsageLogger(None)

    # -- backend --------------------------------------------------------------
    def _get_backend(self):
        if self.cache_only:
            raise RuntimeError("cache-only mode forbids constructing a KGGen backend")
        if self._backend is None:
            from kg_gen import KGGen  # lazy import; not needed for offline tests
            from .config import resolve_api_key

            kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            api_key = resolve_api_key(self.cfg)
            if api_key:
                kwargs["api_key"] = api_key
            api_base = getattr(self.cfg.llm, "api_base", None)
            if api_base:
                kwargs["api_base"] = api_base
            self._backend = KGGen(**kwargs)
            install_kggen_relation_contract()
            lm = getattr(self._backend, "lm", None)
            if lm is None:
                raise RuntimeError("KGGen backend did not expose its DSPy LM")
            self._dspy_adapter = configure_dspy_lm(lm, self.cfg, self.usage)
        return self._backend

    @contextmanager
    def _dspy_context(self, backend: Any) -> Iterator[None]:
        """Keep the strict adapter active across extraction and official clustering."""
        lm = getattr(backend, "lm", None)
        if lm is None:
            # Injected offline backends do not use DSPy.
            yield
            return
        import dspy

        adapter = self._dspy_adapter or strict_json_object_adapter(
            extra_body=self.runtime_options["extra_body"], usage=self.usage
        )
        with dspy.context(lm=lm, adapter=adapter):
            yield

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        params = {
            "v": 3,
            "extractor_protocol": "kggen-0.4-dashscope-strict-v1",
            "backend": self.backend_fingerprint,
            "llm": llm_runtime_fingerprint(self.cfg),
            "max_tokens": self.max_tokens,
            "cluster": self.cluster,
            "chunk_chars": self.chunk_chars,
        }
        payload = json.dumps(params, sort_keys=True) + "\x00" + text
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Graph | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return Graph.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, graph: Graph) -> None:
        dest = self._cache_path(key)
        # Unique temp name per writer: concurrent threads extracting identical text share a
        # cache key, so a fixed tmp name would race on replace() (FileNotFoundError). A unique
        # tmp + atomic os.replace makes concurrent writes of the same key safe and idempotent.
        import os

        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(graph.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)  # atomic -> crash-safe / resumable

    # -- generation -----------------------------------------------------------
    def _call_backend(self, text: str) -> Graph:
        backend = self._get_backend()
        gen_kwargs: dict[str, Any] = {"input_data": text, "cluster": self.cluster}
        if len(text) > self.chunk_chars:
            gen_kwargs["chunk_size"] = self.chunk_chars
        with self._dspy_context(backend):
            # Keep KGGen's native generate -> official LLM cluster path and its
            # empty clustering context exactly as in the scientific baseline.
            g = backend.generate(**gen_kwargs)
        entities = {str(e) for e in getattr(g, "entities", set())}
        relations = {
            tuple(str(x) for x in r)
            for r in getattr(g, "relations", set())
            if isinstance(r, (list, tuple)) and len(r) == 3
        }
        return Graph(entities, relations)

    def extract(self, text: str, kind: str = "graph") -> Graph:
        """Extract a graph from a single text, using the disk cache.

        Returns an empty graph (no API call) for empty/whitespace text -- this is how
        empty queries (Data2txt / Summary) yield G_q = empty.
        """
        text = (text or "").strip()
        if not text:
            return Graph.empty()

        key = self._cache_key(text)
        cached = self._load_cache(key)
        if cached is not None:
            self.usage.record_call(kind, key, 0.0, cached=True)
            return cached
        if self.cache_only:
            raise CacheOnlyMissError(f"KG cache miss for {kind} key {key}")

        start = time.perf_counter()
        # Each actual DSPy provider call owns its transient-only retry loop. A
        # late 503 therefore retries only that request, never the whole graph.
        graph = self._call_backend(text)
        elapsed = time.perf_counter() - start
        self._save_cache(key, graph)
        self.usage.record_call(kind, key, elapsed, cached=False)
        return graph

    def extract_reference(self, context: str, query: str | None) -> tuple[Graph, Graph]:
        """Extract (G_c, G_q). G_q is empty when query is empty."""
        g_c = self.extract(context, kind="context")
        g_q = self.extract(query or "", kind="query")
        return g_c, g_q

    def relation_contract(
        self, text: str, entities: Iterable[str]
    ) -> set[tuple[str, str, str]]:
        """Run KGGen's real relation signature through the exact live API path.

        This deliberately bypasses the graph cache so repeated contract probes
        exercise independent provider responses.
        """
        if self.cache_only:
            raise RuntimeError("cache-only mode forbids a live relation contract probe")
        backend = self._get_backend()
        install_kggen_relation_contract()
        with self._dspy_context(backend):
            from .api_runtime import strict_kggen_get_relations

            relations = strict_kggen_get_relations(text, list(entities))
        return {tuple(str(x) for x in relation) for relation in relations}


# --------------------------------------------------------------------------------------
# Offline fake backend (for tests and `run.py --fake-extractor` plumbing checks)
# --------------------------------------------------------------------------------------
class FakeKGGen:
    """A deterministic, dependency-free stand-in for kg_gen.KGGen.

    Extracts a toy graph from text so the full pipeline runs offline. It is NOT a real
    extractor -- only for exercising plumbing / cache determinism, never for real metrics.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _tokens(text: str) -> list[str]:
        import re

        words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
        # keep capitalized-ish / longer tokens as pseudo entities
        return [w for w in words if len(w) >= 4]

    def generate(self, input_data, cluster=True, chunk_size=None, context=None):  # noqa: ARG002
        toks = self._tokens(input_data if isinstance(input_data, str) else str(input_data))
        uniq = list(dict.fromkeys(toks))[:12]
        rels = set()
        for a, b in zip(uniq, uniq[1:]):
            rels.add((a, "co_occurs_with", b))

        class _G:
            entities = set(uniq)
            relations = rels

        return _G()
