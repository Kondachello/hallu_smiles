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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type


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
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.calls = 0
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._litellm_hooked = False
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def try_hook_litellm(self) -> None:
        """Attach a LiteLLM success callback to accumulate cost/tokens if available."""
        if self._litellm_hooked:
            return
        try:
            import litellm  # type: ignore

            def _cb(kwargs, completion_response, start_time, end_time):  # pragma: no cover
                try:
                    usage = getattr(completion_response, "usage", None)
                    if usage:
                        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    cost = kwargs.get("response_cost")
                    if cost is None:
                        cost = litellm.completion_cost(completion_response=completion_response)
                    self.cost += float(cost or 0.0)
                except Exception:
                    pass

            if _cb not in (litellm.success_callback or []):
                litellm.success_callback = list(litellm.success_callback or []) + [_cb]
            self._litellm_hooked = True
        except Exception:
            # LiteLLM not importable (e.g. offline/fake mode) -- fall back to call counts.
            self._litellm_hooked = False

    def record_call(self, kind: str, cache_key: str, seconds: float, cached: bool) -> None:
        with self._lock:
            self.calls += 0 if cached else 1
            if not self.path:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind, "cache_key": cache_key, "seconds": round(seconds, 4),
                    "cached": cached, "cum_calls": self.calls,
                    "cum_cost_usd": round(self.cost, 6),
                    "cum_prompt_tokens": self.prompt_tokens,
                    "cum_completion_tokens": self.completion_tokens,
                }) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "estimated_cost_usd": round(self.cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------
class KGExtractor:
    def __init__(self, cfg, backend: Any | None = None, usage: UsageLogger | None = None):
        self.cfg = cfg
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.cluster = cfg.extraction.cluster
        self.chunk_chars = cfg.extraction.context_chunk_chars
        self.max_retries = cfg.llm.max_retries
        self.backoff_base = cfg.llm.retry_backoff_base_s
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend  # if None, lazily construct KGGen on first use
        self.usage = usage or UsageLogger(None)

    # -- backend --------------------------------------------------------------
    def _get_backend(self):
        if self._backend is None:
            from kg_gen import KGGen  # lazy import; not needed for offline tests
            from .config import resolve_api_key

            kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
            api_key = resolve_api_key(self.cfg)
            if api_key:
                kwargs["api_key"] = api_key
            api_base = getattr(self.cfg.llm, "api_base", None)
            if api_base:
                kwargs["api_base"] = api_base
            self._backend = KGGen(**kwargs)
            self.usage.try_hook_litellm()
        return self._backend

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "cluster": self.cluster,
            "chunk_chars": self.chunk_chars,
            "v": 1,  # bump to invalidate cache on prompt/logic changes
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

        start = time.perf_counter()
        graph: Graph | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_base),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                graph = self._call_backend(text)
        # With reraise=True, we only reach here on success, so graph is bound.
        assert graph is not None
        elapsed = time.perf_counter() - start
        self._save_cache(key, graph)
        self.usage.record_call(kind, key, elapsed, cached=False)
        return graph

    def extract_reference(self, context: str, query: str | None) -> tuple[Graph, Graph]:
        """Extract (G_c, G_q). G_q is empty when query is empty."""
        g_c = self.extract(context, kind="context")
        g_q = self.extract(query or "", kind="query")
        return g_c, g_q


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
