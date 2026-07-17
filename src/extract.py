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
        self.total_requests = 0
        self.cache_hits = 0
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
                    # A localhost vLLM deployment has no meaningful USD price.
                    # LiteLLM 1.60's generic cost calculator serialises its
                    # OpenAI-compatible response through Pydantic. Some guided
                    # JSON responses from a local 8B model can spin there after
                    # vLLM has already completed, leaving the paid GPU idle.
                    # Preserve an explicitly supplied provider cost, but never
                    # invoke the calculator merely for telemetry.
                    cost = kwargs.get("response_cost")
                    if cost is not None:
                        self.cost += float(cost)
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
        }


# --------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------
class KGExtractor:
    def __init__(self, cfg, backend: Any | None = None, usage: UsageLogger | None = None):
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
        raw_cluster_max_items = (
            cfg.extraction.get("cluster_max_items", None)
            if hasattr(cfg.extraction, "get")
            else getattr(cfg.extraction, "cluster_max_items", None)
        )
        self.cluster_max_items = (
            int(raw_cluster_max_items) if raw_cluster_max_items is not None else None
        )
        if self.cluster_max_items is not None and self.cluster_max_items <= 0:
            raise ValueError("extraction.cluster_max_items must be positive or null")
        self.chunk_chars = cfg.extraction.context_chunk_chars
        self.serial_chunking = (
            cfg.extraction.get("serial_chunking", False)
            if hasattr(cfg.extraction, "get")
            else getattr(cfg.extraction, "serial_chunking", False)
        )
        self.max_retries = cfg.llm.max_retries
        self.backoff_base = cfg.llm.retry_backoff_base_s
        self.request_timeout_s = float(
            cfg.llm.get("request_timeout_s", 90)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "request_timeout_s", 90)
        )
        if self.request_timeout_s <= 0:
            raise ValueError("llm.request_timeout_s must be positive")
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
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            api_key = resolve_api_key(self.cfg)
            if api_key:
                kwargs["api_key"] = api_key
            api_base = getattr(self.cfg.llm, "api_base", None)
            if api_base:
                kwargs["api_base"] = api_base
            self._backend = KGGen(**kwargs)
            # KGGen 0.4 does not expose DSPy's HTTP timeout in its constructor.
            # Bound the underlying local-vLLM request anyway: a request that is
            # never accepted by the server must surface as a retryable error,
            # not occupy a paid GPU indefinitely.  KGExtractor's tenacity loop
            # remains the single retry policy, hence DSPy's own retries are off.
            lm = getattr(self._backend, "lm", None)
            if lm is not None:
                lm.kwargs["timeout"] = self.request_timeout_s
                lm.num_retries = 0
            self.usage.try_hook_litellm()
        return self._backend

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cluster": self.cluster,
            "chunk_chars": self.chunk_chars,
            "serial_chunking": self.serial_chunking,
            "v": 3,  # serial chunk scheduling changes the extraction contract
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
    @staticmethod
    def _split_text(text: str, chunk_size: int) -> list[str]:
        """Use KGGen's own sentence-aware splitter without its thread pool."""
        from kg_gen.utils.chunk_text import chunk_text

        return chunk_text(text, chunk_size)

    def _should_cluster_backend_graph(self, graph: Any) -> bool:
        """Return whether optional KGGen canonicalisation is safe to run.

        KGGen creates dynamic ``Literal[...]`` schemas for all entities and
        predicates before asking the model to cluster them.  That is useful
        canonicalisation, but a small local model can occasionally emit a very
        large candidate list.  Building that schema becomes CPU-bound before a
        new vLLM request is sent.  A finite limit retains every raw triple and
        simply skips this optional post-processing for that outlier.  ``None``
        is deliberately the default so non-DataSphere behaviour is unchanged.
        """
        if not self.cluster:
            return False
        if self.cluster_max_items is None:
            return True
        entities = getattr(graph, "entities", set()) or set()
        edges = getattr(graph, "edges", set()) or {
            relation[1] for relation in (getattr(graph, "relations", set()) or set())
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        largest_group = max(len(entities), len(edges))
        if largest_group <= self.cluster_max_items:
            return True
        print(
            "[kg] skipping optional KGGen clustering: "
            f"entities={len(entities)} predicates={len(edges)} "
            f"limit={self.cluster_max_items}; raw triples are retained",
            flush=True,
        )
        return False

    def _call_backend(self, text: str) -> Graph:
        backend = self._get_backend()
        if self.serial_chunking and len(text) > self.chunk_chars:
            # KGGen 0.4 implements ``chunk_size`` with an unbounded nested
            # ThreadPoolExecutor over one shared DSPy LM.  Combining that with
            # response-level parallelism can deadlock a local vLLM client after
            # some successful calls.  Preserve KGGen's algorithm (chunk,
            # aggregate, then cluster) but schedule the chunks serially.
            graphs = [
                backend.generate(input_data=chunk, cluster=False)
                for chunk in self._split_text(text, self.chunk_chars)
            ]
            g = backend.aggregate(graphs)
            if self._should_cluster_backend_graph(g):
                g = backend.cluster(g)
        else:
            # With a finite local-runtime bound, generate raw triples first so
            # the decision to call KGGen's optional clustering is made from the
            # actual candidate cardinality.  With the default ``None`` this is
            # the original one-call KGGen behaviour.
            bounded_clustering = self.cluster and self.cluster_max_items is not None
            gen_kwargs: dict[str, Any] = {
                "input_data": text,
                "cluster": False if bounded_clustering else self.cluster,
            }
            if len(text) > self.chunk_chars:
                gen_kwargs["chunk_size"] = self.chunk_chars
            g = backend.generate(**gen_kwargs)
            if bounded_clustering and self._should_cluster_backend_graph(g):
                g = backend.cluster(g)
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
