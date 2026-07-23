"""Standalone no-gold bridge: text/RAGTruth -> KGGen graphs -> dynamic typing."""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Iterable

from .agent import DynamicTypingAgent, graph_from_fixture
from .models import AnswerInput, SourceInput
from .persistence import ArtifactWriter


def _hallu_root() -> Path:
    return Path(__file__).resolve().parents[4] / "hallu_smiles"


def _kggen_modules() -> tuple[Any, Any, Any, Any]:
    root = _hallu_root()
    if not root.is_dir():
        raise RuntimeError(f"HalluGraph KGGen source is unavailable: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package_root = Path(__file__).resolve().parents[2]
    # KGGen needs only the PyTorch path. The typing environment also contains HHEM's
    # TensorFlow stack, whose Keras-3 import is incompatible with this Transformers build.
    # Keep all KGGen auxiliary state local and make its NLTK resources explicit.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("DSPY_CACHEDIR", str(package_root / ".cache" / "dspy"))
    os.environ.setdefault("NLTK_DATA", str(package_root / "local_resources" / "nltk_data"))
    from src.config import load_config
    from src.extract import FakeKGGen, KGExtractor, Graph
    return load_config, FakeKGGen, KGExtractor, Graph


def graph_dict(graph: Any) -> dict[str, Any]:
    return {"entities": sorted(str(x) for x in graph.entities), "relations": sorted([list(map(str, edge)) for edge in graph.relations])}


def _unicode_fake_graph(text: str, graph_type: Any) -> Any:
    """Create a deterministic non-empty graph for non-Latin offline smoke inputs.

    HalluGraph's bundled FakeKGGen intentionally tokenizes only ASCII words. The
    standalone package uses this fallback only when that toy backend produces no
    entity for non-empty text; it is never presented as KGGen output or a semantic
    extraction result.
    """
    tokens = re.findall(r"[^\W\d_][\w-]{2,}", text.casefold(), flags=re.UNICODE)
    entities = list(dict.fromkeys(token for token in tokens if len(token) >= 4))[:12]
    relations = {
        (left, "co_occurs_with", right)
        for left, right in zip(entities, entities[1:])
    }
    return graph_type(entities=set(entities), relations=relations)


def _install_strict_cluster_adapter(extractor: Any) -> None:
    """Keep KGGen's nested clustering call on the gateway's JSON-Schema protocol.

    KGGen 0.4 enters a new ``dspy.context(lm=...)`` while clustering and thereby
    drops the outer adapter supplied by ``KGExtractor``.  DSPy's stock fallback
    then sends ``response_format.type=json_object``; the live gateway accepts
    only ``json_schema``.  Patch only this local backend instance, retaining
    KGGen's own ``cluster_graph`` algorithm and prompts.
    """
    backend = extractor._get_backend()
    if getattr(backend, "_hallugraph_strict_cluster_adapter", False):
        return
    import dspy
    from kg_gen.steps._3_cluster_graph import cluster_graph
    from src.dspy_adapter import is_retryable_llm_exception, strict_json_schema_adapter
    from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential, wait_random

    adapter = strict_json_schema_adapter()

    class RetriedProgram:
        """Proxy a DSPy program, including KGGen's internally caught validations."""

        def __init__(self, program: Any, label: str):
            self._program = program
            self._label = label

        def __getattr__(self, name: str) -> Any:
            return getattr(self._program, name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            def before_sleep(state: Any) -> None:
                exc = state.outcome.exception() if state.outcome is not None else None
                sleep_s = getattr(state.next_action, "sleep", None)
                print(
                    "[kg] retry:cluster-llm "
                    f"program={self._label} attempt={state.attempt_number} "
                    f"next_sleep_s={float(sleep_s or 0):.2f} "
                    f"error={type(exc).__name__ if exc else 'unknown'}",
                    flush=True,
                )

            wait = wait_exponential(
                multiplier=float(extractor.backoff_base), max=float(extractor.backoff_max)
            )
            if extractor.backoff_jitter:
                wait = wait + wait_random(min=0, max=float(extractor.backoff_jitter))
            for attempt in Retrying(
                stop=stop_after_attempt(int(extractor.max_retries)),
                wait=wait,
                retry=retry_if_exception(is_retryable_llm_exception),
                before_sleep=before_sleep,
                reraise=True,
            ):
                with attempt:
                    return self._program(*args, **kwargs)
            raise RuntimeError("unreachable tenacity retry state")

    def cluster_with_strict_schema(
        bound_self: Any,
        graph: Any,
        context: str = "",
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> Any:
        if any([model, temperature, api_key, api_base]):
            bound_self.init_model(
                model=model or bound_self.model,
                temperature=temperature or bound_self.temperature,
                api_key=api_key or bound_self.api_key,
                api_base=api_base or bound_self.api_base,
            )
        # ``cluster_graph`` catches validation exceptions itself.  Wrap every
        # program it creates so temporary 429/5xx responses are retried *before*
        # that catch can silently turn a failed validation into a new cluster.
        original_predict = dspy.Predict
        original_chain_of_thought = dspy.ChainOfThought

        def retried_factory(factory: Any, label: str) -> Any:
            def construct(*args: Any, **kwargs: Any) -> RetriedProgram:
                return RetriedProgram(factory(*args, **kwargs), label)
            return construct

        dspy.Predict = retried_factory(original_predict, "predict")
        dspy.ChainOfThought = retried_factory(original_chain_of_thought, "chain_of_thought")
        try:
            with dspy.context(lm=bound_self.lm, adapter=adapter):
                return cluster_graph(graph, context)
        finally:
            dspy.Predict = original_predict
            dspy.ChainOfThought = original_chain_of_thought

    backend.cluster = types.MethodType(cluster_with_strict_schema, backend)
    backend._hallugraph_strict_cluster_adapter = True


def _set_config_value(config: Any, name: str, value: Any) -> None:
    """Update both views exposed by HalluGraph's attribute-backed Config.

    ``src.config.Config.get`` reads its original ``_data`` mapping, while
    ordinary consumers read attributes.  A runtime-only override therefore
    has to update both views or different parts of KGExtractor observe
    different configurations.
    """
    setattr(config, name, value)
    data = getattr(config, "_data", None)
    if isinstance(data, dict):
        data[name] = value


def make_extractor(*, kggen_config: str | None, fake: bool, cache_root: str | None = None) -> tuple[Any, str]:
    load_config, FakeKGGen, KGExtractor, Graph = _kggen_modules()
    if fake:
        backend = FakeKGGen()

        class FakeExtractor:
            def extract(self, text: str, kind: str) -> Any:
                generated = backend.generate(text, cluster=True)
                if generated.entities or not text.strip():
                    return Graph(
                        entities=set(generated.entities),
                        relations=set(generated.relations),
                    )
                return _unicode_fake_graph(text, Graph)
        return FakeExtractor(), "fake-kggen-v1"
    if not kggen_config:
        raise ValueError("real KGGen requires --kggen-config; use --fake-kggen only for offline plumbing")
    cfg = load_config(kggen_config)
    # The standalone typing runtime already reads this authenticated gateway from the
    # environment. Reuse it only when the HalluGraph profile deliberately leaves its
    # job-local URL blank; neither URL nor key is written to an artifact or YAML file.
    if not getattr(cfg.llm, "api_base", None):
        gateway = os.environ.get("HALLU_GATEWAY_URL", "").rstrip("/")
        if not gateway:
            raise ValueError("KGGen config has no llm.api_base and HALLU_GATEWAY_URL is unset")
        _set_config_value(
            cfg.llm,
            "api_base",
            gateway if gateway.endswith("/v1") else f"{gateway}/v1",
        )
    if os.environ.get("HALLU_TYPING_MODEL"):
        _set_config_value(cfg.llm, "model", os.environ["HALLU_TYPING_MODEL"])
    # The standalone gateway accepts structured responses only through the
    # OpenAI-compatible ``json_schema`` transport.  KGGen/DSPy's default JSON
    # adapter falls back to ``json_object`` after a parse failure, which this
    # gateway rejects with HTTP 400.  Configure HalluGraph's strict adapter
    # locally (without mutating its shared YAML) so every KGGen call carries a
    # complete schema.  The two provenance fields are deliberately overridable
    # from the environment for a pinned deployed gateway; the local markers
    # identify an unpinned developer-machine run in the cache contract.
    _set_config_value(cfg.llm, "structured_output_transport", "response_format")
    _set_config_value(cfg.llm, "structured_output_backend", "vertex")
    _set_config_value(cfg.llm, "structured_output_request_backend", None)
    _set_config_value(
        cfg.llm,
        "model_revision",
        os.environ.get("HALLU_KGGEN_MODEL_REVISION", f"standalone:{cfg.llm.model}"),
    )
    _set_config_value(
        cfg.llm,
        "runtime_fingerprint",
        os.environ.get(
            "HALLU_KGGEN_RUNTIME_FINGERPRINT",
            "standalone-live-gateway-unpinned",
        ),
    )
    if cache_root:
        _set_config_value(cfg, "cache_dir", str(cache_root))
    extractor = KGExtractor(cfg)
    _install_strict_cluster_adapter(extractor)
    return extractor, "hallu-smiles-kggen-v14-synchronized-strict-schema"


def text_records(path: str | Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(records, 1):
        if not str(row.get("context", "")).strip():
            raise ValueError(f"line {index}: context is required")
        forbidden = {"gold", "gold_label", "gold_labels", "labels", "hallucination_labels"}
        if forbidden & set(row):
            raise ValueError(f"line {index}: no-gold input contains forbidden fields")
        row.setdefault("case_id", f"text-{index:04d}")
        row.setdefault("source_id", row["case_id"])
    return records


def ragtruth_records(source_info: str | Path, response: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Materialize only context/query/response and identifiers; never load labels."""
    root = _hallu_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.data import build_context_query
    sources = {str(row["source_id"]): row for row in _jsonl(source_info)}
    rows: list[dict[str, Any]] = []
    for item in _jsonl(response):
        source = sources.get(str(item.get("source_id")))
        if source is None:
            continue
        context, query = build_context_query(source)
        rows.append({"case_id": f"ragtruth-{item['id']}", "source_id": str(item["source_id"]), "response_id": str(item["id"]), "context": context, "query": query or "", "response": str(item.get("response", "")), "ragtruth": {"task_type": source.get("task_type"), "split": item.get("split"), "model": item.get("model")}})
        if limit and len(rows) >= limit:
            break
    return rows


def _jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def run_pipeline(*, agent: DynamicTypingAgent, records: Iterable[dict[str, Any]], output: str | Path, kggen_config: str | None, fake_kggen: bool, kggen_cache_root: str | None = None) -> list[dict[str, Any]]:
    extractor, protocol = make_extractor(kggen_config=kggen_config, fake=fake_kggen, cache_root=kggen_cache_root)
    root = Path(output); summary: list[dict[str, Any]] = []
    for row in records:
        case = str(row["case_id"])
        context = extractor.extract(str(row["context"]), kind="context")
        query = extractor.extract(str(row.get("query", "")), kind="query")
        graphs = {"context": graph_dict(context), "query": graph_dict(query)}
        source = SourceInput(source_id=str(row["source_id"]), context_raw=str(row["context"]), query_raw=str(row.get("query", "")), context_graph=graph_from_fixture(graph_id=f"{case}:context", role="context", payload=graphs["context"]), query_graph=graph_from_fixture(graph_id=f"{case}:query", role="query", payload=graphs["query"]))
        source_run = agent.build_source_registry(source)
        case_dir = root / case
        if source_run.registry is None:
            ArtifactWriter(case_dir).write_json("pipeline_input.json", {"case_id": case, "source": source.model_dump(mode="json"), "kggen": {"protocol": protocol, "graphs": graphs}, "failure": source_run.failure})
            summary.append({"case_id": case, "status": "failed", "failure": source_run.failure}); continue
        answer_run = None
        answer_text = str(row.get("response", "")).strip()
        if answer_text:
            answer_graph = extractor.extract(answer_text, kind="answer")
            answer = AnswerInput(source_id=source.source_id, response_id=str(row.get("response_id", case)), response_raw=answer_text, answer_graph=graph_from_fixture(graph_id=f"{case}:answer", role="answer", payload=graph_dict(answer_graph)), registry=source_run.registry)
            answer_run = agent.annotate_answer(answer)
        path = agent.write_run_artifacts(run_id=case, source_run=source_run, answer_run=answer_run)
        ArtifactWriter(path).write_json("pipeline_input.json", {"schema_version": "kggen-typing-pipeline-v1", "case_id": case, "source": source.model_dump(mode="json"), "answer": answer.model_dump(mode="json", exclude={"registry"}) if answer_text else None, "kggen": {"protocol": protocol, "graphs": {**graphs, **({"answer": graph_dict(answer_graph)} if answer_text else {})}}, "ragtruth": row.get("ragtruth")})
        summary.append({"case_id": case, "status": answer_run.status.value if answer_run else source_run.status.value, "artifact_dir": str(path)})
    ArtifactWriter(root).write_json("pipeline_summary.json", {"schema_version": "kggen-typing-run-v1", "kggen_protocol": protocol, "cases": summary})
    return summary
