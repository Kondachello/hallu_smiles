"""Standalone LangGraph agent facade with source-only and answer-only entry points."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import yaml
from jsonschema import Draft202012Validator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .errors import DynamicTypingError, InputContractError
from .heuristics import make_spans, registry_checksum
from .models import (
    AnswerAnnotation,
    AnswerInput,
    AnswerRun,
    BackendKind,
    EvidenceSpan,
    EvidenceLevel,
    FrozenRegistry,
    GraphInput,
    NliResult,
    RunStatus,
    SourceInput,
    SourceRun,
    TypeAssignment,
    canonical_json,
    normalize,
    stable_id,
)
from .persistence import ArtifactWriter, JsonFileCache
from .prompt_registry import PromptRegistry
from .quality_workflow import ALGORITHM_VERSION, QualityTypingWorkflow
from .transports import DeterministicNli, FakeStructuredModel, HhemNli, LiteLLMStructuredModel


class SourceState(TypedDict, total=False):
    source: dict[str, Any]
    spans: list[dict[str, Any]]
    types: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    overview: dict[str, Any]
    source_nli_results: list[dict[str, Any]]
    registry: dict[str, Any]
    artifacts: list[dict[str, Any]]
    failure: str


class AnswerState(TypedDict, total=False):
    answer: dict[str, Any]
    annotations: list[dict[str, Any]]
    nli_results: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    failure: str


class DynamicTypingAgent:
    """Core agent. No method accepts gold labels or mutates graph payloads."""

    def __init__(
        self,
        *,
        prompt_root: str | Path | None = None,
        cache_root: str | Path = ".cache/dynamic-typing-agent",
        artifacts_root: str | Path = "runs",
        backend: BackendKind | str = BackendKind.FAKE,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        invoke_model_nodes: bool = False,
        nli_backend: str = "fake",
        hhem_model_path: str | Path | None = None,
        hhem_revision: str | None = None,
        hhem_entailment_threshold: float = 0.80,
        hhem_contradiction_threshold: float = 0.20,
        model_timeout_seconds: float = 90.0,
        model_max_attempts: int = 5,
        retry_backoff_base_seconds: float = 2.0,
        retry_backoff_max_seconds: float = 60.0,
        retry_jitter_seconds: float = 1.0,
        model_temperature: float = 0.0,
        structured_schema_profile: str = "native",
        max_entity_attempts: int = 2,
    ):
        self.prompts = PromptRegistry(prompt_root)
        self.cache = JsonFileCache(cache_root)
        self.artifacts_root = Path(artifacts_root)
        self.backend = BackendKind(backend)
        self.invoke_model_nodes = invoke_model_nodes or self.backend is BackendKind.LIVE
        if self.backend is BackendKind.LIVE:
            if not model_name or not api_base or not api_key:
                raise InputContractError("live backend requires model_name, api_base and an environment-supplied API key")
            self.model = LiteLLMStructuredModel(
                model_name,
                api_base,
                api_key,
                timeout_seconds=model_timeout_seconds,
                max_attempts=model_max_attempts,
                temperature=model_temperature,
                retry_backoff_base_seconds=retry_backoff_base_seconds,
                retry_backoff_max_seconds=retry_backoff_max_seconds,
                retry_jitter_seconds=retry_jitter_seconds,
                structured_schema_profile=structured_schema_profile,
            )
        else:
            self.model = FakeStructuredModel()
        if nli_backend == "fake":
            self.nli = DeterministicNli()
        elif nli_backend == "hhem":
            if not hhem_model_path or not hhem_revision:
                raise InputContractError("HHEM NLI requires an explicit local model path and pinned revision")
            self.nli = HhemNli(
                model_path=hhem_model_path,
                revision=hhem_revision,
                entailment_threshold=hhem_entailment_threshold,
                contradiction_threshold=hhem_contradiction_threshold,
            )
        else:
            raise InputContractError(f"unsupported nli.backend: {nli_backend}")
        self.nli_backend = nli_backend
        self.max_entity_attempts = max(1, max_entity_attempts)
        self.checkpointer = MemorySaver()
        self.source_graph = self._build_source_graph()
        self.answer_graph = self._build_answer_graph()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        cache_root: str | Path | None = None,
        artifacts_root: str | Path | None = None,
    ) -> "DynamicTypingAgent":
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        model = config.get("model", {})
        nli = config.get("nli", {})
        persistence = config.get("persistence", {})
        source_config = config.get("source", {})
        backend = model.get("backend", "fake")
        gateway_url = model.get("api_base") or cls._required_environment(model.get("api_base_env"), "model.api_base")
        api_base = cls._openai_api_base(gateway_url) if backend == "live" else gateway_url
        api_key = cls._required_environment(model.get("api_key_env"), "model API key") if backend == "live" else None
        model_name = model.get("model") or cls._required_environment(model.get("model_env"), "model.model")
        hhem_path = nli.get("model_path") or cls._required_environment(nli.get("model_path_env"), "nli.model_path")
        return cls(
            prompt_root=config.get("prompt_set", {}).get("root"),
            cache_root=cache_root or persistence.get("cache_root", ".cache/dynamic-typing-agent"),
            artifacts_root=artifacts_root or persistence.get("artifacts_root", "runs"),
            backend=backend,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            invoke_model_nodes=bool(model.get("invoke_model_nodes", backend == "live")),
            nli_backend=nli.get("backend", "fake"),
            hhem_model_path=hhem_path,
            hhem_revision=nli.get("revision"),
            hhem_entailment_threshold=float(nli.get("entailment_threshold", 0.80)),
            hhem_contradiction_threshold=float(nli.get("contradiction_threshold", 0.20)),
            model_timeout_seconds=float(model.get("timeout_seconds", 90)),
            model_max_attempts=int(model.get("max_attempts", 5)),
            retry_backoff_base_seconds=float(model.get("retry_backoff_base_seconds", 2)),
            retry_backoff_max_seconds=float(model.get("retry_backoff_max_seconds", 60)),
            retry_jitter_seconds=float(model.get("retry_jitter_seconds", 1)),
            model_temperature=float(model.get("temperature", 0)),
            structured_schema_profile=str(model.get("structured_schema_profile", "native")),
            max_entity_attempts=int(source_config.get("max_entity_attempts", source_config.get("max_repair_attempts", 2))),
        )

    @staticmethod
    def _required_environment(name: str | None, label: str) -> str | None:
        if not name:
            return None
        value = os.environ.get(name, "").strip()
        if not value:
            raise InputContractError(f"{label} requires non-empty environment variable {name}")
        return value

    @staticmethod
    def _openai_api_base(gateway_url: str | None) -> str | None:
        """Convert the project Cloud Run origin to its OpenAI-compatible `/v1` base."""
        if gateway_url is None:
            return None
        base = gateway_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"

    def _quality_workflow(self) -> QualityTypingWorkflow:
        """Build from current injected transports so tests and callers can replace them."""
        return QualityTypingWorkflow(
            prompts=self.prompts,
            model=self.model,
            invoke_model_nodes=self.invoke_model_nodes,
            verify_nli=self._verify_nli,
            max_entity_attempts=self.max_entity_attempts,
        )

    def _build_source_graph(self):
        graph = StateGraph(SourceState)
        graph.add_node("validate_source", self._validate_source)
        graph.add_node("source_cache", self._source_cache)
        graph.add_node("segment_source", self._segment_source)
        graph.add_node("schema_overview", self._schema_overview)
        graph.add_node("derive_registry", self._derive_registry)
        graph.add_node("freeze_registry", self._freeze_registry)
        graph.add_edge(START, "validate_source")
        graph.add_edge("validate_source", "source_cache")
        graph.add_conditional_edges("source_cache", self._route_source_cache, {"cached": "freeze_registry", "fresh": "segment_source"})
        graph.add_edge("segment_source", "schema_overview")
        graph.add_edge("schema_overview", "derive_registry")
        graph.add_edge("derive_registry", "freeze_registry")
        graph.add_edge("freeze_registry", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _build_answer_graph(self):
        graph = StateGraph(AnswerState)
        graph.add_node("validate_answer", self._validate_answer)
        graph.add_node("annotate_answer", self._annotate_answer)
        graph.add_node("nli_answer", self._nli_answer)
        graph.add_node("emit_answer", self._emit_answer)
        graph.add_edge(START, "validate_answer")
        graph.add_edge("validate_answer", "annotate_answer")
        graph.add_edge("annotate_answer", "nli_answer")
        graph.add_edge("nli_answer", "emit_answer")
        graph.add_edge("emit_answer", END)
        return graph.compile(checkpointer=self.checkpointer)

    def build_source_registry(self, source: SourceInput) -> SourceRun:
        try:
            state = self.source_graph.invoke(
                {"source": source.model_dump(mode="json"), "artifacts": []},
                config={"configurable": {"thread_id": f"source:{source.source_id}:{uuid.uuid4().hex}"}},
            )
            registry = FrozenRegistry.model_validate(state["registry"])
            return SourceRun(status=RunStatus.OK, registry=registry, artifacts=tuple(state.get("artifacts", [])))
        except Exception as exc:
            return SourceRun(status=RunStatus.FAILED, failure=repr(exc))

    def annotate_answer(self, answer: AnswerInput) -> AnswerRun:
        try:
            state = self.answer_graph.invoke(
                {"answer": answer.model_dump(mode="json"), "artifacts": []},
                config={"configurable": {"thread_id": f"answer:{answer.response_id}:{uuid.uuid4().hex}"}},
            )
            annotation = AnswerAnnotation(
                answer_assignments=tuple(TypeAssignment.model_validate(item) for item in state["annotations"]),
                nli_results=tuple(NliResult.model_validate(item) for item in state["nli_results"]),
            )
            return AnswerRun(status=RunStatus.OK, annotations=annotation, artifacts=tuple(state.get("artifacts", [])))
        except Exception as exc:
            return AnswerRun(status=RunStatus.FAILED, failure=repr(exc))

    def _validate_source(self, state: SourceState) -> dict[str, Any]:
        source = SourceInput.model_validate(state["source"])
        forbidden = {"gold", "gold_label", "gold_labels", "hallucination_labels"}
        if forbidden.intersection(source.model_dump()):
            raise InputContractError("source input contains a forbidden gold field")
        return self._event(state, "validate_source", {"source": source.model_dump(mode="json")}, {"valid": True})

    def _source_cache_key(self, source: SourceInput) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "source_id": source.source_id,
                    "context_graph_id": source.context_graph.graph_id,
                    "query_graph_id": source.query_graph.graph_id,
                    "context_graph": source.context_graph.model_dump(mode="json"),
                    "query_graph": source.query_graph.model_dump(mode="json"),
                    "context": source.context_raw,
                    "query": source.query_raw,
                    "prompt_manifest": self.prompts.manifest_sha256,
                    "algorithm_version": ALGORITHM_VERSION,
                }
            ).encode("utf-8")
        ).hexdigest()

    def _source_cache(self, state: SourceState) -> dict[str, Any]:
        source = SourceInput.model_validate(state["source"])
        key = self._source_cache_key(source)
        cached = self.cache.get("frozen_registry", key)
        valid_cached: dict[str, Any] = {}
        invalid_reason: str | None = None
        if cached:
            try:
                valid_cached = FrozenRegistry.model_validate(cached).model_dump(mode="json")
            except Exception as exc:
                invalid_reason = repr(exc)
        return self._event(
            state,
            "source_cache",
            {"cache_key": key, "algorithm_version": ALGORITHM_VERSION},
            {
                "hit": bool(valid_cached),
                "route": "cached" if valid_cached else "fresh",
                "invalid_cached_registry": invalid_reason,
            },
            registry=valid_cached,
        )

    def _route_source_cache(self, state: SourceState) -> str:
        return "cached" if state.get("registry", {}).get("frozen") else "fresh"

    def _segment_source(self, state: SourceState) -> dict[str, Any]:
        source = SourceInput.model_validate(state["source"])
        spans = (*make_spans(source.context_raw, "context"), *make_spans(source.query_raw, "query"))
        values = [item.model_dump(mode="json") for item in spans]
        return self._event(state, "segment_source", {"context_chars": len(source.context_raw), "query_chars": len(source.query_raw)}, {"spans": values}, spans=values)

    def _schema_overview(self, state: SourceState) -> dict[str, Any]:
        if not self.invoke_model_nodes:
            overview = {
                "schema_version": "schema-overview-v2",
                "source_summary": "Offline mode: semantic overview was not generated.",
                "type_hints": [],
                "hierarchy_hints": [],
                "unsafe_relation_warnings": [],
            }
            return self._event(
                state,
                "schema_overview",
                {"mode": "fake"},
                {"response": overview},
                overview=overview,
            )
        source = SourceInput.model_validate(state["source"])
        rendered = self.prompts.render(
            "schema_overview",
            {
                "source_id": source.source_id,
                "context_text": source.context_raw,
                "query_text": source.query_raw,
                "evidence_spans_json": canonical_json(state["spans"]),
                "graph_summary_json": canonical_json({"context": source.context_graph.model_dump(), "query": source.query_graph.model_dump()}),
                "policy_json": canonical_json(
                    {
                        "source_only": True,
                        "external_knowledge_is_not_evidence": True,
                        "overview_does_not_finalize_types": True,
                        "one_entity_per_followup_call": True,
                    }
                ),
            },
        )
        result = self.model.invoke(
            operation="schema_overview",
            messages=rendered.messages,
            output_schema=rendered.output_schema,
            idempotency_key=stable_id("request", {"source": source.source_id, "prompt": rendered.manifest_sha256}),
        )
        Draft202012Validator(rendered.output_schema).validate(result)
        return self._event(
            state,
            "schema_overview",
            {
                "operation": "schema_overview",
                "messages": [
                    {"type": getattr(message, "type", message.__class__.__name__), "content": message.content}
                    for message in rendered.messages
                ],
                "output_schema": rendered.output_schema,
                "idempotency_key": stable_id(
                    "request", {"source": source.source_id, "prompt": rendered.manifest_sha256}
                ),
            },
            {"response": result, "prompt_manifest_sha256": rendered.manifest_sha256},
            overview=dict(result),
        )

    def _derive_registry(self, state: SourceState) -> dict[str, Any]:
        source = SourceInput.model_validate(state["source"])
        spans = tuple(EvidenceSpan.model_validate(item) for item in state["spans"])
        types, assignments, nli_results, workflow_events = self._quality_workflow().type_source(
            source=source,
            spans=spans,
            overview=state["overview"],
        )
        type_values = [item.model_dump(mode="json") for item in types]
        assignment_values = [item.model_dump(mode="json") for item in assignments]
        nli_values = [item.model_dump(mode="json") for item in nli_results]
        enriched_state = {**state, "artifacts": [*state.get("artifacts", []), *workflow_events]}
        return self._event(
            enriched_state,
            "derive_registry",
            {"overview": state["overview"], "algorithm_version": ALGORITHM_VERSION},
            {
                "types": type_values,
                "assignments": assignment_values,
                "nli_results": nli_values,
            },
            types=type_values,
            assignments=assignment_values,
            source_nli_results=nli_values,
        )

    def _freeze_registry(self, state: SourceState) -> dict[str, Any]:
        if state.get("registry", {}).get("frozen"):
            FrozenRegistry.model_validate(state["registry"])
            return self._event(state, "freeze_registry", {"cache": True}, {"registry_id": state["registry"].get("registry_id")})
        source = SourceInput.model_validate(state["source"])
        bare = {
            "source_id": source.source_id,
            "context_graph_id": source.context_graph.graph_id,
            "query_graph_id": source.query_graph.graph_id,
            "types": state["types"],
            "assignments": state["assignments"],
            "evidence_spans": state["spans"],
            "nli_results": state.get("source_nli_results", []),
            "prompt_manifest_sha256": self.prompts.manifest_sha256,
            "frozen": True,
        }
        expected = {
            stable_id("node", {"graph": graph.graph_id, "entity": entity})
            for graph in (source.context_graph, source.query_graph)
            for entity in graph.entities
        }
        actual = {item["node_id"] for item in state["assignments"]}
        if actual != expected:
            raise InputContractError(
                f"source typing coverage mismatch: expected {len(expected)} graph vertices, got {len(actual)}"
            )
        FrozenRegistry.model_validate(
            {
                "registry_id": stable_id("registry-validation", bare),
                **bare,
                "registry_sha256": "0" * 64,
            }
        )
        checksum = registry_checksum(bare)
        registry = {"registry_id": stable_id("registry", bare), **bare, "registry_sha256": checksum}
        key = self._source_cache_key(source)
        self.cache.put_immutable("frozen_registry", key, registry)
        return self._event(state, "freeze_registry", {"cache_key": key}, {"registry": registry}, registry=registry)

    def _validate_answer(self, state: AnswerState) -> dict[str, Any]:
        answer = AnswerInput.model_validate(state["answer"])
        bare = answer.registry.model_dump(mode="json", exclude={"registry_id", "registry_sha256"})
        if registry_checksum(bare) != answer.registry.registry_sha256:
            raise InputContractError("frozen registry checksum is invalid")
        return self._event(state, "validate_answer", {"response_id": answer.response_id, "registry_id": answer.registry.registry_id}, {"valid": True}, answer=answer.model_dump(mode="json"))

    def _annotate_answer(self, state: AnswerState) -> dict[str, Any]:
        answer = AnswerInput.model_validate(state["answer"])
        assignments, nli_results, workflow_events = self._quality_workflow().type_answer(answer=answer)
        values = [item.model_dump(mode="json") for item in assignments]
        nli_values = [item.model_dump(mode="json") for item in nli_results]
        enriched_state = {**state, "artifacts": [*state.get("artifacts", []), *workflow_events]}
        return self._event(
            enriched_state,
            "annotate_answer",
            {"entities": list(answer.answer_graph.entities), "relations": list(answer.answer_graph.relations)},
            {"assignments": values, "nli_results": nli_values},
            annotations=values,
            nli_results=nli_values,
        )

    def _nli_answer(self, state: AnswerState) -> dict[str, Any]:
        values = list(state.get("nli_results", []))
        return self._event(
            state,
            "nli_answer",
            {"backend": self.nli_backend, "policy": "every entity decision was already NLI-audited"},
            {"results": values, "count": len(values)},
        )

    def _verify_nli(
        self,
        *,
        hypothesis_kind: str,
        premise: str,
        hypothesis: str,
        evidence_span_ids: tuple[str, ...],
        request_key: str,
    ) -> NliResult:
        if self.nli_backend in {"hhem", "fake"}:
            return self.nli.verify(
                hypothesis_kind=hypothesis_kind,
                premise=premise,
                hypothesis=hypothesis,
                evidence_span_ids=evidence_span_ids,
                idempotency_key=request_key,
            )
        rendered = self.prompts.render(
            "nli_verification",
            {
                "hypothesis_kind": hypothesis_kind,
                "premise": premise,
                "hypothesis": hypothesis,
                "evidence_span_ids_json": canonical_json(list(evidence_span_ids)),
                "policy_json": canonical_json({"closed_world": True, "neutral_is_contradiction": False}),
            },
        )
        result = self.model.invoke(
            operation="nli_verification",
            messages=rendered.messages,
            output_schema=rendered.output_schema,
            idempotency_key=request_key,
        )
        Draft202012Validator(rendered.output_schema).validate(result)
        return NliResult(
            request_id=stable_id("nli", {"key": request_key, "prompt": rendered.manifest_sha256}),
            verdict=result["verdict"],
            evidence_span_ids=tuple(result.get("supporting_span_ids", []) or result.get("conflicting_span_ids", [])),
            rationale=result["rationale"],
            evidence_level=result["evidence_level"],
            hypothesis_kind=hypothesis_kind,
            hypothesis=hypothesis,
        )

    def _emit_answer(self, state: AnswerState) -> dict[str, Any]:
        return self._event(state, "emit_answer", {"assignment_count": len(state.get("annotations", [])), "nli_count": len(state.get("nli_results", []))}, {"emitted": True})

    def write_run_artifacts(self, *, run_id: str, source_run: SourceRun, answer_run: AnswerRun | None = None) -> Path:
        writer = ArtifactWriter(self.artifacts_root / run_id)
        writer.write_json("source_registry.json", source_run.model_dump(mode="json"))
        if answer_run is not None:
            writer.write_json("answer_annotations.json", answer_run.model_dump(mode="json"))
        writer.write_json("manifest.json", {"run_id": run_id, "prompt_manifest_sha256": self.prompts.manifest_sha256, "backend": self.backend.value})
        writer.write_json("execution_trace.json", {"schema_version": "execution-trace-v1", "source_events": list(source_run.artifacts), "answer_events": list(answer_run.artifacts) if answer_run else []})
        return writer.root

    @staticmethod
    def _event(state: Mapping[str, Any], node: str, inputs: Mapping[str, Any], outputs: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
        """Append a serializable, secret-free observability event to the run artifact."""
        event = {"event": "node_completed", "node": node, "inputs": dict(inputs), "outputs": dict(outputs)}
        return {**updates, "artifacts": [*state.get("artifacts", []), event]}


def _legacy_disabled_type_relation(relation: str) -> bool:
    """Retained only for old imports; graph relations never assign entity types."""
    return False and normalize(relation) in {
        "is a",
        "is",
        "instance of",
        "type",
        "является",
        "является типом",
        "имеет тип",
        "тип",
    }


def graph_from_fixture(*, graph_id: str, role: str, payload: Mapping[str, Any]) -> GraphInput:
    return GraphInput(graph_id=graph_id, role=role, entities=tuple(payload.get("entities", [])), relations=tuple(tuple(item) for item in payload.get("relations", [])))
