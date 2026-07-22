"""Contracts and deterministic placeholders for post-KGGen dynamic typing.

This module deliberately has no prompt, model client or external knowledge source.
It defines the sealed boundary that a future type-induction agent must implement:
the registry is built from ``context + query`` only, frozen, and then used to
annotate the answer graph without changing the common KGGen graphs.  The system
design, artifact schema and future-agent responsibilities are documented in
``docs/dynamic-typing-experiment-infrastructure.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from src.matching import normalize

from .artifacts import canonical_json, sha256_bytes
from .shared_graphs import SharedGraphArtifact, SharedGraphBundle


TYPING_CONTRACT_VERSION = "dynamic-typing-contract-v1"
TYPE_STATUS_AVAILABLE = "available"
TYPE_STATUS_UNKNOWN = "unknown"


def _stable_id(namespace: str, payload: Mapping[str, Any]) -> str:
    digest = sha256_bytes(canonical_json(dict(payload)).encode("utf-8"))
    return f"{namespace}:{digest}"


def graph_node_id(graph: SharedGraphArtifact, surface_text: str) -> str:
    """Stable temporary node identity until the future rich graph IR exposes node IDs."""
    return _stable_id(
        "node",
        {"graph_id": graph.graph_id, "surface_text": str(surface_text)},
    )


@dataclass(frozen=True)
class TypeAssignment:
    """One type decision with explicit abstention and graph provenance."""

    assignment_id: str
    node_id: str
    graph_id: str
    graph_role: str
    surface_text: str
    normalized_text: str
    type_label: str | None
    status: str
    assignment_method: str
    confidence: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "type_assignment_id": self.assignment_id,
            "node_id": self.node_id,
            "graph_id": self.graph_id,
            "graph_role": self.graph_role,
            "surface_text": self.surface_text,
            "normalized_text": self.normalized_text,
            "type_label": self.type_label,
            "assignment_status": self.status,
            "assignment_method": self.assignment_method,
            "confidence": self.confidence,
            "typing_contract_version": TYPING_CONTRACT_VERSION,
        }


@dataclass(frozen=True)
class SourceTypingInput:
    """No-gold input available while inducing a source-local type registry."""

    source_id: str
    context_raw: str
    query_raw: str
    graph_bundle_id: str
    context_graph: SharedGraphArtifact
    query_graph: SharedGraphArtifact


@dataclass(frozen=True)
class AnswerTypingInput:
    """No-gold answer input; it cannot mutate a previously frozen registry."""

    source_id: str
    response_id: str
    response_raw: str
    graph_bundle_id: str
    answer_graph: SharedGraphArtifact


@dataclass(frozen=True)
class SourceTypeRegistry:
    """A source-only type inventory frozen before the answer is annotated."""

    registry_id: str
    source_id: str
    context_graph_id: str
    query_graph_id: str
    registry_labels: tuple[str, ...]
    assignments: tuple[TypeAssignment, ...]
    frozen: bool
    provider_name: str
    provider_version: str

    def to_record(self) -> dict[str, Any]:
        return {
            "type_registry_id": self.registry_id,
            "source_id": self.source_id,
            "context_graph_id": self.context_graph_id,
            "query_graph_id": self.query_graph_id,
            "registry_labels": list(self.registry_labels),
            "assignments": [assignment.to_record() for assignment in self.assignments],
            "frozen": self.frozen,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "typing_contract_version": TYPING_CONTRACT_VERSION,
        }


@dataclass(frozen=True)
class TypeAnnotationBundle:
    """Frozen source registry plus annotations for one response graph."""

    annotation_bundle_id: str
    source_id: str
    response_id: str
    graph_bundle_id: str
    registry: SourceTypeRegistry
    answer_graph_id: str
    answer_assignments: tuple[TypeAssignment, ...]
    status: str

    def reference(self) -> dict[str, str]:
        return {
            "type_annotation_bundle_id": self.annotation_bundle_id,
            "type_registry_id": self.registry.registry_id,
            "typing_contract_version": TYPING_CONTRACT_VERSION,
            "typing_status": self.status,
        }

    def summary(self) -> dict[str, Any]:
        available = sum(item.status == TYPE_STATUS_AVAILABLE for item in self.answer_assignments)
        return {
            **self.reference(),
            "source_registry_frozen": self.registry.frozen,
            "source_registry_label_count": len(self.registry.registry_labels),
            "answer_assignment_count": len(self.answer_assignments),
            "answer_available_type_count": available,
            "answer_unknown_type_count": len(self.answer_assignments) - available,
            "provider_name": self.registry.provider_name,
            "provider_version": self.registry.provider_version,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.reference(),
            "source_id": self.source_id,
            "response_id": self.response_id,
            "shared_graph_bundle_id": self.graph_bundle_id,
            "answer_graph_id": self.answer_graph_id,
            "registry": self.registry.to_record(),
            "answer_assignments": [assignment.to_record() for assignment in self.answer_assignments],
        }


@runtime_checkable
class DynamicTypingProvider(Protocol):
    """Future agent boundary; implementations must stay source-only and no-gold."""

    provider_name: str
    provider_version: str

    def build_source_registry(self, request: SourceTypingInput) -> SourceTypeRegistry:
        """Return a frozen registry from context/query graphs and their source text only."""

    def annotate_answer(
        self, request: AnswerTypingInput, registry: SourceTypeRegistry
    ) -> TypeAnnotationBundle:
        """Assign existing registry labels or abstain; do not extend ``registry``."""

    def prepare(self, *, item: Any, bundle: SharedGraphBundle) -> TypeAnnotationBundle:
        """Build/reuse the source registry and annotate the answer for one no-gold item."""

    def artifact_records(self) -> Mapping[str, Iterable[Mapping[str, Any]]]:
        """Expose immutable archive rows without making the runner type-aware."""


class StaticTypingProvider:
    """Deterministic no-network provider for infrastructure tests and offline plumbing.

    ``labels_by_surface`` can emulate a tiny known registry.  An answer label is
    accepted only if the same label appeared in the frozen source registry; this
    enforces the future source-only contract even in the stub.
    """

    provider_name = "static_dynamic_typing_stub"
    provider_version = "v1"

    def __init__(self, labels_by_surface: Mapping[str, str] | None = None):
        self.labels_by_surface = {
            normalize(surface): str(label)
            for surface, label in (labels_by_surface or {}).items()
            if normalize(surface) and str(label).strip()
        }
        self._registries: dict[str, SourceTypeRegistry] = {}
        self._annotations: dict[str, TypeAnnotationBundle] = {}

    def _assignment(self, graph: SharedGraphArtifact, *, role: str, surface_text: str) -> TypeAssignment:
        normalized = normalize(surface_text)
        label = self.labels_by_surface.get(normalized)
        status = TYPE_STATUS_AVAILABLE if label is not None else TYPE_STATUS_UNKNOWN
        payload = {
            "graph_id": graph.graph_id,
            "role": role,
            "surface_text": str(surface_text),
            "label": label,
            "provider_version": self.provider_version,
        }
        return TypeAssignment(
            assignment_id=_stable_id("type-assignment", payload),
            node_id=graph_node_id(graph, str(surface_text)),
            graph_id=graph.graph_id,
            graph_role=role,
            surface_text=str(surface_text),
            normalized_text=normalized,
            type_label=label,
            status=status,
            assignment_method=self.provider_name,
            confidence=1.0 if label is not None else None,
        )

    def _assign_graph(self, graph: SharedGraphArtifact, *, role: str) -> tuple[TypeAssignment, ...]:
        return tuple(
            self._assignment(graph, role=role, surface_text=entity)
            for entity in sorted(graph.graph.entities)
        )

    def build_source_registry(self, request: SourceTypingInput) -> SourceTypeRegistry:
        cache_key = f"{request.source_id}:{request.context_graph.graph_id}:{request.query_graph.graph_id}"
        cached = self._registries.get(cache_key)
        if cached is not None:
            return cached
        assignments = (
            *self._assign_graph(request.context_graph, role="context"),
            *self._assign_graph(request.query_graph, role="query"),
        )
        labels = tuple(sorted({item.type_label for item in assignments if item.type_label is not None}))
        registry = SourceTypeRegistry(
            registry_id=_stable_id(
                "type-registry",
                {
                    "source_id": request.source_id,
                    "context_graph_id": request.context_graph.graph_id,
                    "query_graph_id": request.query_graph.graph_id,
                    "labels": labels,
                    "provider_version": self.provider_version,
                },
            ),
            source_id=request.source_id,
            context_graph_id=request.context_graph.graph_id,
            query_graph_id=request.query_graph.graph_id,
            registry_labels=labels,
            assignments=tuple(assignments),
            frozen=True,
            provider_name=self.provider_name,
            provider_version=self.provider_version,
        )
        self._registries[cache_key] = registry
        return registry

    def annotate_answer(
        self, request: AnswerTypingInput, registry: SourceTypeRegistry
    ) -> TypeAnnotationBundle:
        cache_key = f"{request.response_id}:{request.answer_graph.graph_id}:{registry.registry_id}"
        cached = self._annotations.get(cache_key)
        if cached is not None:
            return cached
        proposed = self._assign_graph(request.answer_graph, role="response")
        assignments: list[TypeAssignment] = []
        allowed = set(registry.registry_labels)
        for assignment in proposed:
            if assignment.type_label is None or assignment.type_label in allowed:
                assignments.append(assignment)
                continue
            # A response cannot enlarge the source registry. Preserve abstention rather
            # than silently accepting a label introduced only by the answer.
            assignments.append(
                TypeAssignment(
                    assignment_id=assignment.assignment_id,
                    node_id=assignment.node_id,
                    graph_id=assignment.graph_id,
                    graph_role=assignment.graph_role,
                    surface_text=assignment.surface_text,
                    normalized_text=assignment.normalized_text,
                    type_label=None,
                    status=TYPE_STATUS_UNKNOWN,
                    assignment_method=assignment.assignment_method,
                    confidence=None,
                )
            )
        bundle = TypeAnnotationBundle(
            annotation_bundle_id=_stable_id(
                "type-annotation-bundle",
                {
                    "source_id": request.source_id,
                    "response_id": request.response_id,
                    "graph_bundle_id": request.graph_bundle_id,
                    "registry_id": registry.registry_id,
                    "answer_graph_id": request.answer_graph.graph_id,
                    "assignments": [item.assignment_id for item in assignments],
                },
            ),
            source_id=request.source_id,
            response_id=request.response_id,
            graph_bundle_id=request.graph_bundle_id,
            registry=registry,
            answer_graph_id=request.answer_graph.graph_id,
            answer_assignments=tuple(assignments),
            status="ok",
        )
        self._annotations[cache_key] = bundle
        return bundle

    def prepare(self, *, item: Any, bundle: SharedGraphBundle) -> TypeAnnotationBundle:
        source_request = SourceTypingInput(
            source_id=str(item.source_id),
            context_raw=str(item.context),
            query_raw=str(item.query or ""),
            graph_bundle_id=bundle.bundle_id,
            context_graph=bundle.context,
            query_graph=bundle.query,
        )
        registry = self.build_source_registry(source_request)
        answer_request = AnswerTypingInput(
            source_id=str(item.source_id),
            response_id=str(item.response_id),
            response_raw=str(item.response),
            graph_bundle_id=bundle.bundle_id,
            answer_graph=bundle.response,
        )
        return self.annotate_answer(answer_request, registry)

    def artifact_records(self) -> Mapping[str, Iterable[Mapping[str, Any]]]:
        return {
            "typing/type_registries.jsonl": [
                registry.to_record() for _, registry in sorted(self._registries.items())
            ],
            "typing/type_annotation_bundles.jsonl": [
                annotation.to_record() for _, annotation in sorted(self._annotations.items())
            ],
        }


class UnknownTypeProvider(StaticTypingProvider):
    """The primary placeholder: every entity abstains, so scoring must equal B0."""

    provider_name = "unknown_dynamic_typing_stub"
    provider_version = "v1"

    def __init__(self):
        super().__init__({})


__all__ = [
    "AnswerTypingInput",
    "DynamicTypingProvider",
    "SourceTypeRegistry",
    "SourceTypingInput",
    "StaticTypingProvider",
    "TYPE_STATUS_AVAILABLE",
    "TYPE_STATUS_UNKNOWN",
    "TYPING_CONTRACT_VERSION",
    "TypeAnnotationBundle",
    "TypeAssignment",
    "UnknownTypeProvider",
    "graph_node_id",
]
