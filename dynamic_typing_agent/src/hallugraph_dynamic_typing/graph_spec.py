"""Executable inventory of graph nodes for architecture and prompt coverage tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NodeKind(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    NLI = "nli"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_id: str
    kind: NodeKind
    phase: str
    prompt_id: str | None = None
    may_read_answer: bool = False
    description: str = ""


NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec("validate_source_input", NodeKind.BOUNDARY, "source", description="Reject forbidden/gold fields."),
    NodeSpec("resolve_source_cache", NodeKind.DETERMINISTIC, "source", description="Load only hash-compatible immutable artifacts."),
    NodeSpec("segment_source", NodeKind.DETERMINISTIC, "source", description="Create stable evidence spans."),
    NodeSpec("schema_overview", NodeKind.MODEL, "source", "schema_overview", description="Create hints, never final types."),
    NodeSpec("build_entity_profiles", NodeKind.DETERMINISTIC, "source", description="Join nodes, relations and source spans."),
    NodeSpec("entity_type_decision", NodeKind.MODEL, "source", "entity_type_decision", description="Type exactly one source entity per call."),
    NodeSpec("nli_verify_source", NodeKind.NLI, "source", "nli_verification", description="Verify every semantic source assignment."),
    NodeSpec("commit_entity_type", NodeKind.DETERMINISTIC, "source", description="Commit entailed types or structural root fallback."),
    NodeSpec("registry_consistency_review", NodeKind.MODEL, "source", "registry_consistency_review", description="Propose bounded hierarchy or equivalence changes."),
    NodeSpec("nli_verify_hierarchy", NodeKind.NLI, "source", "nli_verification", description="Verify every parent or merge proposal."),
    NodeSpec("validate_registry", NodeKind.DETERMINISTIC, "source", description="Require full coverage, final types and an acyclic graph."),
    NodeSpec("freeze_registry", NodeKind.BOUNDARY, "source", description="Hash and serialize the source-only registry."),
    NodeSpec("validate_answer_input", NodeKind.BOUNDARY, "answer", may_read_answer=True, description="Require a valid frozen registry."),
    NodeSpec("build_answer_profiles", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="Join answer nodes and explicit type phrases."),
    NodeSpec("answer_typing", NodeKind.MODEL, "answer", "answer_typing", True, "Assign frozen types to one answer entity per call."),
    NodeSpec("nli_verify_answer", NodeKind.NLI, "answer", "nli_verification", True, "Verify every semantic answer assignment."),
    NodeSpec("validate_answer_coverage", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="Require one final assignment per answer vertex."),
    NodeSpec("emit_annotation_bundle", NodeKind.BOUNDARY, "answer", may_read_answer=True, description="Emit auditable annotations, never a score."),
)

MODEL_NODE_IDS = frozenset(
    item.node_id for item in NODE_SPECS if item.kind in {NodeKind.MODEL, NodeKind.NLI}
)
