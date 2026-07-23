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
    NodeSpec("schema_overview", NodeKind.MODEL, "source", "schema_overview", description="Draft local distinctions from context/query."),
    NodeSpec("schema_reconcile", NodeKind.MODEL, "source", "schema_reconcile", description="Merge chunk drafts without inventing evidence."),
    NodeSpec("build_entity_profiles", NodeKind.DETERMINISTIC, "source", description="Join nodes, relations and source spans."),
    NodeSpec("retrieve_type_candidates", NodeKind.DETERMINISTIC, "source", description="Bound candidate sets; similarity is not a verdict."),
    NodeSpec("entity_type_decision", NodeKind.MODEL, "source", "entity_type_decision", description="Choose closed-set actions in batches."),
    NodeSpec("route_source_nli", NodeKind.DETERMINISTIC, "source", description="Route only ambiguous or high-impact decisions."),
    NodeSpec("nli_verify_source", NodeKind.NLI, "source", "nli_verification", description="Three-way source-grounded verification."),
    NodeSpec("registry_consistency_review", NodeKind.MODEL, "source", "registry_consistency_review", description="Review close type pairs globally."),
    NodeSpec("cluster_split_review", NodeKind.MODEL, "source", "cluster_split_review", description="Review heterogeneous type clusters."),
    NodeSpec("validate_registry", NodeKind.DETERMINISTIC, "source", description="Check IDs, DAG constraints, evidence and history."),
    NodeSpec("registry_repair", NodeKind.MODEL, "source", "registry_repair", description="Bounded repair from machine-readable violations."),
    NodeSpec("freeze_registry", NodeKind.BOUNDARY, "source", description="Hash and serialize the source-only registry."),
    NodeSpec("validate_answer_input", NodeKind.BOUNDARY, "answer", may_read_answer=True, description="Require a valid frozen registry."),
    NodeSpec("build_answer_profiles", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="Join answer nodes and explicit type phrases."),
    NodeSpec("answer_typing", NodeKind.MODEL, "answer", "answer_typing", True, "Assign registered types or abstain."),
    NodeSpec("route_answer_nli", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="Route explicit specializations/conflicts."),
    NodeSpec("nli_verify_answer", NodeKind.NLI, "answer", "nli_verification", True, "Verify answer type assertions against source."),
    NodeSpec("global_entity_alignment", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="One-to-one identity alignment; type is secondary."),
    NodeSpec("edge_candidate_resolution", NodeKind.MODEL, "answer", "edge_candidate_resolution", True, "Resolve bounded ambiguous edge candidates."),
    NodeSpec("route_edge_nli", NodeKind.DETERMINISTIC, "answer", may_read_answer=True, description="Route uncertain relation/direction/role claims."),
    NodeSpec("nli_verify_edge", NodeKind.NLI, "answer", "nli_verification", True, "Verify a full edge hypothesis against source."),
    NodeSpec("emit_annotation_bundle", NodeKind.BOUNDARY, "answer", may_read_answer=True, description="Emit auditable annotations, never a score."),
)

MODEL_NODE_IDS = frozenset(
    item.node_id for item in NODE_SPECS if item.kind in {NodeKind.MODEL, NodeKind.NLI}
)

