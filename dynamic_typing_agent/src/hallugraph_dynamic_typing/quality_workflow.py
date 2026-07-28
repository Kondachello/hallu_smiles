"""Entity-by-entity typing and NLI-gated registry finalization.

This module contains the semantic core.  It deliberately does not read gold labels,
write artifacts, or mutate input graphs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from .errors import InputContractError
from .models import (
    AnswerInput,
    EvidenceLevel,
    EvidenceSpan,
    NliResult,
    SourceInput,
    TypeAssignment,
    TypeDefinition,
    canonical_json,
    normalize,
    stable_id,
)


ROOT_TYPE_ID = "T-ENTITY"
ALGORITHM_VERSION = "entity-by-entity-nli-v4-best-effort"

# When no proposed semantic type is NLI-confirmed within the attempt budget, the
# entity would otherwise collapse to the structural root ("entity"), a useless
# service type. Instead we keep the most plausible proposal ranked by NLI verdict
# (entailed > neutral > contradicted) and then evidence strength.
_VERDICT_RANK = {"entailed": 3, "neutral": 2, "contradicted": 1}
_EVIDENCE_RANK = {
    EvidenceLevel.SOURCE_ENTAILED: 3,
    EvidenceLevel.DEFINITION_ONLY: 2,
    EvidenceLevel.UNKNOWN: 1,
}


def root_type() -> TypeDefinition:
    return TypeDefinition(
        type_id=ROOT_TYPE_ID,
        label="entity",
        definition="Structural root for every knowledge-graph vertex.",
        parent_type_ids=(),
        evidence_span_ids=(),
        evidence_level=EvidenceLevel.SOURCE_ENTAILED,
        status="final",
    )


def type_id_for_label(label: str) -> str:
    return "T-" + hashlib.sha256(normalize(label).encode("utf-8")).hexdigest()[:12].upper()


def _span_map(spans: Sequence[EvidenceSpan]) -> dict[str, EvidenceSpan]:
    return {span.span_id: span for span in spans}


def _relevant_span_ids(surface: str, spans: Sequence[EvidenceSpan]) -> tuple[str, ...]:
    key = normalize(surface)
    return tuple(span.span_id for span in spans if key and key in normalize(span.text))


def _premise(span_ids: Sequence[str], spans_by_id: Mapping[str, EvidenceSpan]) -> str:
    return " ".join(spans_by_id[item].text for item in span_ids if item in spans_by_id)


def _message_payload(messages: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": getattr(message, "content", str(message)),
        }
        for message in messages
    ]


def build_source_profiles(source: SourceInput, spans: Sequence[EvidenceSpan]) -> list[dict[str, Any]]:
    """Join each unique source surface with every graph occurrence and local edge."""
    profiles: dict[str, dict[str, Any]] = {}
    for graph in (source.context_graph, source.query_graph):
        for entity in graph.entities:
            key = normalize(entity)
            profile = profiles.setdefault(
                key,
                {
                    "entity_id": stable_id("source-entity", {"source": source.source_id, "surface": key}),
                    "surface_text": entity,
                    "occurrences": [],
                    "neighbourhood": [],
                    "evidence_span_ids": list(_relevant_span_ids(entity, spans)),
                },
            )
            profile["occurrences"].append(
                {
                    "graph_id": graph.graph_id,
                    "graph_role": graph.role,
                    "node_id": stable_id("node", {"graph": graph.graph_id, "entity": entity}),
                }
            )
        for subject, relation, obj in graph.relations:
            triple = {"subject": subject, "relation": relation, "object": obj, "graph_role": graph.role}
            for surface in (subject, obj):
                key = normalize(surface)
                if key in profiles and triple not in profiles[key]["neighbourhood"]:
                    profiles[key]["neighbourhood"].append(triple)
    return sorted(
        profiles.values(),
        key=lambda item: (-len(item["neighbourhood"]), normalize(item["surface_text"])),
    )


def build_answer_profiles(answer: AnswerInput) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for entity in answer.answer_graph.entities:
        neighbourhood = [
            {"subject": subject, "relation": relation, "object": obj}
            for subject, relation, obj in answer.answer_graph.relations
            if normalize(subject) == normalize(entity) or normalize(obj) == normalize(entity)
        ]
        profiles.append(
            {
                "entity_id": stable_id(
                    "answer-entity",
                    {"response": answer.response_id, "graph": answer.answer_graph.graph_id, "surface": normalize(entity)},
                ),
                "node_id": stable_id("node", {"graph": answer.answer_graph.graph_id, "entity": entity}),
                "surface_text": entity,
                "neighbourhood": neighbourhood,
            }
        )
    return sorted(profiles, key=lambda item: (-len(item["neighbourhood"]), normalize(item["surface_text"])))


class QualityTypingWorkflow:
    """Sequential semantic workflow used by LangGraph stage nodes."""

    def __init__(
        self,
        *,
        prompts: Any,
        model: Any,
        invoke_model_nodes: bool,
        verify_nli: Callable[..., NliResult],
        max_entity_attempts: int = 3,
        retry_on_neutral: bool = True,
    ):
        self.prompts = prompts
        self.model = model
        self.invoke_model_nodes = invoke_model_nodes
        self.verify_nli = verify_nli
        self.max_entity_attempts = max(1, max_entity_attempts)
        # When True a purely-neutral attempt still triggers another proposal to
        # seek an entailed type before falling back to the best-effort choice.
        self.retry_on_neutral = retry_on_neutral

    def type_source(
        self,
        *,
        source: SourceInput,
        spans: Sequence[EvidenceSpan],
        overview: Mapping[str, Any],
    ) -> tuple[list[TypeDefinition], list[TypeAssignment], list[NliResult], list[dict[str, Any]]]:
        spans_by_id = _span_map(spans)
        profiles = build_source_profiles(source, spans)
        types: dict[str, TypeDefinition] = {ROOT_TYPE_ID: root_type()}
        assignments: list[TypeAssignment] = []
        nli_results: list[NliResult] = []
        events: list[dict[str, Any]] = [
            {
                "event": "node_completed",
                "node": "build_entity_profiles",
                "inputs": {"source_id": source.source_id},
                "outputs": {"profile_count": len(profiles), "profiles": profiles},
            }
        ]

        for profile in profiles:
            accepted_ids: list[str] = []
            accepted_new: dict[str, TypeDefinition] = {}
            previous_attempt: dict[str, Any] | None = None
            last_reason = ""
            # Every semantic (non-root) type NLI-checked across all attempts, kept so
            # that an entity with no confirmed type can still fall back to the most
            # plausible proposal rather than to the useless structural root.
            best_effort_pool: list[dict[str, Any]] = []
            for attempt in range(1, self.max_entity_attempts + 1):
                decision, model_event = self._source_decision(
                    source=source,
                    profile=profile,
                    overview=overview,
                    types=types,
                    previous_attempt=previous_attempt,
                    attempt=attempt,
                )
                events.append(model_event)
                if decision["entity_id"] != profile["entity_id"]:
                    # entity_type_decision types exactly one entity per call, so the
                    # decision is unambiguously about `profile`; some models echo a
                    # wrong/hallucinated entity_id. Coerce it to the known id rather
                    # than discarding the whole record over a cosmetic mismatch.
                    decision["entity_id"] = profile["entity_id"]
                selected = list(dict.fromkeys(str(item) for item in decision["selected_type_ids"]))
                unknown_selected = [type_id for type_id in selected if type_id not in types]
                selected = [type_id for type_id in selected if type_id in types]
                if unknown_selected and not decision["proposed_types"] and attempt < self.max_entity_attempts:
                    previous_attempt = {
                        "attempt": attempt,
                        "decision": decision,
                        "protocol_error": {
                            "unknown_selected_type_ids": unknown_selected,
                            "instruction": (
                                "selected_type_ids may contain only IDs from CURRENT FINAL REGISTRY. "
                                "To introduce another type, return its complete definition in proposed_types."
                            ),
                        },
                    }
                    events.append(
                        {
                            "event": "node_rejected",
                            "node": "validate_entity_type_decision",
                            "inputs": {"entity_id": profile["entity_id"], "attempt": attempt},
                            "outputs": {"unknown_selected_type_ids": unknown_selected, "retry": True},
                        }
                    )
                    continue

                proposed: dict[str, tuple[str, TypeDefinition]] = {}
                rejected_identity_labels: list[str] = []
                for item in decision["proposed_types"]:
                    label = str(item["label"]).strip()
                    if normalize(label) == normalize(profile["surface_text"]):
                        rejected_identity_labels.append(label)
                        continue
                    candidate_id = str(item["candidate_id"])
                    type_id = type_id_for_label(label)
                    existing = next((value for value in types.values() if normalize(value.label) == normalize(label)), None)
                    if existing is not None:
                        selected.append(existing.type_id)
                        continue
                    evidence_ids = tuple(item["evidence_span_ids"])
                    if any(span_id not in spans_by_id for span_id in evidence_ids):
                        raise InputContractError("proposed type cites an unknown evidence span")
                    proposed[candidate_id] = (
                        type_id,
                        TypeDefinition(
                            type_id=type_id,
                            label=label,
                            definition=str(item["definition"]),
                            parent_type_ids=(ROOT_TYPE_ID,),
                            aliases=tuple(str(alias) for alias in item["aliases"]),
                            evidence_span_ids=evidence_ids,
                            evidence_level=EvidenceLevel.SOURCE_ENTAILED,
                            status="final",
                        ),
                    )

                if rejected_identity_labels and not selected and not proposed and attempt < self.max_entity_attempts:
                    previous_attempt = {
                        "attempt": attempt,
                        "decision": decision,
                        "protocol_error": {
                            "identity_like_type_labels": rejected_identity_labels,
                            "instruction": (
                                "A graph vertex cannot be typed by merely copying its surface label. "
                                "Choose an immediate broader reusable category, for example loan -> "
                                "financial agreement or commercial bank -> financial institution."
                            ),
                        },
                    }
                    events.append(
                        {
                            "event": "node_rejected",
                            "node": "validate_entity_type_decision",
                            "inputs": {"entity_id": profile["entity_id"], "attempt": attempt},
                            "outputs": {
                                "identity_like_type_labels": rejected_identity_labels,
                                "retry": True,
                            },
                        }
                    )
                    continue

                targets: list[tuple[str, str, str]] = []
                for type_id in dict.fromkeys(selected):
                    targets.append((type_id, type_id, types[type_id].label))
                for candidate_id, (type_id, definition) in proposed.items():
                    targets.append((candidate_id, type_id, definition.label))
                hypotheses = {str(item["target_ref"]): item for item in decision["hypotheses"]}
                accepted_ids = []
                accepted_new = {}
                attempt_results: list[dict[str, Any]] = []

                if not targets:
                    targets = [(ROOT_TYPE_ID, ROOT_TYPE_ID, "entity")]

                for target_ref, type_id, label in targets:
                    hypothesis_record = hypotheses.get(target_ref)
                    evidence_ids = tuple(hypothesis_record["evidence_span_ids"]) if hypothesis_record else tuple(
                        profile["evidence_span_ids"]
                    )
                    evidence_ids = tuple(item for item in evidence_ids if item in spans_by_id)
                    if not evidence_ids:
                        evidence_ids = tuple(profile["evidence_span_ids"]) or tuple(spans_by_id)[:1]
                    hypothesis = f"{profile['surface_text']} is a {label}."
                    result = self.verify_nli(
                        hypothesis_kind="source_type_assignment",
                        premise=_premise(evidence_ids, spans_by_id),
                        hypothesis=hypothesis,
                        evidence_span_ids=evidence_ids,
                        request_key=stable_id(
                            "nli-request",
                            {
                                "source": source.source_id,
                                "entity": profile["entity_id"],
                                "type": type_id,
                                "attempt": attempt,
                                "hypothesis": hypothesis,
                            },
                        ),
                    ).model_copy(
                        update={
                            "hypothesis_kind": "source_type_assignment",
                            "hypothesis": hypothesis,
                            "subject_id": profile["entity_id"],
                            "target_type_id": type_id,
                        }
                    )
                    nli_results.append(result)
                    attempt_results.append(result.model_dump(mode="json"))
                    if type_id != ROOT_TYPE_ID:
                        candidate_def = (
                            proposed[target_ref][1] if target_ref in proposed else types.get(type_id)
                        )
                        best_effort_pool.append(
                            {
                                "type_id": type_id,
                                "label": label,
                                "verdict": result.verdict,
                                "evidence_level": result.evidence_level,
                                "definition": candidate_def,
                                "attempt": attempt,
                            }
                        )
                    if type_id == ROOT_TYPE_ID or result.verdict != "contradicted":
                        accepted_ids.append(type_id)
                        if target_ref in proposed:
                            evidence_level = (
                                EvidenceLevel.SOURCE_ENTAILED
                                if result.verdict == "entailed"
                                else EvidenceLevel.DEFINITION_ONLY
                            )
                            accepted_new[type_id] = proposed[target_ref][1].model_copy(
                                update={"evidence_level": evidence_level}
                            )

                events.append(
                    {
                        "event": "node_completed",
                        "node": "nli_verify_source",
                        "inputs": {
                            "entity_id": profile["entity_id"],
                            "surface_text": profile["surface_text"],
                            "attempt": attempt,
                        },
                        "outputs": {"results": attempt_results, "accepted_type_ids": accepted_ids},
                    }
                )
                semantic_targets = [item for item in targets if item[1] != ROOT_TYPE_ID]
                this_attempt = [item for item in best_effort_pool if item["attempt"] == attempt]
                attempt_entailed = [item for item in this_attempt if item["verdict"] == "entailed"]
                attempt_neutral = [item for item in this_attempt if item["verdict"] == "neutral"]
                last_reason = str(decision["reason"])
                # Stop as soon as a source-entailed type is found.
                if attempt_entailed:
                    break
                # Nothing semantic to type this attempt: stop and let the fallback decide.
                if not semantic_targets:
                    break
                # Neutral candidates exist: accept immediately unless we keep trying for entailed.
                if attempt_neutral and not self.retry_on_neutral:
                    break
                if attempt == self.max_entity_attempts:
                    break
                previous_attempt = {
                    "attempt": attempt,
                    "decision": decision,
                    "nli_results": attempt_results,
                    "instruction": (
                        "NLI did not source-confirm the previous type. Propose a DIFFERENT, "
                        "narrower reusable category supported by the evidence spans; do not repeat "
                        "a neutral or contradicted claim and do not restate the entity name."
                    ),
                }

            for type_id, definition in accepted_new.items():
                types.setdefault(type_id, definition)
            semantic_ids = tuple(sorted({type_id for type_id in accepted_ids if type_id != ROOT_TYPE_ID}))
            if not semantic_ids and best_effort_pool:
                # No NLI-confirmed type. Rather than collapse to the structural root
                # ("entity"), keep the most plausible proposal by NLI verdict, then
                # evidence strength. Ties keep the earliest (most specific) proposal.
                ranked = sorted(
                    best_effort_pool,
                    key=lambda item: (
                        _VERDICT_RANK.get(item["verdict"], 0),
                        _EVIDENCE_RANK.get(item["evidence_level"], 0),
                    ),
                    reverse=True,
                )
                best = ranked[0]
                chosen_id = str(best["type_id"])
                definition_obj = best["definition"]
                if definition_obj is not None:
                    types.setdefault(
                        chosen_id,
                        definition_obj.model_copy(update={"evidence_level": EvidenceLevel.UNKNOWN}),
                    )
                if chosen_id in types:
                    semantic_ids = (chosen_id,)
                    last_reason = (
                        f"Best-effort type '{best['label']}' kept after {self.max_entity_attempts} "
                        f"attempt(s); highest NLI verdict was '{best['verdict']}' with no source-entailed "
                        f"type. Chosen over the structural root 'entity'."
                    )
                    events.append(
                        {
                            "event": "node_completed",
                            "node": "type_best_effort_fallback",
                            "inputs": {
                                "entity_id": profile["entity_id"],
                                "surface_text": profile["surface_text"],
                                "attempts": self.max_entity_attempts,
                            },
                            "outputs": {
                                "chosen_type_id": chosen_id,
                                "chosen_label": best["label"],
                                "chosen_verdict": best["verdict"],
                                "ranked_candidates": [
                                    {
                                        "type_id": item["type_id"],
                                        "label": item["label"],
                                        "verdict": item["verdict"],
                                        "evidence_level": str(item["evidence_level"]),
                                        "attempt": item["attempt"],
                                    }
                                    for item in ranked
                                ],
                            },
                        }
                    )
            final_ids = semantic_ids or (ROOT_TYPE_ID,)
            for occurrence in profile["occurrences"]:
                assignments.append(
                    TypeAssignment(
                        node_id=occurrence["node_id"],
                        surface_text=profile["surface_text"],
                        graph_role=occurrence["graph_role"],
                        type_ids=final_ids,
                        status="assigned",
                        evidence_span_ids=tuple(profile["evidence_span_ids"]),
                        reason=last_reason or "Structural entity fallback after NLI-gated typing.",
                    )
                )
            events.append(
                {
                    "event": "node_completed",
                    "node": "commit_entity_type",
                    "inputs": {"entity_id": profile["entity_id"], "surface_text": profile["surface_text"]},
                    "outputs": {"final_type_ids": list(final_ids), "occurrences": profile["occurrences"]},
                }
            )

        types, assignments, hierarchy_nli, hierarchy_events = self._review_hierarchy(
            source=source,
            spans=spans,
            overview=overview,
            types=types,
            assignments=assignments,
        )
        nli_results.extend(hierarchy_nli)
        events.extend(hierarchy_events)
        return (
            sorted(types.values(), key=lambda item: item.type_id),
            sorted(assignments, key=lambda item: item.node_id),
            nli_results,
            events,
        )

    def _source_decision(
        self,
        *,
        source: SourceInput,
        profile: Mapping[str, Any],
        overview: Mapping[str, Any],
        types: Mapping[str, TypeDefinition],
        previous_attempt: Mapping[str, Any] | None,
        attempt: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.invoke_model_nodes:
            decision = {
                "schema_version": "entity-type-decision-v2",
                "entity_id": profile["entity_id"],
                "selected_type_ids": [ROOT_TYPE_ID],
                "proposed_types": [],
                "role_labels": [],
                "hypotheses": [],
                "reason": "Offline structural fallback; no model was invoked.",
            }
            return decision, {
                "event": "node_completed",
                "node": "entity_type_decision",
                "inputs": {"entity_id": profile["entity_id"], "attempt": attempt, "mode": "fake"},
                "outputs": {"response": decision},
            }
        rendered = self.prompts.render(
            "entity_type_decision",
            {
                "source_id": source.source_id,
                "policy_json": canonical_json(
                    {
                        "one_entity_per_call": True,
                        "nli_for_every_semantic_assignment": True,
                        "no_arbitrary_relation_object_typing": True,
                        "final_types_only": True,
                    }
                ),
                "overview_json": canonical_json(overview),
                "entity_profile_json": canonical_json(profile),
                "current_registry_json": canonical_json(
                    [item.model_dump(mode="json") for item in sorted(types.values(), key=lambda value: value.type_id)]
                ),
                "previous_attempt_json": canonical_json(previous_attempt),
            },
        )
        request_key = stable_id(
            "request",
            {
                "source": source.source_id,
                "entity": profile["entity_id"],
                "attempt": attempt,
                "registry": sorted(types),
                "prompt": rendered.manifest_sha256,
            },
        )
        result = dict(
            self.model.invoke(
                operation="entity_type_decision",
                messages=rendered.messages,
                output_schema=rendered.output_schema,
                idempotency_key=request_key,
            )
        )
        Draft202012Validator(rendered.output_schema).validate(result)
        return result, {
            "event": "node_completed",
            "node": "entity_type_decision",
            "inputs": {
                "entity_id": profile["entity_id"],
                "surface_text": profile["surface_text"],
                "attempt": attempt,
                "messages": _message_payload(rendered.messages),
                "output_schema": rendered.output_schema,
                "idempotency_key": request_key,
            },
            "outputs": {"response": result, "prompt_manifest_sha256": rendered.manifest_sha256},
        }

    def _review_hierarchy(
        self,
        *,
        source: SourceInput,
        spans: Sequence[EvidenceSpan],
        overview: Mapping[str, Any],
        types: dict[str, TypeDefinition],
        assignments: list[TypeAssignment],
    ) -> tuple[dict[str, TypeDefinition], list[TypeAssignment], list[NliResult], list[dict[str, Any]]]:
        if not self.invoke_model_nodes or len(types) <= 2:
            return types, assignments, [], [
                {
                    "event": "node_completed",
                    "node": "registry_consistency_review",
                    "inputs": {"mode": "skipped", "type_count": len(types)},
                    "outputs": {"proposals": []},
                }
            ]
        rendered = self.prompts.render(
            "registry_consistency_review",
            {
                "source_id": source.source_id,
                "policy_json": canonical_json(
                    {"nli_required": True, "allow_multiple_parents": True, "flat_registry_allowed": True}
                ),
                "overview_json": canonical_json(overview),
                "registry_json": canonical_json(
                    [item.model_dump(mode="json") for item in sorted(types.values(), key=lambda value: value.type_id)]
                ),
                "assignment_examples_json": canonical_json(
                    [item.model_dump(mode="json") for item in assignments]
                ),
                "source_evidence_json": canonical_json([item.model_dump(mode="json") for item in spans]),
            },
        )
        request_key = stable_id(
            "request",
            {"source": source.source_id, "operation": "registry_consistency_review", "prompt": rendered.manifest_sha256},
        )
        response = dict(
            self.model.invoke(
                operation="registry_consistency_review",
                messages=rendered.messages,
                output_schema=rendered.output_schema,
                idempotency_key=request_key,
            )
        )
        Draft202012Validator(rendered.output_schema).validate(response)
        events: list[dict[str, Any]] = [
            {
                "event": "node_completed",
                "node": "registry_consistency_review",
                "inputs": {
                    "messages": _message_payload(rendered.messages),
                    "output_schema": rendered.output_schema,
                    "idempotency_key": request_key,
                },
                "outputs": {"response": response, "prompt_manifest_sha256": rendered.manifest_sha256},
            }
        ]
        spans_by_id = _span_map(spans)
        nli_results: list[NliResult] = []
        aliases: dict[str, str] = {}

        def resolve(type_id: str) -> str:
            while type_id in aliases:
                type_id = aliases[type_id]
            return type_id

        for proposal in response["proposals"]:
            action = proposal["action"]
            first = resolve(str(proposal["first_type_id"]))
            second = resolve(str(proposal["second_type_id"]))
            if action == "keep_separate":
                continue
            if first not in types or second not in types or first == second or ROOT_TYPE_ID in {first, second}:
                continue
            hypothesis_results: list[NliResult] = []
            for index, hypothesis_record in enumerate(proposal["hypotheses"]):
                evidence_ids = tuple(
                    span_id for span_id in hypothesis_record["evidence_span_ids"] if span_id in spans_by_id
                )
                if not evidence_ids:
                    evidence_ids = tuple(
                        span_id for span_id in proposal["evidence_span_ids"] if span_id in spans_by_id
                    )
                if not evidence_ids:
                    evidence_ids = tuple(spans_by_id)[:1]
                hypothesis = str(hypothesis_record["text"])
                result = self.verify_nli(
                    hypothesis_kind=f"type_{action}",
                    premise=_premise(evidence_ids, spans_by_id),
                    hypothesis=hypothesis,
                    evidence_span_ids=evidence_ids,
                    request_key=stable_id(
                        "nli-request",
                        {"source": source.source_id, "proposal": proposal["proposal_id"], "index": index},
                    ),
                ).model_copy(
                    update={
                        "hypothesis_kind": f"type_{action}",
                        "hypothesis": hypothesis,
                        "subject_id": first,
                        "target_type_id": second,
                    }
                )
                hypothesis_results.append(result)
                nli_results.append(result)
            required = 1 if action == "child_of" else 2
            accepted = len(hypothesis_results) >= required and all(
                result.verdict == "entailed" for result in hypothesis_results[:required]
            )
            if accepted and action == "child_of":
                child = types[first]
                candidate = child.model_copy(update={"parent_type_ids": (second,)})
                candidate_types = {**types, first: candidate}
                if _acyclic(candidate_types):
                    types[first] = candidate
            elif accepted and action == "merge":
                candidate_types, candidate_assignments = _merge_types(
                    types, assignments, keep=first, remove=second
                )
                if _acyclic(candidate_types):
                    types, assignments = candidate_types, candidate_assignments
                    aliases[second] = first
                else:
                    accepted = False
            events.append(
                {
                    "event": "node_completed",
                    "node": "nli_verify_hierarchy",
                    "inputs": {"proposal": proposal},
                    "outputs": {
                        "accepted": accepted,
                        "results": [item.model_dump(mode="json") for item in hypothesis_results],
                    },
                }
            )
        return types, assignments, nli_results, events

    def type_answer(
        self,
        *,
        answer: AnswerInput,
    ) -> tuple[list[TypeAssignment], list[NliResult], list[dict[str, Any]]]:
        profiles = build_answer_profiles(answer)
        registry_types = {item.type_id: item for item in answer.registry.types}
        source_by_surface: dict[str, set[str]] = {}
        for assignment in answer.registry.assignments:
            source_by_surface.setdefault(normalize(assignment.surface_text), set()).update(assignment.type_ids)
        spans_by_id = _span_map(answer.registry.evidence_spans)
        source_evidence = [span.model_dump(mode="json") for span in answer.registry.evidence_spans]
        assignments: list[TypeAssignment] = []
        nli_results: list[NliResult] = []
        events: list[dict[str, Any]] = [
            {
                "event": "node_completed",
                "node": "build_answer_profiles",
                "inputs": {"response_id": answer.response_id},
                "outputs": {"profile_count": len(profiles), "profiles": profiles},
            }
        ]
        for profile in profiles:
            exact_ids = sorted(source_by_surface.get(normalize(profile["surface_text"]), ()))
            trusted_exact_surface = bool(exact_ids)
            if exact_ids:
                decision = {
                    "schema_version": "answer-typing-v2",
                    "entity_id": profile["entity_id"],
                    "selected_type_ids": exact_ids,
                    "hypotheses": [],
                    "reason": "Exact normalized surface reuses source entity types; every semantic type is still NLI-audited.",
                }
                events.append(
                    {
                        "event": "node_completed",
                        "node": "answer_typing",
                        "inputs": {"entity_id": profile["entity_id"], "mode": "exact-source-surface"},
                        "outputs": {"response": decision},
                    }
                )
            else:
                decision, event = self._answer_decision(answer, profile, registry_types, source_evidence)
                events.append(event)
            if decision["entity_id"] != profile["entity_id"]:
                # answer_typing types one entity per call; coerce a wrong/hallucinated
                # echoed entity_id to the known id instead of failing the record.
                decision["entity_id"] = profile["entity_id"]
            selected_ids = list(dict.fromkeys(str(item) for item in decision["selected_type_ids"]))
            unknown_selected = [type_id for type_id in selected_ids if type_id not in registry_types]
            selected_ids = [type_id for type_id in selected_ids if type_id in registry_types]
            if unknown_selected:
                events.append(
                    {
                        "event": "node_rejected",
                        "node": "validate_answer_type_decision",
                        "inputs": {"entity_id": profile["entity_id"]},
                        "outputs": {
                            "unknown_selected_type_ids": unknown_selected,
                            "fallback_type_id": ROOT_TYPE_ID,
                        },
                    }
                )
            hypotheses = {str(item["target_type_id"]): item for item in decision["hypotheses"]}
            accepted: list[str] = []
            entity_results: list[dict[str, Any]] = []
            for type_id in selected_ids or [ROOT_TYPE_ID]:
                type_definition = registry_types[type_id]
                hypothesis_record = hypotheses.get(type_id)
                evidence_ids = tuple(hypothesis_record["evidence_span_ids"]) if hypothesis_record else tuple(
                    span_id
                    for source_assignment in answer.registry.assignments
                    if normalize(source_assignment.surface_text) == normalize(profile["surface_text"])
                    for span_id in source_assignment.evidence_span_ids
                )
                evidence_ids = tuple(span_id for span_id in evidence_ids if span_id in spans_by_id)
                if not evidence_ids:
                    evidence_ids = tuple(spans_by_id)[: min(3, len(spans_by_id))]
                hypothesis = (
                    str(hypothesis_record["text"])
                    if hypothesis_record
                    else f"{profile['surface_text']} is a {type_definition.label}."
                )
                result = self.verify_nli(
                    hypothesis_kind="answer_type_assignment",
                    premise=_premise(evidence_ids, spans_by_id),
                    hypothesis=hypothesis,
                    evidence_span_ids=evidence_ids,
                    request_key=stable_id(
                        "nli-request",
                        {
                            "source": answer.source_id,
                            "response": answer.response_id,
                            "entity": profile["entity_id"],
                            "type": type_id,
                            "hypothesis": hypothesis,
                        },
                    ),
                ).model_copy(
                    update={
                        "hypothesis_kind": "answer_type_assignment",
                        "hypothesis": hypothesis,
                        "subject_id": profile["entity_id"],
                        "target_type_id": type_id,
                    }
                )
                nli_results.append(result)
                entity_results.append(result.model_dump(mode="json"))
                if trusted_exact_surface or type_id == ROOT_TYPE_ID or result.verdict == "entailed":
                    accepted.append(type_id)
            semantic_ids = tuple(sorted({type_id for type_id in accepted if type_id != ROOT_TYPE_ID}))
            final_ids = semantic_ids or (ROOT_TYPE_ID,)
            assignments.append(
                TypeAssignment(
                    node_id=profile["node_id"],
                    surface_text=profile["surface_text"],
                    graph_role="answer",
                    type_ids=final_ids,
                    status="assigned",
                    evidence_span_ids=(),
                    reason=str(decision["reason"]),
                )
            )
            events.append(
                {
                    "event": "node_completed",
                    "node": "nli_verify_answer",
                    "inputs": {"entity_id": profile["entity_id"], "surface_text": profile["surface_text"]},
                    "outputs": {"results": entity_results, "final_type_ids": list(final_ids)},
                }
            )
        return sorted(assignments, key=lambda item: item.node_id), nli_results, events

    def _answer_decision(
        self,
        answer: AnswerInput,
        profile: Mapping[str, Any],
        registry_types: Mapping[str, TypeDefinition],
        source_evidence: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.invoke_model_nodes:
            decision = {
                "schema_version": "answer-typing-v2",
                "entity_id": profile["entity_id"],
                "selected_type_ids": [ROOT_TYPE_ID],
                "hypotheses": [],
                "reason": "Offline structural fallback; no model was invoked.",
            }
            return decision, {
                "event": "node_completed",
                "node": "answer_typing",
                "inputs": {"entity_id": profile["entity_id"], "mode": "fake"},
                "outputs": {"response": decision},
            }
        rendered = self.prompts.render(
            "answer_typing",
            {
                "policy_json": canonical_json(
                    {
                        "registry_is_immutable": True,
                        "all_vertices_must_be_typed": True,
                        "nli_for_every_semantic_assignment": True,
                    }
                ),
                "answer_text": answer.response_raw,
                "answer_entity_profile_json": canonical_json(profile),
                "frozen_registry_json": canonical_json(
                    [item.model_dump(mode="json") for item in sorted(registry_types.values(), key=lambda value: value.type_id)]
                ),
                "source_evidence_json": canonical_json(source_evidence),
                "previous_attempt_json": "null",
            },
        )
        request_key = stable_id(
            "request",
            {
                "source": answer.source_id,
                "response": answer.response_id,
                "entity": profile["entity_id"],
                "prompt": rendered.manifest_sha256,
            },
        )
        result = dict(
            self.model.invoke(
                operation="answer_typing",
                messages=rendered.messages,
                output_schema=rendered.output_schema,
                idempotency_key=request_key,
            )
        )
        Draft202012Validator(rendered.output_schema).validate(result)
        return result, {
            "event": "node_completed",
            "node": "answer_typing",
            "inputs": {
                "entity_id": profile["entity_id"],
                "surface_text": profile["surface_text"],
                "messages": _message_payload(rendered.messages),
                "output_schema": rendered.output_schema,
                "idempotency_key": request_key,
            },
            "outputs": {"response": result, "prompt_manifest_sha256": rendered.manifest_sha256},
        }


def _acyclic(types: Mapping[str, TypeDefinition]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(type_id: str) -> bool:
        if type_id in visiting:
            return False
        if type_id in visited:
            return True
        visiting.add(type_id)
        if not all(visit(parent) for parent in types[type_id].parent_type_ids):
            return False
        visiting.remove(type_id)
        visited.add(type_id)
        return True

    return all(visit(type_id) for type_id in types)


def _merge_types(
    types: dict[str, TypeDefinition],
    assignments: list[TypeAssignment],
    *,
    keep: str,
    remove: str,
) -> tuple[dict[str, TypeDefinition], list[TypeAssignment]]:
    kept = types[keep]
    removed = types[remove]
    aliases = tuple(
        dict.fromkeys([*kept.aliases, removed.label, *removed.aliases])
    )
    parents = tuple(
        sorted(
            {
                ROOT_TYPE_ID if parent == remove else parent
                for parent in (*kept.parent_type_ids, *removed.parent_type_ids)
                if parent not in {keep, remove}
            }
        )
    ) or (ROOT_TYPE_ID,)
    types = {
        type_id: definition.model_copy(
            update={
                "parent_type_ids": tuple(
                    sorted(
                        {
                            keep if parent == remove else parent
                            for parent in definition.parent_type_ids
                            if (keep if parent == remove else parent) != type_id
                        }
                    )
                )
            }
        )
        for type_id, definition in types.items()
        if type_id != remove
    }
    types[keep] = kept.model_copy(
        update={
            "aliases": aliases,
            "parent_type_ids": parents,
            "evidence_span_ids": tuple(dict.fromkeys((*kept.evidence_span_ids, *removed.evidence_span_ids))),
        }
    )
    assignments = [
        item.model_copy(
            update={"type_ids": tuple(sorted({keep if type_id == remove else type_id for type_id in item.type_ids}))}
        )
        for item in assignments
    ]
    return types, assignments
