"""Production typer: assign types to a record's graph vertices via the agent.

Wraps :class:`hallugraph_dynamic_typing.DynamicTypingAgent`. The agent is built
once (loads HHEM + gateway client) and reused for every record. Per record it

  1. builds a frozen source registry from the context+query graphs (typing every
     reference vertex, verified by HHEM NLI against the raw context text), then
  2. annotates the response graph against that frozen registry.

It returns ``(ref_vertex_types, answer_vertex_types)`` -- each a mapping of vertex
surface -> assigned type labels -- exactly the shape :class:`TypedVertexDetector`
needs for the type-aware EG metric. Typing does make gateway LLM calls (this is
the new, permitted cost of the run); the HalluGraph/GraphEval graphs are still
read cache-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# The agent package lives under dynamic_typing_agent/src, which is not on the
# main repo import path; add it once so ``hallugraph_dynamic_typing`` imports.
_AGENT_SRC = Path(__file__).resolve().parent.parent / "dynamic_typing_agent" / "src"


def ensure_agent_importable() -> None:
    if _AGENT_SRC.is_dir() and str(_AGENT_SRC) not in sys.path:
        sys.path.insert(0, str(_AGENT_SRC))


def _clean_relations(relations: Sequence[Sequence[str]]) -> tuple[tuple[str, str, str], ...]:
    """Keep only well-formed non-empty triples (agent GraphInput rejects the rest)."""
    out: list[tuple[str, str, str]] = []
    for row in relations:
        if len(row) == 3 and all(str(x).strip() for x in row):
            out.append((str(row[0]), str(row[1]), str(row[2])))
    return tuple(out)


def _entities(entities: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for e in entities:
        s = str(e).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


class AgentTyper:
    """Callable that types one record's graphs with the dynamic typing agent."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        cache_root: str | Path,
        artifacts_root: str | Path,
    ) -> "AgentTyper":
        ensure_agent_importable()
        from hallugraph_dynamic_typing.agent import DynamicTypingAgent

        # DynamicTypingAgent parses its YAML directly (model/nli/persistence
        # sections + env-var resolution), so hand it the config path rather than
        # a pre-loaded mapping.
        agent = DynamicTypingAgent.from_yaml(
            config_path, cache_root=cache_root, artifacts_root=artifacts_root
        )
        return cls(agent)

    def _labels_by_surface(self, assignments, types) -> dict[str, list[str]]:
        label_of = {t.type_id: t.label for t in types}
        out: dict[str, list[str]] = {}
        for a in assignments:
            labels = [label_of[tid] for tid in a.type_ids if tid in label_of]
            if labels:
                out.setdefault(a.surface_text, [])
                for lbl in labels:
                    if lbl not in out[a.surface_text]:
                        out[a.surface_text].append(lbl)
        return out

    def __call__(
        self,
        *,
        context_raw: str,
        query_raw: str,
        response_raw: str,
        context_entities: Sequence[str],
        context_relations: Sequence[tuple[str, str, str]],
        query_entities: Sequence[str],
        query_relations: Sequence[tuple[str, str, str]],
        response_entities: Sequence[str],
        response_relations: Sequence[tuple[str, str, str]],
        record_id: str = "typed",
    ) -> tuple[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]]]:
        ensure_agent_importable()
        from hallugraph_dynamic_typing.models import (
            AnswerInput,
            GraphInput,
            SourceInput,
        )

        ctx_ents = _entities(context_entities)
        if not ctx_ents:
            raise ValueError("context graph has no entities; cannot build source registry")
        qry_ents = _entities(query_entities)
        # GraphInput forbids empty entities; an empty query graph gets a harmless
        # placeholder vertex (its type cannot match any answer vertex).
        if not qry_ents:
            qry_ents = ("__no_query__",)
            query_relations = ()

        source = SourceInput(
            source_id=str(record_id),
            context_raw=context_raw or " ",
            query_raw=query_raw or "",
            context_graph=GraphInput(
                graph_id=f"{record_id}:context",
                role="context", entities=ctx_ents, relations=_clean_relations(context_relations)
            ),
            query_graph=GraphInput(
                graph_id=f"{record_id}:query",
                role="query", entities=qry_ents, relations=_clean_relations(query_relations)
            ),
        )
        source_run = self.agent.build_source_registry(source)
        if source_run.registry is None:
            raise RuntimeError(f"source typing failed: {source_run.failure}")
        registry = source_run.registry
        ref_vertex_types = self._labels_by_surface(registry.assignments, registry.types)

        answer_vertex_types: dict[str, list[str]] = {}
        resp_ents = _entities(response_entities)
        if resp_ents:
            answer = AnswerInput(
                source_id=str(record_id),
                response_id=str(record_id),
                response_raw=response_raw or " ",
                answer_graph=GraphInput(
                    graph_id=f"{record_id}:answer",
                    role="answer", entities=resp_ents, relations=_clean_relations(response_relations)
                ),
                registry=registry,
            )
            answer_run = self.agent.annotate_answer(answer)
            if answer_run.annotations is None:
                raise RuntimeError(f"answer typing failed: {answer_run.failure}")
            answer_vertex_types = self._labels_by_surface(
                answer_run.annotations.answer_assignments, registry.types
            )
        return ref_vertex_types, answer_vertex_types
