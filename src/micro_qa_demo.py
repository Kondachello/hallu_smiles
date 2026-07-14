"""One-record KGGen extraction demo with Obsidian-friendly graph artifacts.

This module deliberately stops after extraction.  It is intended to make the
``C -> G_C``, ``Q -> G_Q`` and ``A -> G_A`` steps inspectable without pulling
in the SentenceTransformer scoring stack.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .data import Instance, load_instances
from .extract import Graph, KGExtractor, UsageLogger


GRAPH_ORDER = ("G_C", "G_Q", "G_A", "G_ref")


def _nodes(graph: Graph) -> list[str]:
    """Return graph nodes, defensively including relation endpoints."""
    nodes = set(graph.entities)
    for subject, _, object_ in graph.relations:
        nodes.add(subject)
        nodes.add(object_)
    return sorted(nodes)


def graph_stats(graph: Graph) -> dict[str, float | int | None]:
    """Describe a simple directed graph without treating self-loops as density."""
    nodes = _nodes(graph)
    non_self_edges = sum(subject != object_ for subject, _, object_ in graph.relations)
    self_loops = len(graph.relations) - non_self_edges
    n_nodes = len(nodes)
    max_edges = n_nodes * (n_nodes - 1)
    return {
        "nodes": n_nodes,
        "edges": len(graph.relations),
        "self_loops": self_loops,
        "average_out_degree": round(len(graph.relations) / n_nodes, 4) if n_nodes else None,
        "directed_density": round(non_self_edges / max_edges, 6) if max_edges else None,
    }


def select_qa_instance(instances: Iterable[Instance], max_context_chars: int) -> Instance:
    """Choose one compact, non-trivial QA record deterministically.

    The target lengths favour a useful visual graph while staying below the
    chunking threshold.  Source and response IDs make ties stable across runs.
    """
    candidates = [
        inst
        for inst in instances
        if inst.task == "QA"
        and inst.context.strip()
        and (inst.query or "").strip()
        and inst.response.strip()
        and len(inst.context) <= max_context_chars
    ]
    if not candidates:
        raise ValueError(
            "No QA record with non-empty C/Q/A fits "
            f"--max-context-chars={max_context_chars}."
        )

    def preference(inst: Instance) -> tuple[int, str, str]:
        # Avoid a one-sentence toy record, but keep the source compact.
        score = abs(len(inst.context) - 1600) + abs(len(inst.response) - 600)
        return score, inst.source_id, inst.response_id

    return min(candidates, key=preference)


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _mermaid_text(value: str) -> str:
    """Escape user/model text used inside a Mermaid label."""
    escaped = html.escape(_one_line(value), quote=True)
    return (
        escaped.replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("|", "&#124;")
    )


def mermaid_graph(graph_name: str, graph: Graph) -> str:
    """Render one graph as a portable Mermaid ``flowchart LR`` block body."""
    nodes = _nodes(graph)
    if not nodes:
        return "flowchart LR\n  empty[\"No entities or relations extracted\"]"

    node_ids = {node: f"n{index}" for index, node in enumerate(nodes)}
    lines = ["flowchart LR"]
    for node in nodes:
        lines.append(f'  {node_ids[node]}["{_mermaid_text(node)}"]')
    for subject, relation, object_ in sorted(graph.relations):
        lines.append(
            f"  {node_ids[subject]} -->|{_mermaid_text(relation)}| {node_ids[object_]}"
        )
    return "\n".join(lines)


def _table_cell(value: Any) -> str:
    return _one_line(str(value)).replace("|", "\\|")


def _format_density(value: float | int | None) -> str:
    return "n/a (V < 2)" if value is None else f"{value:.6f}"


def _entity_stem(entity: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", entity.casefold()).strip("-") or "entity"
    slug = slug[:56]
    suffix = hashlib.sha1(entity.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{suffix}"


def _write_entity_pages(output_dir: Path, graphs: dict[str, Graph]) -> dict[str, str]:
    """Write an Obsidian Graph View-compatible page for every graph node."""
    entities_dir = output_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    all_nodes = sorted({node for graph in graphs.values() for node in _nodes(graph)})
    stems = {node: _entity_stem(node) for node in all_nodes}
    outgoing: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for graph_name, graph in graphs.items():
        for subject, relation, object_ in sorted(graph.relations):
            outgoing[subject].append((graph_name, relation, object_))

    for entity in all_nodes:
        lines = [f"# {_one_line(entity)}", "", "## Outgoing relations", ""]
        relations = outgoing.get(entity, [])
        if relations:
            for graph_name, relation, object_ in relations:
                target = f"[[entities/{stems[object_]}|{_one_line(object_)}]]"
                lines.append(
                    f"- **{graph_name}** · `{_table_cell(relation)}` → {target}"
                )
        else:
            lines.append("- No outgoing relations extracted.")
        lines.extend(["", "## Metadata", "", f"- Exact entity string: `{_table_cell(entity)}`", ""])
        (entities_dir / f"{stems[entity]}.md").write_text("\n".join(lines), encoding="utf-8")

    index = ["# Entity index", ""]
    for entity in all_nodes:
        index.append(f"- [[entities/{stems[entity]}|{_one_line(entity)}]]")
    (output_dir / "entity_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return stems


def write_obsidian_artifacts(
    output_dir: Path,
    graphs: dict[str, Graph],
    stats: dict[str, dict[str, float | int | None]],
    metadata: dict[str, Any],
) -> None:
    """Write a Mermaid overview plus entity pages that can be opened as a vault."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_entity_pages(output_dir, graphs)

    selected = metadata["selected_instance"]
    lengths = metadata["input_lengths_chars"]
    lines = [
        "# KGGen micro demo — QA extraction",
        "",
        "This vault contains extraction only: no embedding matching, scoring, or audit.",
        "",
        "## Run metadata",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Source ID: `{selected['source_id']}`",
        f"- Response ID: `{selected['response_id']}`",
        f"- Split / human label: `{selected['split']}` / `{selected['label']}`",
        f"- Input lengths (characters): C={lengths['context']}, Q={lengths['query']}, A={lengths['response']}",
        "- Full C/Q/A text and machine-readable graph data: `graphs.json`.",
        "- [[entity_index|Open the entity index]] or open this directory as an Obsidian vault for Graph View.",
        "",
        "## Graph statistics",
        "",
        "Directed density excludes self-loops and is `E / (V × (V − 1))`.",
        "",
        "| Graph | V | E | Self-loops | Avg. out-degree | Directed density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for graph_name in GRAPH_ORDER:
        stat = stats[graph_name]
        avg = stat["average_out_degree"]
        avg_text = "n/a" if avg is None else f"{avg:.4f}"
        lines.append(
            f"| {graph_name} | {stat['nodes']} | {stat['edges']} | {stat['self_loops']} | "
            f"{avg_text} | {_format_density(stat['directed_density'])} |"
        )

    for graph_name in GRAPH_ORDER:
        graph = graphs[graph_name]
        lines.extend([
            "",
            f"## {graph_name}",
            "",
            "```mermaid",
            mermaid_graph(graph_name, graph),
            "```",
            "",
            "| Subject | Relation | Object |",
            "|---|---|---|",
        ])
        if graph.relations:
            for subject, relation, object_ in sorted(graph.relations):
                lines.append(
                    f"| {_table_cell(subject)} | {_table_cell(relation)} | {_table_cell(object_)} |"
                )
        else:
            lines.append("| — | — | — |")

    (output_dir / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_one(
    extractor: KGExtractor,
    usage: UsageLogger,
    label: str,
    text: str,
    kind: str,
) -> tuple[Graph, float]:
    print(f"[extract] {label}: {len(text)} chars — starting", flush=True)
    calls_before = usage.calls
    start = time.perf_counter()
    graph = extractor.extract(text, kind=kind)
    elapsed = time.perf_counter() - start
    cache_state = "cache hit" if usage.calls == calls_before else "live model call"
    stats = graph_stats(graph)
    print(
        f"[extract] {label}: {cache_state}; {elapsed:.2f}s; "
        f"V={stats['nodes']} E={stats['edges']} density={_format_density(stats['directed_density'])}",
        flush=True,
    )
    return graph, elapsed


def run_micro_demo(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    max_context_chars: int = 3000,
    response_id: str | None = None,
) -> dict[str, Any]:
    """Run the one-record extraction demo and return its serializable payload."""
    cfg = load_config(config_path)
    model = str(cfg.llm.model)
    if "PLACEHOLDER" in model:
        raise ValueError(
            "llm.model is still PLACEHOLDER. Pass a demo YAML config with a concrete model."
        )
    key_env = getattr(cfg.llm, "api_key_env", None)
    if model.startswith("openrouter/") and (not key_env or not os.environ.get(key_env)):
        raise RuntimeError(
            f"{key_env or 'OPENROUTER_API_KEY'} is not set. Export it only in this terminal session."
        )

    instances = load_instances(data_dir)
    if response_id:
        selected = next((inst for inst in instances if inst.response_id == response_id), None)
        if selected is None:
            raise ValueError(f"No RAGTruth response found with id={response_id!r}.")
        if selected.task != "QA":
            raise ValueError(f"response_id={response_id!r} is task={selected.task!r}, not QA.")
        if len(selected.context) > max_context_chars:
            raise ValueError(
                f"response_id={response_id!r} has a {len(selected.context)}-character context, "
                f"above --max-context-chars={max_context_chars}."
            )
    else:
        selected = select_qa_instance(instances, max_context_chars)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    usage = UsageLogger(out / "usage.jsonl")
    extractor = KGExtractor(cfg, usage=usage)
    print(
        f"[demo] model={model}; source_id={selected.source_id}; "
        f"response_id={selected.response_id}; task=QA",
        flush=True,
    )

    gc, gc_seconds = _extract_one(extractor, usage, "G_C (retrieved context)", selected.context, "context")
    gq, gq_seconds = _extract_one(extractor, usage, "G_Q (user query)", selected.query or "", "query")
    ga, ga_seconds = _extract_one(extractor, usage, "G_A (model response)", selected.response, "response")
    gref = gc.union(gq)

    graphs = {"G_C": gc, "G_Q": gq, "G_A": ga, "G_ref": gref}
    stats = {name: graph_stats(graph) for name, graph in graphs.items()}
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "selected_instance": {
            "source_id": selected.source_id,
            "response_id": selected.response_id,
            "task": selected.task,
            "split": selected.split,
            "label": selected.y,
        },
        "input_lengths_chars": {
            "context": len(selected.context),
            "query": len(selected.query or ""),
            "response": len(selected.response),
        },
        "timings_seconds": {
            "G_C": round(gc_seconds, 4),
            "G_Q": round(gq_seconds, 4),
            "G_A": round(ga_seconds, 4),
        },
    }
    payload = {
        **metadata,
        "inputs": {
            "context": selected.context,
            "query": selected.query or "",
            "response": selected.response,
        },
        "graphs": {name: graph.to_dict() for name, graph in graphs.items()},
        "statistics": stats,
        "usage": usage.summary(),
    }
    (out / "graphs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_obsidian_artifacts(out, graphs, stats, metadata)
    print(f"[done] artifacts: {out.resolve()}", flush=True)
    return payload
