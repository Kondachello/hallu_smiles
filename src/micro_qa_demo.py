"""One-record KGGen demo with Obsidian-friendly graph and audit artifacts.

The default is extraction-only, so ``C -> G_C``, ``Q -> G_Q`` and ``A -> G_A``
remain quick to inspect.  An explicit audit mode uses the project's normal
matching and audit modules for one illustrative record.
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

from .cache import config_value
from .config import load_config
from .data import Instance, load_instances
from .extract import Graph, KGExtractor, UsageLogger
from .audit import build_audit_record, write_audit
from .matching import Embedder, RefGraph, SBERTEmbedder
from .metrics import score_response


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

    The target lengths favour a compact but non-trivial visual graph. This is
    still a complete C/Q/A extraction-and-clustering pass, but it avoids a
    large source whose many entities make a free provider spend minutes in
    KGGen's repeated cluster validation calls. Source and response IDs make
    ties stable across runs.
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
        # A ~0.6k context plus a ~0.25k answer has enough relations to inspect
        # while keeping the complete cluster stage practical for a free LLM.
        score = abs(len(inst.context) - 600) + abs(len(inst.response) - 240)
        return score, inst.source_id, inst.response_id

    return min(candidates, key=preference)


def list_qa_candidates(
    instances: Iterable[Instance],
    min_context_chars: int = 0,
    min_query_chars: int = 0,
    min_response_chars: int = 0,
    limit: int = 20,
) -> list[Instance]:
    """Return QA records with substantial C/Q/A text, ranked for manual selection.

    Character length is only a pre-filter: the actual KG vertex count remains a
    property of the model's extraction and is printed by the live demo.
    """
    minimums = (min_context_chars, min_query_chars, min_response_chars)
    if any(value < 0 for value in minimums):
        raise ValueError("candidate minimum lengths must be non-negative.")
    if limit < 1:
        raise ValueError("candidate limit must be at least 1.")

    candidates = [
        inst
        for inst in instances
        if inst.task == "QA"
        and inst.context.strip()
        and (inst.query or "").strip()
        and inst.response.strip()
        and len(inst.context) >= min_context_chars
        and len(inst.query or "") >= min_query_chars
        and len(inst.response) >= min_response_chars
    ]

    # The QA questions are intrinsically much shorter than passages/answers,
    # so balance their lengths against reference scales rather than letting C
    # alone dominate the manually curated shortlist.
    def richness(inst: Instance) -> tuple[float, int, str, str]:
        balanced = min(
            len(inst.context) / 900,
            len(inst.query or "") / 100,
            len(inst.response) / 1000,
        )
        total = len(inst.context) + len(inst.query or "") + len(inst.response)
        return (-balanced, -total, inst.source_id, inst.response_id)

    return sorted(candidates, key=richness)[:limit]


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


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fenced_text(value: str) -> list[str]:
    """Render arbitrary input text inside a safe Markdown code fence."""
    fence = "```"
    while fence in value:
        fence += "`"
    return [f"{fence}text", value, fence]


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
    inputs: dict[str, str] | None = None,
) -> None:
    """Write a Mermaid overview plus entity pages that can be opened as a vault."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_entity_pages(output_dir, graphs)

    selected = metadata["selected_instance"]
    lengths = metadata["input_lengths_chars"]
    audit = metadata.get("audit")
    lines = [
        "# KGGen micro demo — QA extraction",
        "",
        (
            "This vault contains extraction plus a one-record structural audit."
            if audit
            else "This vault contains extraction only: no embedding matching, scoring, or audit."
        ),
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
    ]

    if inputs:
        lines.extend(["## Input text", ""])
        for name, label in (
            ("context", "Retrieved context (C)"),
            ("query", "User query (Q)"),
            ("response", "Model response (A)"),
        ):
            lines.extend([f"### {label}", "", *_fenced_text(inputs.get(name, "")), ""])

    if audit:
        lines.extend([
            "## One-record audit",
            "",
            f"- Illustrative α: `{audit['alpha']:.2f}` (not tuned on one record)",
            f"- EG: `{_format_metric(audit['EG'])}` · RP: `{_format_metric(audit['RP'])}` · "
            f"CFI: `{_format_metric(audit['CFI'])}` · H: `{_format_metric(audit['H'])}`",
            f"- Ungrounded entities: {len(audit['ungrounded_entities'])} · "
            f"Unsupported relations: {len(audit['unsupported_relations'])}",
            f"- Full audit record: `audit/{audit['response_id']}.json`.",
            "",
        ])

    lines.extend([
        "## Graph statistics",
        "",
        "Directed density excludes self-loops and is `E / (V × (V − 1))`.",
        "",
        "| Graph | V | E | Self-loops | Avg. out-degree | Directed density |",
        "|---|---:|---:|---:|---:|---:|",
    ])
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


def audit_micro_graphs(
    cfg: Any,
    instance: Instance,
    graphs: dict[str, Graph],
    alpha: float = 0.7,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Score micro graphs with the normal EG/RP and audit implementation.

    A one-record demo cannot tune alpha or derive a decision threshold, so its
    alpha is intentionally caller-supplied and only affects CFI/H.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"audit alpha must be in [0, 1], got {alpha!r}.")
    if embedder is None:
        matching = cfg.matching
        embedder = SBERTEmbedder(
            matching.embedding_model,
            model_revision=config_value(matching, "embedding_model_revision"),
            model_path=config_value(matching, "embedding_model_path"),
            device=str(config_value(matching, "embedding_device", "cpu")),
            local_files_only=bool(config_value(matching, "local_files_only", True)),
        )
    ref_graph = RefGraph(
        graphs["G_ref"].entities,
        graphs["G_ref"].relations,
        cfg.matching,
        embedder,
    )
    result = score_response(graphs["G_A"], ref_graph, graphs["G_C"], graphs["G_Q"])
    return build_audit_record(instance, result, alpha)


def run_micro_demo(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    max_context_chars: int = 3000,
    response_id: str | None = None,
    audit: bool = False,
    audit_alpha: float = 0.7,
) -> dict[str, Any]:
    """Run the one-record extraction demo and return its serializable payload."""
    cfg = load_config(config_path)
    model = str(cfg.llm.model)
    if "PLACEHOLDER" in model:
        raise ValueError(
            "llm.model is still PLACEHOLDER. Pass a demo YAML config with a concrete model."
        )
    key_env = getattr(cfg.llm, "api_key_env", None)
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError(
            f"{key_env or 'API key'} is not set. Export it only in this terminal session."
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
    if audit:
        print("[audit] local entity/relation matching — starting", flush=True)
        audit_record = audit_micro_graphs(cfg, selected, graphs, alpha=audit_alpha)
        audit_path = write_audit(audit_record, out / "audit")
        payload["audit"] = audit_record
        print(
            f"[audit] EG={_format_metric(audit_record['EG'])} "
            f"RP={_format_metric(audit_record['RP'])} H={_format_metric(audit_record['H'])}; "
            f"ungrounded={len(audit_record['ungrounded_entities'])} "
            f"unsupported={len(audit_record['unsupported_relations'])}; "
            f"{audit_path.resolve()}",
            flush=True,
        )
    (out / "graphs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_obsidian_artifacts(
        out,
        graphs,
        stats,
        {**metadata, "audit": payload.get("audit")},
        inputs=payload["inputs"],
    )
    print(f"[done] artifacts: {out.resolve()}", flush=True)
    return payload
