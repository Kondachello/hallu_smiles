#!/usr/bin/env python3
"""Probe the first deterministic QA reference graph, phase by phase.

The small synthetic KGGen probe proves the local vLLM + DSPy protocol.  It
cannot expose a pathological candidate list emitted for a real RAGTruth
context, though.  This program uses the exact first selected QA source, the
same generated runtime config and the same content-addressed KG cache as the
pilot.  A successful context graph is therefore reused by the strict run.

The implementation intentionally follows KGGen 0.4's no-chunk ``generate``
path one stage at a time: entities -> relations -> optional clustering.  The
separate progress lines are crucial when a bounded shell timeout is the only
safe way to interrupt an upstream client stall on a paid GPU VM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _emit(phase: str, **fields: Any) -> None:
    """Print a flush-safe structured progress record for DataSphere stdout."""
    print(json.dumps({"probe": "qa-reference", "phase": phase, **fields}, sort_keys=True), flush=True)


def run_probe(config_path: str | Path) -> dict[str, Any]:
    # Imported lazily so offline test/import paths stay dependency-light.
    import dspy
    from kg_gen.models import Graph as KGGenGraph
    from kg_gen.steps._1_get_entities import get_entities
    from kg_gen.steps._2_get_relations import get_relations

    from src.config import load_config
    from src.data import load_instances
    from src.extract import Graph, KGExtractor
    from src.sampling import select_qa_pilot

    cfg = load_config(config_path)
    selected = select_qa_pilot(load_instances(cfg.data.dir), seed=int(cfg.eval.seed))
    if not selected:
        raise RuntimeError("QA pilot selection unexpectedly returned no records")
    inst = selected[0]
    text = (inst.context or "").strip()
    if not text:
        raise RuntimeError(f"selected QA source {inst.source_id} has an empty context")
    if len(text) > int(cfg.extraction.context_chunk_chars):
        raise RuntimeError("reference probe is only valid for an unchunked selected context")

    extractor = KGExtractor(cfg)
    cache_key = extractor._cache_key(text)  # Same cache contract as the pilot.
    cached = extractor._load_cache(cache_key)
    if cached is not None:
        _emit(
            "cache_hit",
            source_id=inst.source_id,
            response_id=inst.response_id,
            context_chars=len(text),
            entities=len(cached.entities),
            relations=len(cached.relations),
        )
        return {
            "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "ready",
            "cached": True,
            "source_id": inst.source_id,
            "response_id": inst.response_id,
            "context_chars": len(text),
            "entities": len(cached.entities),
            "relations": len(cached.relations),
        }

    started = time.perf_counter()
    backend = extractor._get_backend()
    with dspy.context(lm=backend.lm):
        _emit(
            "entities:start",
            source_id=inst.source_id,
            response_id=inst.response_id,
            context_chars=len(text),
            max_tokens=extractor.max_tokens,
        )
        entities = get_entities(text)
        _emit("entities:done", count=len(entities), elapsed_seconds=round(time.perf_counter() - started, 3))

        _emit("relations:start", entity_count=len(entities))
        relations = get_relations(text, entities)
        _emit("relations:done", count=len(relations), elapsed_seconds=round(time.perf_counter() - started, 3))

    raw_graph = KGGenGraph(
        entities=set(entities),
        relations=set(relations),
        edges={relation[1] for relation in relations},
    )
    clustering_ran = extractor._should_cluster_backend_graph(raw_graph)
    if clustering_ran:
        _emit(
            "cluster:start",
            entities=len(raw_graph.entities),
            predicates=len(raw_graph.edges),
        )
        graph = backend.cluster(raw_graph)
        _emit(
            "cluster:done",
            entities=len(graph.entities),
            predicates=len(graph.edges),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    else:
        graph = raw_graph
        _emit(
            "cluster:skipped",
            entities=len(raw_graph.entities),
            predicates=len(raw_graph.edges),
            cluster_max_items=extractor.cluster_max_items,
        )

    extracted = Graph(
        entities={str(entity) for entity in getattr(graph, "entities", set())},
        relations={
            tuple(str(value) for value in relation)
            for relation in getattr(graph, "relations", set())
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        },
    )
    extractor._save_cache(cache_key, extracted)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "cached": False,
        "source_id": inst.source_id,
        "response_id": inst.response_id,
        "context_chars": len(text),
        "entities_before_cluster": len(raw_graph.entities),
        "predicates_before_cluster": len(raw_graph.edges),
        "entities": len(extracted.entities),
        "relations": len(extracted.relations),
        "clustering_ran": clustering_ran,
        "cluster_max_items": extractor.cluster_max_items,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = run_probe(args.config)
    except Exception as exc:  # noqa: BLE001
        failure = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "error",
            "error": repr(exc),
        }
        _atomic_json(report_path, failure)
        _emit("error", error=repr(exc))
        raise
    _atomic_json(report_path, report)
    _emit("ready", **report)


if __name__ == "__main__":
    main()
