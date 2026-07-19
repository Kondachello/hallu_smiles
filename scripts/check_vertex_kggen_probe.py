#!/usr/bin/env python3
"""Exercise real KGGen extraction and clustering through the Vertex gateway."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.extract import KGExtractor, UsageLogger


PROBE_TEXT = (
    "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine. "
    "Marie Curie was born in Warsaw."
)


def _has_endpoints(relations, left: str, right: str) -> bool:
    left, right = left.casefold(), right.casefold()
    return any(
        len(relation) == 3
        and ((left in str(relation[0]).casefold() and right in str(relation[2]).casefold())
             or (left in str(relation[2]).casefold() and right in str(relation[0]).casefold()))
        for relation in relations
    )


def _has_ada_lovelace_anchor(relations) -> bool:
    """Accept either faithful KGGen encoding of the synthetic source fact.

    The source says that Lovelace wrote notes about Babbage's Analytical
    Engine.  A graph may represent that as a direct Lovelace--Engine edge or
    as Lovelace--Babbage plus Babbage--Engine; both preserve the source fact.
    """
    return (
        _has_endpoints(relations, "Ada Lovelace", "Analytical Engine")
        or _has_endpoints(relations, "Ada Lovelace", "Charles Babbage")
    )


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _structural_failure(graph) -> str | None:
    """Check the compatibility contract without grading a model's semantics.

    This is a transport/schema/clustering probe, not a measurement of Gemini's
    factual extraction quality.  A graph with valid endpoints and multiple
    relations proves the full KGGen path ran; a particular decomposition of a
    possessive phrase does not.
    """
    if len(graph.entities) < 4 or len(graph.relations) < 2:
        return "synthetic KGGen probe produced too few entities or relations"
    malformed = [relation for relation in graph.relations if len(relation) != 3]
    if malformed:
        return "synthetic KGGen probe produced a malformed relation"
    dangling = [
        relation
        for relation in graph.relations
        if relation[0] not in graph.entities or relation[2] not in graph.entities
    ]
    if dangling:
        return "synthetic KGGen probe produced a relation with a missing endpoint"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    usage = UsageLogger(None)
    graph = KGExtractor(cfg, usage=usage).extract(PROBE_TEXT, kind="vertex_probe")
    anchors = {
        "ada_lovelace": _has_ada_lovelace_anchor(graph.relations),
        "marie_curie_warsaw": _has_endpoints(graph.relations, "Marie Curie", "Warsaw"),
    }
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": cfg.llm.model,
        "api_base": cfg.llm.api_base,
        "structured_output_transport": cfg.llm.structured_output_transport,
        "structured_output_backend": cfg.llm.structured_output_backend,
        "cluster_requested": bool(cfg.extraction.cluster),
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "semantic_anchors": anchors,
        # This graph is deliberately synthetic, never C/Q/A data.  Keeping it
        # makes a remote probe failure diagnosable without logging prompts.
        "synthetic_graph": graph.to_dict(),
        "usage": usage.summary(),
    }
    failure = _structural_failure(graph)
    report["status"] = "failed" if failure else "ready"
    if failure:
        report["error"] = failure
    _write_report(Path(args.report), report)
    # The graph is synthetic and contains no user prompt.  Print it on both
    # success and failure so DataSphere's log tail remains useful even when a
    # failed Job cannot expose its declared artifact archive.
    print(json.dumps(report, sort_keys=True))
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
