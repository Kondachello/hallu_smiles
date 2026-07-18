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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    usage = UsageLogger(None)
    graph = KGExtractor(cfg, usage=usage).extract(PROBE_TEXT, kind="vertex_probe")
    if len(graph.entities) < 4 or len(graph.relations) < 2:
        raise RuntimeError("synthetic KGGen probe produced too few entities or relations")
    if not _has_endpoints(graph.relations, "Ada Lovelace", "Analytical Engine"):
        raise RuntimeError("synthetic KGGen probe omitted the Ada Lovelace fact")
    if not _has_endpoints(graph.relations, "Marie Curie", "Warsaw"):
        raise RuntimeError("synthetic KGGen probe omitted the Marie Curie fact")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": cfg.llm.model,
        "api_base": cfg.llm.api_base,
        "structured_output_transport": cfg.llm.structured_output_transport,
        "structured_output_backend": cfg.llm.structured_output_backend,
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "usage": usage.summary(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
