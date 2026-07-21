"""GraphEval extractor adapter over an immutable shared KGGen graph.

The adapter intentionally depends on the neutral experiment-side provider rather
than making GraphEval import HalluGraph.  It serializes exactly the graph
relations HalluGraph receives, in deterministic order.
"""
from __future__ import annotations

import json
from typing import Any

from graph_eval.extraction.base import ExtractionOutput


class SharedKGGenGraphEvalExtractor:
    prompt_profile = "shared_kggen_response_graph_v1"

    def __init__(self, provider: Any):
        self.provider = provider
        self.last_artifact = None

    def extract(self, response_text: str) -> ExtractionOutput:
        artifact = self.provider.materialize(response_text, role="response")
        self.last_artifact = artifact
        triples = sorted([list(relation) for relation in artifact.graph.relations])
        return ExtractionOutput(
            raw_output=json.dumps(triples, ensure_ascii=False, separators=(",", ":")),
            usage={
                "extractor_calls": 0,
                "extraction_cache_hit": artifact.cache_hit,
                **artifact.reference(),
            },
        )

    def response_reference(self, item: Any) -> dict[str, str]:
        return self.provider.response_reference(item)


__all__ = ["SharedKGGenGraphEvalExtractor"]
