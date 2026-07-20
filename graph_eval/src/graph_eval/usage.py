"""Per-prediction resource accounting (calls, cache hits, latency, tokens)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Usage:
    extractor_calls: int = 0
    nli_calls: int = 0
    extraction_cache_hits: int = 0
    nli_cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_ms_extract: float = 0.0
    wall_time_ms_verify: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
