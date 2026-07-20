#!/usr/bin/env python3
"""Validate every selected answer against support-critical fallback invariants.

This preflight is deliberately offline. It catches boundary/offset/cache-key
regressions on all fixed QA records before any paid Vertex request is made.
Provider protocol failures themselves are handled at runtime by the
deterministic sentence fallback and separately covered by offline unit tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.critical import _answer_chunks, _sentence_fallback_claims
from src.data import load_instances
from src.sampling import load_manifest_instances


def _check_answer(response: str, *, chunk_chars: int, min_chunk_chars: int) -> tuple[int, int]:
    chunks = _answer_chunks(response, chunk_chars)
    if "".join(chunk.text for chunk in chunks) != response:
        raise RuntimeError("critical answer chunks do not reconstruct the answer exactly")
    if any(
        chunk.start < 0 or chunk.end < chunk.start or response[chunk.start:chunk.end] != chunk.text
        for chunk in chunks
    ):
        raise RuntimeError("critical answer chunk has invalid absolute offsets")
    if any(len(chunk.text) > chunk_chars for chunk in chunks):
        raise RuntimeError("critical answer chunk exceeds configured ceiling")
    # The fallback is the last line of defence for malformed JSON, output
    # truncation at the minimum chunk size, or a non-stopping completion.
    fallback = _sentence_fallback_claims(response, "preflight_fallback_sentence")
    if response.strip() and not fallback:
        raise RuntimeError("non-empty answer has no deterministic fallback candidate")
    if any(
        response[claim.start:claim.end] != claim.text
        or not (0 <= claim.start < claim.end <= len(response))
        for claim in fallback
    ):
        raise RuntimeError("critical fallback claim has invalid exact offsets")
    if min_chunk_chars <= 0 or min_chunk_chars > chunk_chars:
        raise RuntimeError("critical chunk safety floor is outside its configured range")
    return len(chunks), len(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qa-manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    instances = load_manifest_instances(args.qa_manifest, load_instances(args.data_dir))
    sections = (cfg.support_critical.claim_extractor, cfg.support_critical.coverage_reviewer)
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": len(instances),
        "components": {},
    }
    for section, name in zip(sections, ("claim_extractor", "coverage_reviewer"), strict=True):
        chunk_chars = int(section.chunk_chars)
        min_chunk_chars = int(section.min_chunk_chars)
        chunk_counts: list[int] = []
        fallback_counts: list[int] = []
        for inst in instances:
            chunks, fallback = _check_answer(
                inst.response, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars
            )
            chunk_counts.append(chunks)
            fallback_counts.append(fallback)
        report["components"][name] = {
            "chunk_chars": chunk_chars,
            "min_chunk_chars": min_chunk_chars,
            "max_chunks_per_answer": max(chunk_counts, default=0),
            "max_fallback_candidates_per_answer": max(fallback_counts, default=0),
        }
    report["status"] = "ready"
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
