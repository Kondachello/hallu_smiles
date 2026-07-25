#!/usr/bin/env python3
"""Read-only diagnostic: does our cache_key computation match real cache filenames?

No gateway call, no LLM inference, no writes to the checkpoint. Computes the
same cache_key KGExtractor/GraphCacheSource would compute for every selected
historical QA record's response/context/query text, and checks how many of
those keys are already present as files in a given kg/ cache directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.extract import (
    CACHE_KEY_SCHEMA_CURRENT,
    CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY,
    KGExtractor,
)
from experiments.datasets.historical_qa import materialize_historical_qa_no_gold

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--kg-dir", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--qa-sample-size", type=int, default=750)
parser.add_argument("--qa-test-fraction", default="0.2")
parser.add_argument("--sample-seed", type=int, default=42)
args = parser.parse_args()

cfg = load_config(args.config)
extractor = KGExtractor(cfg, cache_only=True)
existing = {p.stem for p in Path(args.kg_dir).glob("*.json")}
records = materialize_historical_qa_no_gold(
    args.data_dir,
    qa_sample_size=args.qa_sample_size,
    qa_test_fraction=args.qa_test_fraction,
    sample_seed=args.sample_seed,
)
roles = {"response": "response_raw", "context": "context_raw", "query": "query_raw"}
current_hits = legacy_hits = total = 0
samples = []
for record in records:
    for role, field in roles.items():
        text = str(record.get(field) or "").strip()
        if not text:
            continue
        total += 1
        current_key = extractor.cache_key_for_schema(text, schema=CACHE_KEY_SCHEMA_CURRENT)
        legacy_key = extractor.cache_key_for_schema(text, schema=CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY)
        current_hit = current_key in existing
        legacy_hit = legacy_key in existing
        current_hits += current_hit
        legacy_hits += legacy_hit
        if len(samples) < 6:
            samples.append({
                "response_id": record.get("response_id"), "role": role, "text_len": len(text),
                "current_key": current_key, "current_hit": current_hit,
                "legacy_key": legacy_key, "legacy_hit": legacy_hit,
            })

report = {
    "record_count": len(records),
    "existing_kg_files": len(existing),
    "keys_probed": total,
    "current_schema_hits": current_hits,
    "legacy_schema_hits": legacy_hits,
    "cache_key_params": extractor._cache_key_params(),
    "existing_filename_sample": sorted(existing)[:5],
    "sample_probes": samples,
}
print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
