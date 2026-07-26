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
parser.add_argument("--kg-dir", required=True, action="append",
                    help="a kg/ cache directory to probe; repeat to chain multiple read sources "
                         "(e.g. the 750 baseline kg plus the 100-QA lineage kg it reads through)")
parser.add_argument("--data-dir", required=True)
parser.add_argument("--qa-sample-size", type=int, default=750)
parser.add_argument("--qa-test-fraction", default="0.2")
parser.add_argument("--sample-seed", type=int, default=42)
args = parser.parse_args()

cfg = load_config(args.config)
extractor = KGExtractor(cfg, cache_only=True)
# One set of filename stems per probed kg directory, in the order given: the
# first directory that owns a key wins, exactly like the replay source chain.
per_dir = [(kg_dir, {p.stem for p in Path(kg_dir).glob("*.json")}) for kg_dir in args.kg_dir]
combined = set().union(*(files for _, files in per_dir)) if per_dir else set()
records = materialize_historical_qa_no_gold(
    args.data_dir,
    qa_sample_size=args.qa_sample_size,
    qa_test_fraction=args.qa_test_fraction,
    sample_seed=args.sample_seed,
)
roles = {"response": "response_raw", "context": "context_raw", "query": "query_raw"}
current_hits = legacy_hits = total = available = 0
per_dir_legacy_hits = {kg_dir: 0 for kg_dir, _ in per_dir}
samples = []
missing = []                       # (response_id, role, source_id) with no graph in any dir
incomplete_by_source: dict[str, set] = {}   # source_id -> set of response_ids still missing a role
complete_records = 0
for record in records:
    record_missing = False
    for role, field in roles.items():
        text = str(record.get(field) or "").strip()
        if not text:
            continue
        total += 1
        current_key = extractor.cache_key_for_schema(text, schema=CACHE_KEY_SCHEMA_CURRENT)
        legacy_key = extractor.cache_key_for_schema(text, schema=CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY)
        current_hit = current_key in combined
        legacy_hit = legacy_key in combined
        current_hits += current_hit
        legacy_hits += legacy_hit
        available += bool(current_hit or legacy_hit)
        for kg_dir, files in per_dir:
            if legacy_key in files or current_key in files:
                per_dir_legacy_hits[kg_dir] += 1
                break
        if not (current_hit or legacy_hit):
            record_missing = True
            src = str(record.get("source_id"))
            missing.append({"response_id": record.get("response_id"), "role": role, "source_id": src})
            incomplete_by_source.setdefault(src, set()).add(str(record.get("response_id")))
        if len(samples) < 6:
            samples.append({
                "response_id": record.get("response_id"), "role": role, "text_len": len(text),
                "current_key": current_key, "current_hit": current_hit,
                "legacy_key": legacy_key, "legacy_hit": legacy_hit,
            })
    if not record_missing:
        complete_records += 1

report = {
    "record_count": len(records),
    "complete_records": complete_records,
    "incomplete_records": len(records) - complete_records,
    "probed_kg_dirs": [
        {"kg_dir": kg_dir, "existing_kg_files": len(files), "records_or_roles_owned": per_dir_legacy_hits[kg_dir]}
        for kg_dir, files in per_dir
    ],
    "combined_existing_kg_files": len(combined),
    "keys_probed": total,
    "current_schema_hits": current_hits,
    "legacy_schema_hits": legacy_hits,
    "combined_available": available,
    "missing_count": len(missing),
    "missing_sources": sorted(incomplete_by_source),
    "missing": missing[:60],
    "cache_key_params": extractor._cache_key_params(),
    "combined_existing_filename_sample": sorted(combined)[:5],
    "sample_probes": samples,
}
print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
