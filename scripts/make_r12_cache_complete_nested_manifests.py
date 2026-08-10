#!/usr/bin/env python3
"""Derive balanced, cache-complete N=300/N=500 subsets of the attested R12 set.

The only eligibility signal is a full zero-network cache preflight.  The
selection neither reads nor ranks held-out labels beyond retaining R12's
already-fixed split/label quotas.  This makes a cache-only recovery explicit
instead of silently treating cache misses as permissible live inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PARENT_TOTAL = 750
PARENT_QUOTAS = {"train_sources": 600, "test_sources": 150}
PARENT_SHA256 = "208dfc9e96c03039b5f8adeffe3e5174b6d51a835677ddc79ae4d2c40006f39c"
SEED = 42
QUARANTINED_SOURCE_ID = "12448"
PROTOCOL = "support-critical-r12-cache-complete-nested-train-size-v1"
SIZES = {
    300: {("train", 0): 120, ("train", 1): 120, ("test", 0): 30, ("test", 1): 30},
    500: {("train", 0): 200, ("train", 1): 200, ("test", 0): 50, ("test", 1): 50},
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(record: dict[str, Any]) -> str:
    # Keep the v1 ordering.  Cache eligibility is the only new filter, so any
    # replacement is a deterministic next element in the original ordering.
    payload = "\x00".join([
        "support-critical-r12-nested-train-size-v1", str(SEED),
        str(record["split"]), str(int(record["y"])),
        str(record["source_id"]), str(record["response_id"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_parent(path: Path) -> list[dict[str, Any]]:
    if _sha256_bytes(path) != PARENT_SHA256:
        raise ValueError("parent manifest SHA-256 is not the attested R12 value")
    parent = json.loads(path.read_text(encoding="utf-8"))
    records = parent.get("records")
    if parent.get("version") != 1 or parent.get("task") != "QA" or parent.get("quotas") != PARENT_QUOTAS:
        raise ValueError("parent is not the attested 750-QA R12 manifest")
    if not isinstance(records, list) or len(records) != PARENT_TOTAL:
        raise ValueError("parent does not contain 750 QA records")
    expected = Counter({("train", 0): 300, ("train", 1): 300, ("test", 0): 75, ("test", 1): 75})
    actual = Counter((str(row.get("split")), int(row.get("y"))) for row in records)
    if actual != expected or len({str(row.get("source_id")) for row in records}) != PARENT_TOTAL:
        raise ValueError("parent R12 split/label shape is invalid")
    quarantine = [row for row in records if str(row.get("source_id")) == QUARANTINED_SOURCE_ID]
    if quarantine != [{
        "source_id": QUARANTINED_SOURCE_ID,
        "response_id": "17712",
        "split": "train",
        "y": 0,
        "gen_model": "gpt-4-0613",
    }]:
        raise ValueError("parent no longer carries the attested source-12448 record")
    return records


def _load_ineligible(path: Path, records: list[dict[str, Any]]) -> tuple[set[str], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != "support-critical-cache-only-preflight-v1":
        raise ValueError("cache report has an unexpected protocol")
    if report.get("cache_only") is not True:
        raise ValueError("cache report is not an explicit cache-only preflight")
    missing = report.get("missing")
    if not isinstance(missing, list) or not missing:
        raise ValueError("cache report has no missing records to make eligibility explicit")
    valid_sources = {str(row["source_id"]) for row in records}
    ineligible: set[str] = set()
    for item in missing:
        if item.get("component") != "critical_claim_verifier":
            raise ValueError("unexpected cache-miss component")
        source_id = str(item.get("source_id", ""))
        if source_id not in valid_sources:
            raise ValueError("cache report names a source outside the parent manifest")
        ineligible.add(source_id)
    if QUARANTINED_SOURCE_ID in ineligible:
        raise ValueError("source 12448 must remain a quarantine, not a cache miss")
    return ineligible, _sha256_bytes(path)


def _choose(records: list[dict[str, Any]], ineligible: set[str]) -> dict[int, list[dict[str, Any]]]:
    chosen: dict[int, list[dict[str, Any]]] = {300: [], 500: []}
    for key, large in SIZES[500].items():
        split, label = key
        ranked = sorted(
            [row for row in records if row["split"] == split and int(row["y"]) == label],
            key=_stable_key,
        )
        ordinary = [
            row for row in ranked
            if row["source_id"] not in ineligible and row["source_id"] != QUARANTINED_SOURCE_ID
        ]
        small_n = SIZES[300][key]
        large_n = large - (1 if key == ("train", 0) else 0)
        if len(ordinary) < large_n:
            raise ValueError(f"not enough cache-complete records for {key}")
        chosen[300].extend(ordinary[:small_n])
        chosen[500].extend(ordinary[:large_n])
        if key == ("train", 0):
            chosen[500].extend(row for row in ranked if row["source_id"] == QUARANTINED_SOURCE_ID)

    for size, rows in chosen.items():
        if len(rows) != size or len({row["source_id"] for row in rows}) != size:
            raise ValueError(f"N={size} has duplicate or missing sources")
        if Counter((row["split"], int(row["y"])) for row in rows) != Counter(SIZES[size]):
            raise ValueError(f"N={size} does not retain the fixed split/label quotas")
        rows.sort(key=lambda row: (row["split"], row["source_id"], row["response_id"]))
    ids_300 = {(row["source_id"], row["response_id"]) for row in chosen[300]}
    ids_500 = {(row["source_id"], row["response_id"]) for row in chosen[500]}
    if not ids_300.issubset(ids_500):
        raise ValueError("N=300 is not nested in N=500")
    if any(row["source_id"] in ineligible for rows in chosen.values() for row in rows):
        raise ValueError("a cache-ineligible source entered a child manifest")
    if any(row["source_id"] == QUARANTINED_SOURCE_ID for row in chosen[300]):
        raise ValueError("N=300 must not include source 12448")
    if sum(row["source_id"] == QUARANTINED_SOURCE_ID for row in chosen[500]) != 1:
        raise ValueError("N=500 must retain exactly one quarantined source 12448")
    return chosen


def _manifest(
    size: int, rows: list[dict[str, Any]], *, report_sha256: str, ineligible: set[str]
) -> dict[str, Any]:
    train_sources = size * 4 // 5
    return {
        "version": 1,
        "task": "QA",
        "seed": SEED,
        "quotas": {"train_sources": train_sources, "test_sources": size - train_sources},
        "records": rows,
        "derivation": {
            "protocol": PROTOCOL,
            "parent_manifest_sha256": PARENT_SHA256,
            "parent_qa_total": PARENT_TOTAL,
            "selection": "v1 stable SHA-256 ranking within original split/label buckets after cache-only eligibility filter",
            "cache_eligibility": {
                "criterion": "no missing support-critical cache-only preflight artifacts",
                "preflight_report_sha256": report_sha256,
                "ineligible_source_count": len(ineligible),
                "ineligible_source_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(ineligible)).encode("utf-8")
                ).hexdigest(),
            },
            "nested_with": "support-critical-r12-cache-complete-nested-300.json" if size == 500 else None,
            "source_12448": "included and explicitly quarantined" if size == 500 else "not selected",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--cache-preflight-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    parent = Path(args.parent_manifest)
    records = _load_parent(parent)
    ineligible, report_sha256 = _load_ineligible(Path(args.cache_preflight_report), records)
    selected = _choose(records, ineligible)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted(selected):
        output = output_dir / f"support-critical-r12-cache-complete-nested-{size}.json"
        output.write_text(
            json.dumps(_manifest(size, selected[size], report_sha256=report_sha256, ineligible=ineligible), indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"{output} {_sha256_bytes(output)}")


if __name__ == "__main__":
    main()
