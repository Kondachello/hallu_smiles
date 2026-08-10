#!/usr/bin/env python3
"""Derive cache-compatible 300/500-QA manifests from the attested R12 set.

The resulting samples are nested, balanced within the original train/test
splits, and contain the exact R12 response IDs.  That last property makes them
eligible for a strict zero-inference replay against the R12 cache lineage.
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
SEED = 42
QUARANTINED_SOURCE_ID = "12448"
SIZES = {
    300: {("train", 0): 120, ("train", 1): 120, ("test", 0): 30, ("test", 1): 30},
    500: {("train", 0): 200, ("train", 1): 200, ("test", 0): 50, ("test", 1): 50},
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(record: dict[str, Any]) -> str:
    payload = "\x00".join(
        [
            "support-critical-r12-nested-train-size-v1",
            str(SEED),
            str(record["split"]),
            str(int(record["y"])),
            str(record["source_id"]),
            str(record["response_id"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_parent(parent: dict[str, Any]) -> list[dict[str, Any]]:
    if parent.get("version") != 1 or parent.get("task") != "QA":
        raise ValueError("parent is not a v1 QA manifest")
    if parent.get("quotas") != PARENT_QUOTAS:
        raise ValueError(f"parent quotas are not the attested R12 shape: {parent.get('quotas')!r}")
    records = parent.get("records")
    if not isinstance(records, list) or len(records) != PARENT_TOTAL:
        raise ValueError("parent does not contain the attested 750 QA records")
    actual = Counter((str(row.get("split")), int(row.get("y"))) for row in records)
    expected = Counter({("train", 0): 300, ("train", 1): 300, ("test", 0): 75, ("test", 1): 75})
    if actual != expected:
        raise ValueError(f"parent split/label quotas mismatch: {dict(actual)}")
    if len({str(row.get("source_id")) for row in records}) != PARENT_TOTAL:
        raise ValueError("parent contains duplicate source IDs")
    quarantine = [row for row in records if str(row.get("source_id")) == QUARANTINED_SOURCE_ID]
    if quarantine != [{
        "source_id": QUARANTINED_SOURCE_ID,
        "response_id": "17712",
        "split": "train",
        "y": 0,
        "gen_model": "gpt-4-0613",
    }]:
        raise ValueError("parent no longer has the attested source-12448 quarantine record")
    return records


def _choose(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key in SIZES[500]:
        split, label = key
        buckets[key] = sorted(
            [row for row in records if row["split"] == split and int(row["y"]) == label],
            key=_stable_key,
        )

    chosen: dict[int, list[dict[str, Any]]] = {300: [], 500: []}
    for key, ranked in buckets.items():
        small = SIZES[300][key]
        large = SIZES[500][key]
        ordinary = [row for row in ranked if row["source_id"] != QUARANTINED_SOURCE_ID]
        selected_300 = ordinary[:small]
        if len(selected_300) != small:
            raise ValueError(f"not enough ordinary records for {key}")
        selected_500 = list(selected_300)
        remaining = [row for row in ordinary if row not in selected_300]
        if key == ("train", 0):
            quarantined = [row for row in ranked if row["source_id"] == QUARANTINED_SOURCE_ID]
            if len(quarantined) != 1:
                raise ValueError("missing quarantined source in train/y=0 bucket")
            # R12 explicitly carried this source but excluded it from analysis.
            # Preserve that protocol at N=500 while keeping the N=300 set nested.
            selected_500.append(quarantined[0])
            remaining = remaining[: large - small - 1]
        else:
            remaining = remaining[: large - small]
        selected_500.extend(remaining)
        if len(selected_500) != large:
            raise ValueError(f"wrong {large}-QA quota for {key}")
        chosen[300].extend(selected_300)
        chosen[500].extend(selected_500)

    for size, rows in chosen.items():
        actual = Counter((row["split"], int(row["y"])) for row in rows)
        if actual != Counter(SIZES[size]):
            raise ValueError(f"wrong split/label quota for N={size}: {dict(actual)}")
        if len({row["source_id"] for row in rows}) != size:
            raise ValueError(f"duplicate source in N={size}")
        rows.sort(key=lambda row: (row["split"], row["source_id"], row["response_id"]))
    if not {row["response_id"] for row in chosen[300]}.issubset(
        {row["response_id"] for row in chosen[500]}
    ):
        raise ValueError("N=300 is not nested in N=500")
    if any(row["source_id"] == QUARANTINED_SOURCE_ID for row in chosen[300]):
        raise ValueError("the 300-QA manifest must not include source 12448")
    if not any(row["source_id"] == QUARANTINED_SOURCE_ID for row in chosen[500]):
        raise ValueError("the 500-QA manifest must include source 12448 for explicit quarantine")
    return chosen


def _manifest(size: int, rows: list[dict[str, Any]], parent_sha256: str) -> dict[str, Any]:
    train_sources = size * 4 // 5
    test_sources = size - train_sources
    return {
        "version": 1,
        "task": "QA",
        "seed": SEED,
        "quotas": {"train_sources": train_sources, "test_sources": test_sources},
        "records": rows,
        "derivation": {
            "protocol": "support-critical-r12-nested-train-size-v1",
            "parent_manifest_sha256": parent_sha256,
            "parent_qa_total": PARENT_TOTAL,
            "selection": "stable SHA-256 ranking within original split/label buckets",
            "nested_with": "support-critical-r12-nested-300.json" if size == 500 else None,
            "source_12448": "included and explicitly quarantined" if size == 500 else "not selected",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent_manifest)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    rows = _validate_parent(parent)
    chosen = _choose(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_sha256 = _sha256_bytes(parent_path)
    for size in sorted(chosen):
        output = output_dir / f"support-critical-r12-nested-{size}.json"
        output.write_text(
            json.dumps(_manifest(size, chosen[size], parent_sha256), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{output} { _sha256_bytes(output) }")


if __name__ == "__main__":
    main()
