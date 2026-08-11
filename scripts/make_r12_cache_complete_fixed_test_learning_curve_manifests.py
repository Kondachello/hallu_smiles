#!/usr/bin/env python3
"""Derive a fixed-test, cache-complete R12 support-critical learning curve.

The R12 cache inventory contains 26 missing critical-verdict entries and the
project protocol separately quarantines source 12448.  This script selects
only entries with an attested cache hit, keeps a single 146-response balanced
test set across every point, and creates nested balanced train sets of
80/240/400/564 responses.  It never uses a score, prediction, or held-out
metric to select a record.
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
PROTOCOL = "support-critical-r12-cache-complete-fixed-test-learning-curve-v1"
FIXED_TEST_PER_LABEL = 73
TRAIN_PER_LABEL = {80: 40, 240: 120, 400: 200, 564: 282}
EXPECTED_ELIGIBLE = {
    ("train", 0): 282,
    ("train", 1): 293,
    ("test", 0): 73,
    ("test", 1): 75,
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(record: dict[str, Any]) -> str:
    payload = "\x00".join([
        PROTOCOL,
        str(SEED),
        str(record["split"]),
        str(int(record["y"])),
        str(record["source_id"]),
        str(record["response_id"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['source_id']}\t{row['response_id']}" for row in sorted(
            rows, key=lambda item: (str(item["source_id"]), str(item["response_id"]))
        )
    )
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


def _ranked_eligible(
    records: list[dict[str, Any]], ineligible: set[str], *, split: str, label: int
) -> list[dict[str, Any]]:
    selected = [
        row for row in records
        if row["split"] == split
        and int(row["y"]) == label
        and str(row["source_id"]) not in ineligible
        and str(row["source_id"]) != QUARANTINED_SOURCE_ID
    ]
    return sorted(selected, key=_stable_key)


def _select(records: list[dict[str, Any]], ineligible: set[str]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    ranked = {
        (split, label): _ranked_eligible(records, ineligible, split=split, label=label)
        for split in ("train", "test") for label in (0, 1)
    }
    counts = {key: len(rows) for key, rows in ranked.items()}
    if counts != EXPECTED_ELIGIBLE:
        raise ValueError(f"unexpected cache-eligible R12 buckets: {counts}")

    fixed_test = ranked[("test", 0)][:FIXED_TEST_PER_LABEL] + ranked[("test", 1)][:FIXED_TEST_PER_LABEL]
    if Counter((row["split"], int(row["y"])) for row in fixed_test) != Counter({("test", 0): 73, ("test", 1): 73}):
        raise ValueError("fixed test is not balanced")

    trains: dict[int, list[dict[str, Any]]] = {}
    for train_size, per_label in TRAIN_PER_LABEL.items():
        rows = ranked[("train", 0)][:per_label] + ranked[("train", 1)][:per_label]
        if len(rows) != train_size or Counter((row["split"], int(row["y"])) for row in rows) != Counter({("train", 0): per_label, ("train", 1): per_label}):
            raise ValueError(f"train={train_size} is not balanced")
        trains[train_size] = rows

    previous: set[tuple[str, str]] = set()
    for train_size in sorted(trains):
        current = {(str(row["source_id"]), str(row["response_id"])) for row in trains[train_size]}
        if not previous.issubset(current):
            raise ValueError(f"train={train_size} is not nested in its predecessor")
        previous = current
    return fixed_test, trains


def _manifest(
    train_size: int,
    fixed_test: list[dict[str, Any]],
    train: list[dict[str, Any]],
    *,
    report_sha256: str,
    ineligible: set[str],
) -> dict[str, Any]:
    records = sorted(train + fixed_test, key=lambda row: (row["split"], row["source_id"], row["response_id"]))
    total = len(records)
    return {
        "version": 1,
        "task": "QA",
        "seed": SEED,
        "quotas": {"train_sources": train_size, "test_sources": len(fixed_test)},
        "records": records,
        "derivation": {
            "protocol": PROTOCOL,
            "parent_manifest_sha256": PARENT_SHA256,
            "parent_qa_total": PARENT_TOTAL,
            "selection": "v1 stable SHA-256 ranking within original split/label buckets after cache-only eligibility filter",
            "fixed_test": {
                "sources": len(fixed_test),
                "ids_sha256": _ids_sha256(fixed_test),
                "original_split": "test",
                "labels": {"0": FIXED_TEST_PER_LABEL, "1": FIXED_TEST_PER_LABEL},
            },
            "nested_train": {
                "sources": train_size,
                "ids_sha256": _ids_sha256(train),
                "labels": {"0": train_size // 2, "1": train_size // 2},
                "all_larger_train_points_include_this_point": True,
            },
            "cache_eligibility": {
                "criterion": "no missing support-critical cache-only preflight artifacts",
                "preflight_report_sha256": report_sha256,
                "ineligible_source_count": len(ineligible),
                "ineligible_source_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(ineligible)).encode("utf-8")
                ).hexdigest(),
            },
            "source_12448": "excluded from every curve manifest by the explicit R12 quarantine",
            "curve_train_sizes": sorted(TRAIN_PER_LABEL),
            "total_sources": total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--cache-preflight-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records = _load_parent(Path(args.parent_manifest))
    ineligible, report_sha256 = _load_ineligible(Path(args.cache_preflight_report), records)
    fixed_test, trains = _select(records, ineligible)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for train_size, train in sorted(trains.items()):
        output = output_dir / f"support-critical-r12-cache-complete-fixed-test-146-train-{train_size}.json"
        output.write_text(
            json.dumps(
                _manifest(train_size, fixed_test, train, report_sha256=report_sha256, ineligible=ineligible),
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"{output} {_sha256_bytes(output)}")


if __name__ == "__main__":
    main()
