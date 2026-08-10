#!/usr/bin/env python3
"""Materialize the attested R12 QA manifest from its successful terminal archive.

This imports only the immutable, non-secret experiment manifest.  It checks
the recorded SHA-256 before atomically writing the file that a diagnostic
cache-preflight Job will pin by Git commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


EXPECTED_SHA256 = "208dfc9e96c03039b5f8adeffe3e5174b6d51a835677ddc79ae4d2c40006f39c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with tarfile.open(args.archive, "r:gz") as archive:
        entries = [
            item for item in archive.getmembers()
            if item.name.endswith("/qa_manifest.json") and "/cache-replay/" not in item.name
        ]
        if len(entries) != 1:
            raise SystemExit(f"expected exactly one terminal qa_manifest.json, found {len(entries)}")
        raw_file = archive.extractfile(entries[0])
        if raw_file is None:
            raise SystemExit("terminal qa_manifest.json is not readable")
        raw = raw_file.read()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("terminal R12 qa_manifest.json SHA-256 mismatch")
    payload = json.loads(raw)
    if payload.get("quotas") != {"train_sources": 600, "test_sources": 150}:
        raise SystemExit("terminal R12 manifest has an unexpected sample shape")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 750:
        raise SystemExit("terminal R12 manifest does not contain 750 records")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, output)
    print(f"{output} {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
