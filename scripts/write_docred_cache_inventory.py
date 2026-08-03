#!/usr/bin/env python3
"""Write a cache-key-free inventory digest for a DocRED cache namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.cache_root)
    if not root.is_dir():
        raise SystemExit("cache root does not exist")
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        # Relative names may include content-addressed cache keys. They affect
        # the local digest but are never serialised in the public artifact.
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                total_bytes += len(block)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol": "docred-cache-inventory-v1",
        "files": len(files),
        "bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
