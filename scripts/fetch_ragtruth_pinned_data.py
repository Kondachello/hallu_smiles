#!/usr/bin/env python3
"""Fetch the two RAGTruth JSONL inputs at one exact upstream commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llama31_eval import DEFAULT_RAGTRUTH_COMMIT, file_sha256  # noqa: E402


def _fetch(url: str, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError("download is empty")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit", default=DEFAULT_RAGTRUTH_COMMIT)
    args = parser.parse_args()
    if args.commit != DEFAULT_RAGTRUTH_COMMIT:
        raise SystemExit(f"controlled evaluation pins {DEFAULT_RAGTRUTH_COMMIT}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{args.commit}/dataset"
    files = ("source_info.jsonl", "response.jsonl")
    for name in files:
        _fetch(f"{base}/{name}", output / name)
    provenance = {
        "protocol": "ragtruth-pinned-jsonl-v1",
        "repository": "https://github.com/ParticleMedia/RAGTruth",
        "commit": args.commit,
        "source_info_jsonl_sha256": file_sha256(output / "source_info.jsonl"),
        "response_jsonl_sha256": file_sha256(output / "response.jsonl"),
    }
    (output / "ragtruth_data_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"commit": args.commit, **{
        key: value for key, value in provenance.items() if key.endswith("sha256")
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
