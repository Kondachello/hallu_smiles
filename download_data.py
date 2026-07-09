#!/usr/bin/env python3
"""Download the RAGTruth corpus (source_info.jsonl + response.jsonl) into ./data.

Usage:
    python download_data.py [--data-dir data]

Files come from the official repo:
    https://github.com/ParticleMedia/RAGTruth/tree/main/dataset
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ["source_info.jsonl", "response.jsonl"]


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest} already exists ({dest.stat().st_size} bytes)")
            continue
        url = f"{BASE}/{name}"
        print(f"[get ] {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"[FAIL] {url}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[ok  ] {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    download(Path(args.data_dir))
