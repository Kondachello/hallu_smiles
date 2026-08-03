#!/usr/bin/env python3
"""Write a secret-free runtime identity for a local Vertex gateway run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    embedding = Path(args.embedding_model_path).resolve()
    config = embedding / "config.json"
    if not config.is_file():
        raise SystemExit("local S-BERT snapshot has no config.json")
    payload = {
        "protocol": "hallu-local-vertex-runtime-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            distribution: _version(distribution)
            for distribution in ("kg-gen", "litellm", "sentence-transformers", "torch")
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "config_sha256": _sha256(config),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["runtime_fingerprint"] = "local-cpu:" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"runtime_fingerprint": payload["runtime_fingerprint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
