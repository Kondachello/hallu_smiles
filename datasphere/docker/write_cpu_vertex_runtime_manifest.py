#!/usr/bin/env python3
"""Write the immutable dependency fingerprint for the CPU Vertex Job image."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--client-freeze", required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--hhem-path", required=True)
    parser.add_argument("--hhem-revision", required=True)
    parser.add_argument("--hhem-foundation-revision", required=True)
    parser.add_argument("--hhem-foundation-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    embedding = Path(args.embedding_path)
    if not (embedding / "config.json").is_file():
        raise SystemExit(f"incomplete embedding snapshot: {embedding}")
    hhem = Path(args.hhem_path)
    if not (hhem / "config.json").is_file() or not (hhem / "model.safetensors").is_file():
        raise SystemExit(f"incomplete HHEM snapshot: {hhem}")
    program = (
        "import json,platform,torch; from importlib import metadata; "
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,'kg-gen':metadata.version('kg-gen'),"
        "'dspy':metadata.version('dspy'),'litellm':metadata.version('litellm')},sort_keys=True))"
    )
    completed = subprocess.run(
        [args.client_python, "-c", program], check=True, capture_output=True, text=True, timeout=60
    )
    payload = {
        "runtime_protocol": "hallu-datasphere-cpu-vertex-v1",
        "source_commit": args.source_commit,
        "python": "3.11",
        "client_runtime": json.loads(completed.stdout),
        "client_freeze_sha256": _sha256(Path(args.client_freeze)),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_revision": args.embedding_revision,
        "embedding_path": str(embedding),
        "hhem_model": "vectara/hallucination_evaluation_model",
        "hhem_revision": args.hhem_revision,
        "hhem_path": str(hhem),
        "hhem_foundation_model": "google/flan-t5-base",
        "hhem_foundation_revision": args.hhem_foundation_revision,
        "hhem_foundation_cache": args.hhem_foundation_cache,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["runtime_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
