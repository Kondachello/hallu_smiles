#!/usr/bin/env python3
"""Fail fast unless the CPU-only DataSphere Vertex runtime is complete."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "torch": "2.6.0+cpu",
    "kg-gen": "0.4.0",
    "dspy": "2.6.27",
    "litellm": "1.60.4",
    "sentence-transformers": "5.6.0",
    "jsonschema": "4.23.0",
}
PROTOCOL = "hallu-datasphere-cpu-vertex-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.runtime_manifest).read_text(encoding="utf-8"))
    if manifest.get("runtime_protocol") != PROTOCOL:
        raise RuntimeError("unexpected CPU runtime protocol")
    if manifest.get("source_commit") != args.expected_source_commit:
        raise RuntimeError("CPU runtime image was built from another source commit")
    if manifest.get("embedding_path") != args.embedding_path:
        raise RuntimeError("CPU runtime embedding path mismatch")
    program = (
        "import json,torch; from importlib import metadata; "
        f"names={list(EXPECTED)!r}; print(json.dumps({{'versions':{{n:metadata.version(n) for n in names}},'torch_cuda':torch.version.cuda}},sort_keys=True))"
    )
    checked = subprocess.run([args.python, "-c", program], check=True, text=True, capture_output=True, timeout=120)
    actual = json.loads(checked.stdout)
    if actual["versions"] != EXPECTED or actual["torch_cuda"] is not None:
        raise RuntimeError(f"CPU runtime dependency mismatch: {actual}")
    embedding_program = (
        "import sys; from sentence_transformers import SentenceTransformer; "
        "m=SentenceTransformer(sys.argv[1],device='cpu',local_files_only=True); "
        "assert m.encode(['offline'],convert_to_numpy=True).shape[0] == 1"
    )
    subprocess.run([args.python, "-c", embedding_program, args.embedding_path], check=True, timeout=120)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "runtime_protocol": PROTOCOL,
        "runtime_manifest": manifest,
        "versions": actual["versions"],
        "torch_cuda": actual["torch_cuda"],
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
