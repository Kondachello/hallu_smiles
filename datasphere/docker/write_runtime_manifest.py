#!/usr/bin/env python3
"""Write the immutable image's dependency and asset fingerprint."""
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


def _python_runtime(python: str) -> dict[str, str | None]:
    program = (
        "import json,platform,torch; from importlib import metadata; "
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'vllm':metadata.version('vllm') if 'server' in __import__('sys').argv else None}))"
    )
    role = "server" if "server" in Path(python).parts else "client"
    completed = subprocess.run(
        [python, "-c", program, role], check=True, capture_output=True, text=True, timeout=60
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--server-freeze", required=True)
    parser.add_argument("--client-freeze", required=True)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    server = Path(args.server_freeze)
    client = Path(args.client_freeze)
    embedding = Path(args.embedding_path)
    if not embedding.is_dir() or not (embedding / "config.json").is_file():
        raise SystemExit(f"incomplete embedding snapshot: {embedding}")
    payload = {
        "runtime_protocol": "hallu-datasphere-vllm085-cu118-v1",
        "source_commit": args.source_commit,
        "python": "3.11",
        "server_runtime": _python_runtime(args.server_python),
        "client_runtime": _python_runtime(args.client_python),
        "server_freeze_sha256": _sha256(server),
        "client_freeze_sha256": _sha256(client),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_revision": args.embedding_revision,
        "embedding_path": str(embedding),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["runtime_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
