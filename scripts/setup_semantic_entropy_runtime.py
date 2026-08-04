#!/usr/bin/env python3
"""Materialize the pinned local NLI asset used by the entropy baseline.

The caller supplies an external-disk destination.  No model file is written to
the repository, and the resulting manifest is content-addressed for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - installation error path
        raise SystemExit("huggingface-hub is required; install requirements-semantic-entropy.txt") from exc

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.model, revision=args.revision)
    resolved_revision = str(info.sha)
    snapshot_download(
        repo_id=args.model,
        revision=resolved_revision,
        local_dir=str(output),
    )
    required = ["config.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"downloaded NLI snapshot is incomplete: missing {missing}")
    files = [
        {"path": item.relative_to(output).as_posix(), "bytes": item.stat().st_size, "sha256": _sha256(item)}
        for item in sorted(output.rglob("*"))
        if item.is_file() and ".cache" not in item.relative_to(output).parts and item.name != "model-manifest.json"
    ]
    payload = {
        "protocol": "semantic-entropy-nli-snapshot-v1",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "materialized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": files,
    }
    temporary = output / f".model-manifest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "model-manifest.json")
    print(json.dumps({
        "model": args.model,
        "resolved_revision": resolved_revision,
        "output_dir": str(output),
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
