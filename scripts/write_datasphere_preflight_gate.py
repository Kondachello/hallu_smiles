#!/usr/bin/env python3
"""Write the successful CPU-preflight identity consumed by later Job gates."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


RUNTIME_PROTOCOL = "hallu-datasphere-vllm085-cu118-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--shared-assets-report", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_report = _load(args.runtime_report)
    shared = _load(args.shared_assets_report)
    if runtime_report.get("status") != "ready":
        raise SystemExit("runtime preflight report is not ready")
    if shared.get("status") != "ready":
        raise SystemExit("shared-assets preflight report is not ready")
    manifest = runtime_report.get("runtime_manifest") or {}
    if manifest.get("source_commit") != args.source_commit:
        raise SystemExit("runtime image was built from a different source commit")
    if manifest.get("runtime_protocol") != RUNTIME_PROTOCOL:
        raise SystemExit("runtime manifest uses an unsupported protocol")
    if not SHA256_RE.fullmatch(str(manifest.get("runtime_fingerprint", ""))):
        raise SystemExit("runtime manifest fingerprint is not a SHA-256 digest")
    if shared.get("model_id") != args.model_id:
        raise SystemExit("shared-assets report belongs to a different model")
    if not shared.get("model_revision"):
        raise SystemExit("shared-assets report has no exact model revision")

    payload = {
        "state": "completed",
        "mode": "preflight",
        "source_commit": args.source_commit,
        "datasphere_docker_image_id": args.docker_image_id,
        "model_id": args.model_id,
        "model_revision": shared.get("model_revision"),
        "runtime_protocol": manifest.get("runtime_protocol"),
        "image_runtime_fingerprint": manifest.get("runtime_fingerprint"),
        "runtime_fingerprint": (
            f"{args.docker_image_id}:{manifest.get('runtime_fingerprint')}"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
