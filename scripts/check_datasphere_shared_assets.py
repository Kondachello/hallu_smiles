#!/usr/bin/env python3
"""Verify the immutable assets that a DataSphere Job reads from shared storage."""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_READY = ".hallu_smiles_model_ready"
MODEL_MANIFEST = "model-manifest.json"
DATA_MANIFEST = "ragtruth-manifest.json"
DATA_FILES = ("source_info.jsonl", "response.jsonl")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(model_path: Path, data_dir: Path, model_id: str) -> dict[str, Any]:
    manifest_path = model_path / MODEL_MANIFEST
    ready_path = model_path / MODEL_READY
    if not ready_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"shared model is not ready: {model_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid model manifest: {manifest_path}") from exc
    if manifest.get("model_id") != model_id:
        raise RuntimeError(f"shared model ID mismatch: {manifest.get('model_id')!r} != {model_id!r}")
    if not manifest.get("revision"):
        raise RuntimeError("model manifest has no immutable revision")
    checked_bytes = 0
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        expected_bytes = entry.get("bytes")
        path = model_path / str(relative)
        if not isinstance(expected_bytes, int) or not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"shared model file does not match manifest: {relative}")
        checked_bytes += expected_bytes
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise RuntimeError("shared model is missing config.json or safetensors weights")

    data_manifest_path = data_dir / DATA_MANIFEST
    if not data_manifest_path.is_file():
        raise RuntimeError(f"RAGTruth manifest is absent: {data_manifest_path}")
    try:
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid RAGTruth manifest: {data_manifest_path}") from exc
    expected_data_sizes = {entry.get("path"): entry.get("bytes") for entry in data_manifest.get("files", [])}
    for name in DATA_FILES:
        path = data_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"RAGTruth file is absent or empty: {path}")
        if expected_data_sizes.get(name) != path.stat().st_size:
            raise RuntimeError(f"RAGTruth file does not match manifest: {path}")
    return {
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_id": model_id,
        "model_revision": manifest["revision"],
        "model_path": str(model_path),
        "model_bytes_checked": checked_bytes,
        "data_dir": str(data_dir),
        "status": "ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--report", help="Optional writable JSON report path.")
    args = parser.parse_args()
    report = check(Path(args.model_path), Path(args.data_dir), args.model_id)
    if args.report:
        _atomic_json(Path(args.report), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
