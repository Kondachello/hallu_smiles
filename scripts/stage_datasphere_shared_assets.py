#!/usr/bin/env python3
"""Stage immutable model and RAGTruth assets from a DataSphere Jupyter c1.4 VM.

This program is intentionally *not* a DataSphere Job entrypoint.  Jobs mount
``DS_PROJECT_HOME`` for reads, so the one-time write must happen in Jupyter.
It never prints the value of ``HF_TOKEN``.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_FAMILY = "llama-3.1-8b"
MODEL_READY = ".hallu_smiles_model_ready"
MODEL_MANIFEST = "model-manifest.json"
DATA_MANIFEST = "ragtruth-manifest.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_inventory(root: Path) -> list[dict[str, int | str]]:
    return [
        {"path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size}
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name not in {MODEL_READY, MODEL_MANIFEST}
    ]


def _ready_model(path: Path, model_id: str) -> bool:
    manifest_path = path / MODEL_MANIFEST
    ready_path = path / MODEL_READY
    if not manifest_path.is_file() or not ready_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("model_id") == model_id and bool(manifest.get("revision"))


def stage_model(shared_root: Path, model_id: str, revision: str | None) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required only while staging the gated Meta model.")

    # Keep this import lazy: offline tests and normal GPU Jobs must not need it.
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    resolved_revision = api.model_info(model_id, revision=revision).sha
    family_root = shared_root / "models" / MODEL_FAMILY
    destination = family_root / resolved_revision
    if _ready_model(destination, model_id):
        print(f"[skip] shared model is ready: {destination}")
        return destination
    if destination.exists():
        raise RuntimeError(
            f"{destination} exists without a valid ready marker; inspect it before staging again."
        )

    partial = family_root / f".{resolved_revision}.partial"
    partial.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {model_id}@{resolved_revision} -> {partial}")
    snapshot_download(model_id, revision=resolved_revision, local_dir=partial, token=token)
    if not (partial / "config.json").is_file() or not list(partial.glob("*.safetensors")):
        raise RuntimeError("model download completed without config.json and safetensors weights")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": resolved_revision,
        "staged_at_utc": _utc_now(),
        "files": _file_inventory(partial),
    }
    _atomic_json(partial / MODEL_MANIFEST, manifest)
    (partial / MODEL_READY).write_text(f"ready {resolved_revision}\n", encoding="utf-8")
    family_root.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    _atomic_json(
        family_root / "active-model.json",
        {
            "schema_version": 1,
            "model_id": model_id,
            "revision": resolved_revision,
            "model_dir": destination.name,
            "activated_at_utc": _utc_now(),
        },
    )
    print(f"[ok  ] shared model ready: {destination}")
    return destination


def stage_ragtruth(shared_root: Path) -> Path:
    import sys

    sys.path.insert(0, str(ROOT))
    from download_data import FILES, download

    data_dir = shared_root / "ragtruth"
    download(data_dir)
    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"RAGTruth files missing after download: {', '.join(missing)}")
    _atomic_json(
        data_dir / DATA_MANIFEST,
        {
            "schema_version": 1,
            "source": "ParticleMedia/RAGTruth dataset",
            "staged_at_utc": _utc_now(),
            "files": [
                {"path": name, "bytes": (data_dir / name).stat().st_size}
                for name in FILES
            ],
        },
    )
    print(f"[ok  ] shared RAGTruth ready: {data_dir}")
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-root",
        default=os.environ.get("DS_SHARED_ROOT", ""),
        help="Shared Project-storage root; normally $DS_PROJECT_HOME/hallu_smiles/shared.",
    )
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--revision", help="Optional HF revision; otherwise resolve the current commit once.")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    if not args.shared_root:
        raise SystemExit("--shared-root or DS_SHARED_ROOT is required")
    if args.model_only and args.data_only:
        raise SystemExit("--model-only and --data-only cannot be used together")
    shared_root = Path(args.shared_root).expanduser().resolve()
    if not args.data_only:
        stage_model(shared_root, args.model_id, args.revision)
    if not args.model_only:
        stage_ragtruth(shared_root)


if __name__ == "__main__":
    main()
