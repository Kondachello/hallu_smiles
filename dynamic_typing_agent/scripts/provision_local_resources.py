#!/usr/bin/env python3
"""Provision pinned HHEM and RAGTruth resources once; normal runtime stays offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HHEM_REPO = "vectara/hallucination_evaluation_model"
HHEM_REVISION = "0e7edb3689e710c52ba120086e8f91ea3ee87f23"
HHEM_MODEL_SHA256 = "634de18a38cf1e991c1acd0f7a9e0d30f7ea187fba42bb4798f862d3edd31e72"
FOUNDATION_REPO = "google/flan-t5-base"
FOUNDATION_REVISION = "d224e0d50f2fe7d975c973cf46d933e4dfaf2a3e"
FOUNDATION_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model")
RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_FILES = ("source_info.jsonl", "response.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_ragtruth(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in RAGTRUTH_FILES:
        target = destination / name
        if target.is_file() and target.stat().st_size:
            continue
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_COMMIT}/dataset/{name}"
        with urllib.request.urlopen(url) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)


def bind_hhem_foundation(hhem_dir: Path, foundation_dir: Path) -> None:
    """Patch HHEM's checked-in custom code to use its local FLAN tokenizer/config."""
    config_path = hhem_dir / "config.json"
    source_path = hhem_dir / "configuration_hhem_v2.py"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    current = payload.get("foundation", FOUNDATION_REPO)
    if current not in (FOUNDATION_REPO, str(foundation_dir)):
        raise RuntimeError(f"unexpected HHEM foundation value: {current!r}")
    source = source_path.read_text(encoding="utf-8")
    expression = re.compile(r'^(?P<prefix>\s*foundation\s*=\s*)["\']google/flan-t5-base["\']\s*$', re.MULTILINE)
    patched, replacements = expression.subn(
        lambda match: f"{match.group('prefix')}{json.dumps(str(foundation_dir))}", source, count=1
    )
    if replacements:
        source_path.write_text(patched, encoding="utf-8")
    elif current != str(foundation_dir):
        raise RuntimeError("could not bind HHEM custom configuration to local FLAN-T5")
    payload["foundation"] = str(foundation_dir)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provision(root: Path, *, verify_only: bool) -> dict[str, object]:
    hhem_dir = root / "hhem-2.1-open"
    foundation_dir = root / "flan-t5-base"
    ragtruth_dir = root / "ragtruth"
    if not verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HHEM_REPO,
            revision=HHEM_REVISION,
            local_dir=hhem_dir,
            local_dir_use_symlinks=False,
        )
        snapshot_download(
            repo_id=FOUNDATION_REPO,
            revision=FOUNDATION_REVISION,
            local_dir=foundation_dir,
            allow_patterns=list(FOUNDATION_FILES),
            local_dir_use_symlinks=False,
        )
        download_ragtruth(ragtruth_dir)
    foundation_required = tuple(foundation_dir / name for name in FOUNDATION_FILES)
    required = (hhem_dir / "config.json", hhem_dir / "model.safetensors", *foundation_required, *(ragtruth_dir / name for name in RAGTRUTH_FILES))
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"local resource validation failed; missing: {missing}")
    model_checksum = sha256(hhem_dir / "model.safetensors")
    if model_checksum != HHEM_MODEL_SHA256:
        raise RuntimeError("HHEM model.safetensors SHA-256 does not match pinned revision")
    bind_hhem_foundation(hhem_dir, foundation_dir)
    files = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required}
    return {
        "schema_version": "dynamic-typing-local-resources-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hhem": {
            "repo_id": HHEM_REPO,
            "revision": HHEM_REVISION,
            "model_sha256": HHEM_MODEL_SHA256,
            "foundation_repo_id": FOUNDATION_REPO,
            "foundation_revision": FOUNDATION_REVISION,
        },
        "ragtruth": {"repository": "ParticleMedia/RAGTruth", "commit": RAGTRUTH_COMMIT},
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_resources")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = provision(args.root.resolve(), verify_only=args.verify_only)
    (args.root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"resource_root": str(args.root), "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
