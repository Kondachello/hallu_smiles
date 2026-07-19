#!/usr/bin/env python3
"""Fail closed unless a downloaded Vertex 3-QA archive cleared its gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.core import canonical_manifest_sha256


def _read_json(archive: tarfile.TarFile, name: str) -> dict:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member is unreadable: {name}")
    payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"archive member is not a JSON object: {name}")
    return payload


def _read_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member is unreadable: {name}")
    return handle.read()


def _root_name(archive: tarfile.TarFile) -> str:
    roots = {
        PurePosixPath(member.name).parts[0]
        for member in archive.getmembers()
        if member.name and not member.name.startswith("/")
    }
    if len(roots) != 1:
        raise ValueError("probe archive must contain exactly one top-level directory")
    root = roots.pop()
    if not root.startswith("vertex-"):
        raise ValueError("archive is not a Vertex probe artifact")
    return root


def _require_empty(archive: tarfile.TarFile, path: str) -> None:
    if _read_bytes(archive, path) != b"":
        raise ValueError(f"probe recorded failed extraction(s): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--gateway-url", required=True)
    args = parser.parse_args()

    with tarfile.open(args.artifact, "r:gz") as archive:
        root = _root_name(archive)
        metadata = _read_json(archive, f"{root}/run_metadata.json")
        manifest = _read_json(archive, f"{root}/gateway-manifest.json")
        strict = _read_json(archive, f"{root}/strict-extract/extraction_summary.json")
        replay = _read_json(archive, f"{root}/cache-replay-extract/extraction_summary.json")
        usage = _read_json(archive, f"{root}/usage-counts.json")
        before = _read_bytes(archive, f"{root}/cache-before-replay.sha256")
        after = _read_bytes(archive, f"{root}/cache-after-replay.sha256")
        _require_empty(archive, f"{root}/strict-extract/failed_extractions.jsonl")
        _require_empty(archive, f"{root}/cache-replay-extract/failed_extractions.jsonl")

    if metadata.get("state") != "completed" or metadata.get("qa_pilot_limit") != 3:
        raise ValueError("probe metadata does not describe a completed deterministic 3-QA gate")
    if strict.get("status") != "ready" or strict.get("failures") != []:
        raise ValueError("probe strict extraction was incomplete")
    if strict.get("expected_sources") != 3 or strict.get("responses_completed") != 3:
        raise ValueError("probe did not complete all three source/answer graphs")
    if strict.get("expected_records") != strict.get("completed_records") or strict != replay:
        raise ValueError("probe cache-only replay does not reproduce strict extraction")
    if before != after:
        raise ValueError("probe cache hashes changed during cache-only replay")
    if usage.get("extraction_cache_only", {}).get("api_calls") != 0:
        raise ValueError("probe cache-only extraction made live HTTP calls")
    if manifest.get("logical_model") != "openai/gemini-2.5-flash":
        raise ValueError("probe used an unexpected logical model")
    gateway_url = metadata.get("gateway_manifest", {}).get("gateway_url")
    # The gateway URL is deliberately bound outside the model cache key: a
    # different origin is a different trust boundary even if it advertises a
    # similarly named model.
    if gateway_url not in (None, args.gateway_url):
        raise ValueError("probe metadata was produced through another gateway origin")

    print(json.dumps({
        "source_commit": metadata.get("source_commit"),
        "gateway_manifest_sha256": canonical_manifest_sha256(manifest),
        "gateway_release": manifest.get("gateway_release"),
        "cloud_run_revision": manifest.get("cloud_run_revision"),
        "artifact_sha256": hashlib.sha256(open(args.artifact, "rb").read()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
