#!/usr/bin/env python3
"""Validate the immutable artifact that authorizes the next DataSphere gate."""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from datasphere_runtime_image import require_runtime_image
except ImportError:  # pragma: no cover - package import in unit tests
    from scripts.datasphere_runtime_image import require_runtime_image


MAX_JSON_BYTES = 5 * 1024 * 1024
RUNTIME_PROTOCOL = "hallu-datasphere-vllm085-cu118-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GateArtifact:
    def __init__(self, path: Path):
        self.path = path
        self.tar = tarfile.open(path, mode="r:*")
        for member in self.tar.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe archive member: {member.name!r}")

    def close(self) -> None:
        self.tar.close()

    def _one(self, suffix: str) -> tarfile.TarInfo:
        matches = [
            member
            for member in self.tar.getmembers()
            if member.isfile() and (member.name == suffix or member.name.endswith("/" + suffix))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one regular {suffix!r} in {self.path}, found {len(matches)}"
            )
        return matches[0]

    def raw(self, suffix: str, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
        member = self._one(suffix)
        if member.size > max_bytes:
            raise ValueError(f"archive member {suffix!r} is unexpectedly large: {member.size}")
        handle = self.tar.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read archive member {suffix!r}")
        return handle.read()

    def json(self, suffix: str) -> dict[str, Any]:
        payload = json.loads(self.raw(suffix).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{suffix!r} must contain a JSON object")
        return payload


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def _validate_preflight(
    artifact: GateArtifact, *, commit: str, image_id: str, model_id: str
) -> dict[str, Any]:
    gate = artifact.json("gate_metadata.json")
    runtime = artifact.json("runtime-dependencies.json")
    shared = artifact.json("preflight.json")
    _expect(gate.get("state"), "completed", "preflight state")
    _expect(gate.get("mode"), "preflight", "preflight mode")
    _expect(gate.get("source_commit"), commit, "preflight source commit")
    _expect(gate.get("datasphere_docker_image_id"), image_id, "preflight Docker image")
    _expect(gate.get("model_id"), model_id, "preflight model")
    _expect(gate.get("runtime_protocol"), RUNTIME_PROTOCOL, "preflight runtime protocol")
    _expect(runtime.get("status"), "ready", "runtime report status")
    _expect(shared.get("status"), "ready", "shared-assets status")
    _expect(shared.get("model_id"), model_id, "shared-assets model")
    _expect(shared.get("model_revision"), gate.get("model_revision"), "model revision")
    manifest = runtime.get("runtime_manifest") or {}
    _expect(manifest.get("source_commit"), commit, "runtime image source commit")
    _expect(manifest.get("runtime_protocol"), RUNTIME_PROTOCOL, "runtime protocol")
    _expect(
        gate.get("image_runtime_fingerprint"),
        manifest.get("runtime_fingerprint"),
        "runtime fingerprint",
    )
    if not SHA256_RE.fullmatch(str(manifest.get("runtime_fingerprint", ""))):
        raise ValueError("preflight runtime fingerprint is not a SHA-256 digest")
    return gate


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _expect(manifest.get("version"), 1, "QA manifest version")
    _expect(manifest.get("task"), "QA", "QA manifest task")
    _expect(manifest.get("seed"), 42, "QA manifest seed")
    _expect(manifest.get("quotas"), {"train_sources": 16, "test_sources": 4}, "QA quotas")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("QA manifest must contain exactly 20 records")
    sources = [str(record.get("source_id", "")) for record in records]
    responses = [str(record.get("response_id", "")) for record in records]
    if "" in sources or len(set(sources)) != 20:
        raise ValueError("QA manifest must contain 20 unique non-empty source_id values")
    if "" in responses or len(set(responses)) != 20:
        raise ValueError("QA manifest must contain 20 unique non-empty response_id values")


def _validate_cluster_probe(
    artifact: GateArtifact, *, commit: str, image_id: str, model_id: str
) -> dict[str, Any]:
    metadata = artifact.json("run_metadata.json")
    identity = artifact.json("runtime-identity.json")
    runtime = artifact.json("runtime-manifest.json")
    shared = artifact.json("shared-assets-preflight.json")
    _expect(metadata.get("state"), "completed", "cluster probe state")
    _expect(metadata.get("mode"), "cluster-runtime-probe", "cluster probe mode")
    _expect(metadata.get("qa_pilot_limit"), 3, "cluster probe QA limit")
    _expect(metadata.get("runs"), ["strict-extract"], "cluster probe runs")
    _expect(metadata.get("source_commit"), commit, "cluster probe source commit")
    _expect(metadata.get("datasphere_docker_image_id"), image_id, "cluster Docker image")
    _expect(metadata.get("model_id"), model_id, "cluster model")
    _expect(metadata.get("guided_decoding_backend"), "xgrammar", "server backend")
    _expect(identity.get("source_commit"), commit, "runtime identity source commit")
    _expect(identity.get("datasphere_docker_image_id"), image_id, "runtime identity image")
    _expect(identity.get("runtime_protocol"), RUNTIME_PROTOCOL, "runtime identity protocol")
    _expect(
        identity.get("image_runtime_fingerprint"),
        runtime.get("runtime_fingerprint"),
        "cluster image runtime fingerprint",
    )
    if not SHA256_RE.fullmatch(str(runtime.get("runtime_fingerprint", ""))):
        raise ValueError("cluster runtime fingerprint is not a SHA-256 digest")
    _expect(metadata.get("runtime_fingerprint"), identity.get("runtime_fingerprint"), "runtime identity")
    _expect(runtime.get("source_commit"), commit, "Docker runtime source commit")
    _expect(runtime.get("runtime_protocol"), RUNTIME_PROTOCOL, "Docker runtime protocol")
    _expect(shared.get("status"), "ready", "cluster shared-assets status")
    _expect(shared.get("model_id"), model_id, "cluster shared-assets model")
    _expect(shared.get("model_revision"), metadata.get("model_revision"), "cluster model revision")
    for report_name in (
        "vllm-response-format-probe.json",
        "kggen-probe.json",
        "verifier-probe.json",
        "qa-reference-probe.json",
    ):
        _expect(artifact.json(report_name).get("status"), "ready", f"{report_name} status")
    if artifact.raw("strict/failed_extractions.jsonl", max_bytes=1024) != b"":
        raise ValueError("cluster probe has failed KG extractions")
    _validate_manifest(artifact.json("qa_pilot_manifest.json"))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("preflight", "cluster-probe-g1"), required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--commit", required=True)
    image = parser.add_mutually_exclusive_group(required=True)
    image.add_argument("--docker-image-id")
    image.add_argument("--docker-image")
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    runtime_image = args.docker_image_id or args.docker_image
    try:
        require_runtime_image(runtime_image, registry=args.docker_image is not None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    artifact_path = Path(args.artifact)
    if not artifact_path.is_file():
        raise SystemExit(f"gate artifact does not exist: {artifact_path}")
    artifact = GateArtifact(artifact_path)
    try:
        if args.gate == "preflight":
            result = _validate_preflight(
                artifact, commit=args.commit, image_id=runtime_image, model_id=args.model_id
            )
        else:
            result = _validate_cluster_probe(
                artifact, commit=args.commit, image_id=runtime_image, model_id=args.model_id
            )
    finally:
        artifact.close()
    print(json.dumps({"status": "ready", "gate": args.gate, "identity": result}, sort_keys=True))


if __name__ == "__main__":
    main()
