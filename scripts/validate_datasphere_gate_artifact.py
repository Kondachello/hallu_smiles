#!/usr/bin/env python3
"""Validate the immutable artifact that authorizes the next DataSphere gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from datasphere_runtime_image import require_runtime_image
except ImportError:  # pragma: no cover - package import in unit tests
    from scripts.datasphere_runtime_image import require_runtime_image


MAX_JSON_BYTES = 5 * 1024 * 1024
RUNTIME_PROTOCOL = "hallu-datasphere-vllm085-cu118-v1"
XGRAMMAR_STRICT_REQUEST_BACKEND = (
    "xgrammar:disable-any-whitespace,no-fallback"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLUSTER_CONTEXT_MODE = "source_text"
CLUSTER_CONTEXT_PROTOCOL = "kggen-native-strict-equivalence-v2"
STRUCTURED_OUTPUT_PROTOCOL = "strict-response-format-v4-xgrammar-runtime-input-contracts"


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
    _expect(
        runtime.get("structured_output_protocol"),
        STRUCTURED_OUTPUT_PROTOCOL,
        "preflight structured-output protocol",
    )
    xgrammar_contract = runtime.get("xgrammar_contract") or {}
    _expect(
        xgrammar_contract.get("request_backend"),
        XGRAMMAR_STRICT_REQUEST_BACKEND,
        "preflight XGrammar request backend",
    )
    _expect(
        xgrammar_contract.get("backend_options"),
        ["disable-any-whitespace", "no-fallback"],
        "preflight XGrammar backend options",
    )
    _expect(
        xgrammar_contract.get("any_whitespace"),
        False,
        "preflight XGrammar whitespace mode",
    )
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


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
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
    split_labels = Counter()
    for record in records:
        split = record.get("split")
        label = record.get("y")
        if split not in {"train", "test"} or label not in {0, 1}:
            raise ValueError("QA manifest records require split=train|test and y=0|1")
        split_labels[(split, int(label))] += 1
    expected_labels = Counter({
        ("train", 0): 8,
        ("train", 1): 8,
        ("test", 0): 2,
        ("test", 1): 2,
    })
    if split_labels != expected_labels:
        raise ValueError(
            f"QA manifest split/label quotas mismatch: {dict(split_labels)}"
        )
    if records != sorted(
        records,
        key=lambda record: (
            str(record.get("split")),
            str(record.get("source_id")),
            str(record.get("response_id")),
        ),
    ):
        raise ValueError("QA manifest records are not in deterministic order")
    return records


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
    _expect(
        metadata.get("structured_output_protocol"),
        STRUCTURED_OUTPUT_PROTOCOL,
        "cluster structured-output protocol",
    )
    _expect(
        metadata.get("guided_decoding_request_backend"),
        XGRAMMAR_STRICT_REQUEST_BACKEND,
        "request backend",
    )
    _expect(metadata.get("xgrammar_any_whitespace"), False, "XGrammar whitespace mode")
    _expect(metadata.get("cluster_context_mode"), CLUSTER_CONTEXT_MODE, "cluster context mode")
    _expect(
        metadata.get("cluster_context_protocol"),
        CLUSTER_CONTEXT_PROTOCOL,
        "cluster context protocol",
    )
    _expect(identity.get("source_commit"), commit, "runtime identity source commit")
    _expect(identity.get("datasphere_docker_image_id"), image_id, "runtime identity image")
    _expect(identity.get("runtime_protocol"), RUNTIME_PROTOCOL, "runtime identity protocol")
    _expect(
        identity.get("structured_output_protocol"),
        STRUCTURED_OUTPUT_PROTOCOL,
        "runtime identity structured-output protocol",
    )
    _expect(
        identity.get("guided_decoding_request_backend"),
        XGRAMMAR_STRICT_REQUEST_BACKEND,
        "runtime identity request backend",
    )
    _expect(
        (identity.get("server_launch") or {}).get("guided_decoding_request_backend"),
        XGRAMMAR_STRICT_REQUEST_BACKEND,
        "runtime launch request backend",
    )
    _expect(
        (identity.get("server_launch") or {}).get("structured_output_protocol"),
        STRUCTURED_OUTPUT_PROTOCOL,
        "runtime launch structured-output protocol",
    )
    _expect(
        (identity.get("server_launch") or {}).get("xgrammar_any_whitespace"),
        False,
        "runtime launch XGrammar whitespace mode",
    )
    _expect(identity.get("cluster_context_mode"), CLUSTER_CONTEXT_MODE, "identity cluster context")
    _expect(
        identity.get("cluster_context_protocol"),
        CLUSTER_CONTEXT_PROTOCOL,
        "identity cluster protocol",
    )
    _expect(
        (identity.get("server_launch") or {}).get("cluster_context_mode"),
        CLUSTER_CONTEXT_MODE,
        "runtime launch cluster context",
    )
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
        report = artifact.json(report_name)
        _expect(report.get("status"), "ready", f"{report_name} status")
        _expect(
            report.get("structured_output_protocol"),
            STRUCTURED_OUTPUT_PROTOCOL,
            f"{report_name} structured-output protocol",
        )
        _expect(
            report.get("guided_decoding_request_backend"),
            XGRAMMAR_STRICT_REQUEST_BACKEND,
            f"{report_name} request backend",
        )
        _expect(
            report.get("xgrammar_any_whitespace"),
            False,
            f"{report_name} XGrammar whitespace mode",
        )
        if report_name == "qa-reference-probe.json":
            _expect(
                report.get("cluster_context_mode"),
                CLUSTER_CONTEXT_MODE,
                "reference probe cluster context",
            )
            audit = report.get("cluster_audit") or {}
            _expect(audit.get("protocol"), CLUSTER_CONTEXT_PROTOCOL, "reference cluster audit")
            _expect(audit.get("context_mode"), CLUSTER_CONTEXT_MODE, "reference audit context")
    manifest_records = _validate_manifest(artifact.json("qa_pilot_manifest.json"))
    extraction = artifact.json("strict/extraction_summary.json")
    _expect(extraction.get("protocol"), "hallu-extraction-summary-v1", "extraction summary protocol")
    _expect(extraction.get("status"), "ready", "extraction summary status")
    expected_prefix = [
        {
            "source_id": str(record["source_id"]),
            "response_id": str(record["response_id"]),
            "split": record["split"],
            "y": int(record["y"]),
        }
        for record in manifest_records[:3]
    ]
    _expect(extraction.get("expected_records"), expected_prefix, "3-QA expected records")
    _expect(extraction.get("completed_records"), expected_prefix, "3-QA completed records")
    _expect(extraction.get("expected_sources"), 3, "3-QA expected sources")
    _expect(extraction.get("references_completed"), 3, "3-QA references")
    _expect(extraction.get("responses_completed"), 3, "3-QA responses")
    _expect(extraction.get("pairs_completed"), 3, "3-QA pairs")
    _expect(extraction.get("failures"), [], "3-QA extraction failures")
    graph_records = extraction.get("graph_records")
    if not isinstance(graph_records, list) or len(graph_records) != 3:
        raise ValueError("extraction summary must contain exactly three graph records")
    graph_identities = [
        {
            "source_id": str(record.get("source_id", "")),
            "response_id": str(record.get("response_id", "")),
            "split": record.get("split"),
            "y": record.get("y"),
        }
        for record in graph_records
    ]
    _expect(graph_identities, expected_prefix, "3-QA graph record identities")
    expected_cache_keys = extraction.get("expected_cache_keys")
    if (
        not isinstance(expected_cache_keys, list)
        or not expected_cache_keys
        or len(expected_cache_keys) != len(set(expected_cache_keys))
        or any(not SHA256_RE.fullmatch(str(key)) for key in expected_cache_keys)
    ):
        raise ValueError("extraction summary expected_cache_keys are invalid")
    cache_records = extraction.get("cache_records")
    if not isinstance(cache_records, list) or {
        str(record.get("cache_key")) for record in cache_records
    } != set(expected_cache_keys):
        raise ValueError("extraction summary cache records do not match expected keys")
    if any(record.get("cache_file_exists") is not True for record in cache_records):
        raise ValueError("extraction summary reports a missing graph cache file")
    graph_cache_keys: set[str] = set()
    for index, record in enumerate(graph_records, start=1):
        for graph_kind in ("context", "query", "answer"):
            graph = record.get(graph_kind)
            if not isinstance(graph, dict):
                raise ValueError(f"graph record {index} has no {graph_kind} summary")
            if not SHA256_RE.fullmatch(str(graph.get("sha256", ""))):
                raise ValueError(f"graph record {index} {graph_kind} digest is invalid")
            if not isinstance(graph.get("entities"), int) or not isinstance(
                graph.get("relations"), int
            ):
                raise ValueError(f"graph record {index} {graph_kind} counts are invalid")
        cache = record.get("cache")
        if not isinstance(cache, dict):
            raise ValueError(f"graph record {index} has no cache mapping")
        for graph_kind in ("context", "answer"):
            entry = cache.get(graph_kind)
            key = str((entry or {}).get("cache_key", ""))
            if not SHA256_RE.fullmatch(key):
                raise ValueError(f"graph record {index} {graph_kind} cache key is invalid")
            graph_cache_keys.add(key)
        query_entry = cache.get("query")
        if query_entry is not None:
            key = str(query_entry.get("cache_key", ""))
            if not SHA256_RE.fullmatch(key):
                raise ValueError(f"graph record {index} query cache key is invalid")
            graph_cache_keys.add(key)
    _expect(graph_cache_keys, set(expected_cache_keys), "3-QA graph cache keys")
    for record in cache_records:
        cache_key = str(record.get("cache_key"))
        file_name = str(record.get("cache_file", ""))
        _expect(file_name, f"{cache_key}.json", f"graph cache {cache_key} file name")
        envelope = artifact.json(f"cache/kg/{file_name}")
        if set(envelope) != {"protocol", "cache_key", "graph", "graph_sha256"}:
            raise ValueError(f"graph cache {cache_key} envelope fields are invalid")
        _expect(envelope.get("protocol"), "hallu-kg-cache-v2", f"graph cache {cache_key} protocol")
        _expect(envelope.get("cache_key"), cache_key, f"graph cache {cache_key} envelope")
        graph = envelope.get("graph")
        if not isinstance(graph, dict) or set(graph) != {"entities", "relations"}:
            raise ValueError(f"graph cache {cache_key} graph payload is invalid")
        entities = graph.get("entities")
        relations = graph.get("relations")
        if (
            not isinstance(entities, list)
            or entities != sorted(set(entities))
            or any(not isinstance(value, str) for value in entities)
            or not isinstance(relations, list)
            or relations != sorted(relations)
            or any(
                not isinstance(relation, list)
                or len(relation) != 3
                or any(not isinstance(value, str) for value in relation)
                for relation in relations
            )
        ):
            raise ValueError(f"graph cache {cache_key} is not canonical")
        entity_set = set(entities)
        if any(
            relation[0] not in entity_set or relation[2] not in entity_set
            for relation in relations
        ):
            raise ValueError(
                f"graph cache {cache_key} contains a relation endpoint outside entities"
            )
        canonical = json.dumps(
            graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        _expect(envelope.get("graph_sha256"), digest, f"graph cache {cache_key} digest")

    audit_lines = [
        line for line in artifact.raw("cache/cluster-audit.jsonl").decode("utf-8").splitlines()
        if line.strip()
    ]
    if not audit_lines:
        raise ValueError("cluster probe has no clustering audit records")
    audit_keys: set[str] = set()
    for index, line in enumerate(audit_lines, start=1):
        record = json.loads(line)
        _expect(record.get("protocol"), CLUSTER_CONTEXT_PROTOCOL, f"cluster audit {index} protocol")
        _expect(record.get("context_mode"), CLUSTER_CONTEXT_MODE, f"cluster audit {index} mode")
        if not SHA256_RE.fullmatch(str(record.get("source_text_sha256", ""))):
            raise ValueError(f"cluster audit {index} source hash is invalid")
        _expect(record.get("status"), "ready", f"cluster audit {index} status")
        _expect(record.get("failures"), [], f"cluster audit {index} failures")
        cache_key = str(record.get("cache_key", ""))
        if not SHA256_RE.fullmatch(cache_key):
            raise ValueError(f"cluster audit {index} cache key is invalid")
        audit_keys.add(cache_key)
        for label in ("entities", "predicates"):
            checks = ((record.get("structural_checks") or {}).get(label) or {})
            _expect(checks.get("available"), True, f"cluster audit {index} {label} mapping")
            _expect(
                checks.get("representatives_match_clustered_items"),
                True,
                f"cluster audit {index} {label} representatives",
            )
            _expect(
                checks.get("representatives_are_members"),
                True,
                f"cluster audit {index} {label} representative membership",
            )
            _expect(checks.get("members_cover_raw_items"), True, f"cluster audit {index} {label} coverage")
            _expect(checks.get("members_are_disjoint"), True, f"cluster audit {index} {label} disjointness")
        relation_checks = ((record.get("structural_checks") or {}).get("relations") or {})
        for check in (
            "raw_rows_are_triples",
            "raw_endpoints_in_entities",
            "raw_predicates_in_edges",
            "clustered_rows_are_triples",
            "clustered_endpoints_in_entities",
            "clustered_predicates_in_edges",
            "relations_match_cluster_remap",
        ):
            _expect(
                relation_checks.get(check),
                True,
                f"cluster audit {index} relation check {check}",
            )
    _expect(audit_keys, set(expected_cache_keys), "cluster audit cache coverage")
    if artifact.raw("strict/failed_extractions.jsonl", max_bytes=1024) != b"":
        raise ValueError("cluster probe has failed KG extractions")
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
