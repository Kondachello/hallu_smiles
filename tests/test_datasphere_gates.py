"""Offline validation of the mechanical preflight -> 3QA -> 20QA gates."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.check_datasphere_active_jobs import active_gpu_jobs


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_datasphere_gate_artifact.py"
COMMIT = "a" * 40
IMAGE = "b" + "1" * 19
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PROTOCOL = "hallu-datasphere-vllm085-cu118-v1"
IMAGE_FINGERPRINT = "c" * 64
REQUEST_BACKEND = "xgrammar:disable-any-whitespace,no-fallback"
CLUSTER_CONTEXT_MODE = "source_text"
CLUSTER_CONTEXT_PROTOCOL = "kggen-native-source-text-v1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _archive(root: Path, output: Path) -> Path:
    with tarfile.open(output, "w:gz") as handle:
        handle.add(root, arcname="artifact")
    return output


def _preflight_tar(tmp_path: Path) -> Path:
    root = tmp_path / "preflight"
    manifest = {
        "source_commit": COMMIT,
        "runtime_protocol": PROTOCOL,
        "runtime_fingerprint": IMAGE_FINGERPRINT,
    }
    _write_json(
        root / "runtime-dependencies.json",
        {
            "status": "ready",
            "runtime_manifest": manifest,
            "xgrammar_contract": {
                "request_backend": REQUEST_BACKEND,
                "backend_options": ["disable-any-whitespace", "no-fallback"],
                "any_whitespace": False,
            },
        },
    )
    _write_json(
        root / "preflight.json",
        {"status": "ready", "model_id": MODEL, "model_revision": "mrev"},
    )
    _write_json(root / "gate_metadata.json", {
        "state": "completed",
        "mode": "preflight",
        "source_commit": COMMIT,
        "datasphere_docker_image_id": IMAGE,
        "model_id": MODEL,
        "model_revision": "mrev",
        "runtime_protocol": PROTOCOL,
        "image_runtime_fingerprint": IMAGE_FINGERPRINT,
        "runtime_fingerprint": f"{IMAGE}:{IMAGE_FINGERPRINT}",
    })
    return _archive(root, tmp_path / "preflight.tar.gz")


def _cluster_tar(tmp_path: Path) -> Path:
    root = tmp_path / "cluster"
    runtime_fingerprint = f"{IMAGE}:{IMAGE_FINGERPRINT}:generation"
    records = []
    for index in range(4):
        records.append({
            "source_id": f"test-s{index}",
            "response_id": f"test-r{index}",
            "split": "test",
            "y": 0 if index < 2 else 1,
            "gen_model": "fixture",
        })
    for index in range(16):
        records.append({
            "source_id": f"train-s{index:02d}",
            "response_id": f"train-r{index:02d}",
            "split": "train",
            "y": 0 if index < 8 else 1,
            "gen_model": "fixture",
        })
    _write_json(root / "run_metadata.json", {
        "state": "completed",
        "mode": "cluster-runtime-probe",
        "qa_pilot_limit": 3,
        "runs": ["strict-extract"],
        "source_commit": COMMIT,
        "datasphere_docker_image_id": IMAGE,
        "model_id": MODEL,
        "model_revision": "mrev",
        "runtime_fingerprint": runtime_fingerprint,
        "guided_decoding_backend": "xgrammar",
        "guided_decoding_request_backend": REQUEST_BACKEND,
        "xgrammar_any_whitespace": False,
        "cluster_context_mode": CLUSTER_CONTEXT_MODE,
        "cluster_context_protocol": CLUSTER_CONTEXT_PROTOCOL,
    })
    _write_json(root / "runtime-identity.json", {
        "source_commit": COMMIT,
        "datasphere_docker_image_id": IMAGE,
        "runtime_protocol": PROTOCOL,
        "image_runtime_fingerprint": IMAGE_FINGERPRINT,
        "runtime_fingerprint": runtime_fingerprint,
        "guided_decoding_request_backend": REQUEST_BACKEND,
        "cluster_context_mode": CLUSTER_CONTEXT_MODE,
        "cluster_context_protocol": CLUSTER_CONTEXT_PROTOCOL,
        "server_launch": {
            "guided_decoding_request_backend": REQUEST_BACKEND,
            "xgrammar_any_whitespace": False,
            "cluster_context_mode": CLUSTER_CONTEXT_MODE,
        },
    })
    _write_json(root / "runtime-manifest.json", {
        "source_commit": COMMIT,
        "runtime_protocol": PROTOCOL,
        "runtime_fingerprint": IMAGE_FINGERPRINT,
    })
    _write_json(root / "shared-assets-preflight.json", {
        "status": "ready", "model_id": MODEL, "model_revision": "mrev",
    })
    for name in (
        "vllm-response-format-probe.json",
        "kggen-probe.json",
        "verifier-probe.json",
        "qa-reference-probe.json",
    ):
        payload = {
            "status": "ready",
            "guided_decoding_request_backend": REQUEST_BACKEND,
            "xgrammar_any_whitespace": False,
        }
        if name == "qa-reference-probe.json":
            payload.update({
                "cluster_context_mode": CLUSTER_CONTEXT_MODE,
                "cluster_audit": {
                    "protocol": CLUSTER_CONTEXT_PROTOCOL,
                    "context_mode": CLUSTER_CONTEXT_MODE,
                },
            })
        _write_json(root / name, payload)
    cache_keys = [f"{index:x}" * 64 for index in (1, 2, 3)]
    (root / "cache").mkdir(parents=True)
    (root / "cache" / "cluster-audit.jsonl").write_text(
        "".join(
            json.dumps({
                "protocol": CLUSTER_CONTEXT_PROTOCOL,
                "context_mode": CLUSTER_CONTEXT_MODE,
                "cache_key": cache_key,
                "kind": "context",
                "status": "ready",
                "failures": [],
                "source_text_sha256": "d" * 64,
                "structural_checks": {
                    **{
                        label: {
                            "available": True,
                            "representatives_match_clustered_items": True,
                            "members_cover_raw_items": True,
                            "members_are_disjoint": True,
                        }
                        for label in ("entities", "predicates")
                    },
                    "relations": {
                        check: True
                        for check in (
                            "raw_rows_are_triples",
                            "raw_endpoints_in_entities",
                            "raw_predicates_in_edges",
                            "clustered_rows_are_triples",
                            "clustered_endpoints_in_entities",
                            "clustered_predicates_in_edges",
                            "relations_match_cluster_remap",
                        )
                    },
                },
            }) + "\n"
            for cache_key in cache_keys
        ),
        encoding="utf-8",
    )
    for cache_key in cache_keys:
        graph = {"entities": [], "relations": []}
        canonical = json.dumps(
            graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        _write_json(root / "cache" / "kg" / f"{cache_key}.json", {
            "protocol": "hallu-kg-cache-v2",
            "cache_key": cache_key,
            "graph": graph,
            "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        })
    (root / "strict").mkdir(parents=True)
    (root / "strict" / "failed_extractions.jsonl").write_bytes(b"")
    _write_json(root / "qa_pilot_manifest.json", {
        "version": 1,
        "task": "QA",
        "seed": 42,
        "quotas": {"train_sources": 16, "test_sources": 4},
        "records": records,
    })
    prefix = [
        {
            "source_id": record["source_id"],
            "response_id": record["response_id"],
            "split": record["split"],
            "y": record["y"],
        }
        for record in records[:3]
    ]
    _write_json(root / "strict" / "extraction_summary.json", {
        "protocol": "hallu-extraction-summary-v1",
        "status": "ready",
        "expected_records": prefix,
        "completed_records": prefix,
        "expected_sources": 3,
        "references_completed": 3,
        "responses_completed": 3,
        "pairs_completed": 3,
        "failures": [],
        "graph_records": [
            {
                **record,
                **{
                    kind: {"entities": 1, "relations": 1, "sha256": "f" * 64}
                    for kind in ("context", "query", "answer")
                },
                "cache": {
                    "context": {"cache_key": cache_keys[index]},
                    "query": None,
                    "answer": {"cache_key": cache_keys[index]},
                },
            }
            for index, record in enumerate(prefix)
        ],
        "expected_cache_keys": cache_keys,
        "cache_records": [
            {
                "cache_key": cache_key,
                "cache_file": f"{cache_key}.json",
                "cache_file_exists": True,
            }
            for cache_key in cache_keys
        ],
    })
    return _archive(root, tmp_path / "cluster.tar.gz")


@pytest.mark.parametrize(
    "gate,builder",
    [("preflight", _preflight_tar), ("cluster-probe-g1", _cluster_tar)],
)
def test_matching_gate_artifact_is_accepted(tmp_path, gate, builder):
    completed = subprocess.run([
        sys.executable,
        str(VALIDATOR),
        "--gate", gate,
        "--artifact", str(builder(tmp_path)),
        "--commit", COMMIT,
        "--docker-image-id", IMAGE,
        "--model-id", MODEL,
    ], check=True, text=True, capture_output=True)
    assert json.loads(completed.stdout)["status"] == "ready"


def test_gate_artifact_rejects_cross_image_reuse(tmp_path):
    completed = subprocess.run([
        sys.executable,
        str(VALIDATOR),
        "--gate", "cluster-probe-g1",
        "--artifact", str(_cluster_tar(tmp_path)),
        "--commit", COMMIT,
        "--docker-image-id", "b" + "2" * 19,
        "--model-id", MODEL,
    ], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "Docker image" in completed.stderr


def test_active_gpu_guard_handles_single_object_array_and_unknown_status():
    assert active_gpu_jobs({"name": "hallu-qa-g1-old", "status": "SUCCESS"}) == []
    active = active_gpu_jobs([
        {"id": "cpu", "name": "hallu-shared-preflight-x", "status": "EXECUTING"},
        {
            "id": "gpu",
            "name": "hallu-cluster-probe-g1-x",
            "status": "JOB_STATUS_EXECUTING",
        },
    ])
    assert active == [{
        "id": "gpu",
        "name": "hallu-cluster-probe-g1-x",
        "status": "JOB_STATUS_EXECUTING",
    }]
    assert active_gpu_jobs({"name": "hallu-qa-g1-new", "status": "FUTURE_STATE"})
