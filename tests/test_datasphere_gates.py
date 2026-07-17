"""Offline validation of the mechanical preflight -> 3QA -> 20QA gates."""
from __future__ import annotations

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
        {"status": "ready", "runtime_manifest": manifest},
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
    })
    _write_json(root / "runtime-identity.json", {
        "source_commit": COMMIT,
        "datasphere_docker_image_id": IMAGE,
        "runtime_protocol": PROTOCOL,
        "image_runtime_fingerprint": IMAGE_FINGERPRINT,
        "runtime_fingerprint": runtime_fingerprint,
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
        _write_json(root / name, {"status": "ready"})
    (root / "strict").mkdir(parents=True)
    (root / "strict" / "failed_extractions.jsonl").write_bytes(b"")
    _write_json(root / "qa_pilot_manifest.json", {
        "version": 1,
        "task": "QA",
        "seed": 42,
        "quotas": {"train_sources": 16, "test_sources": 4},
        "records": [
            {"source_id": f"s{index}", "response_id": f"r{index}"}
            for index in range(20)
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
