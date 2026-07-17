"""Offline contracts for immutable external DataSphere runtime images."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.datasphere_runtime_image import require_runtime_image
from scripts.resolve_datasphere_runtime_image import resolve


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
COMMIT = "a" * 40
OCI_IMAGE = "ghcr.io/kondachello/hallu-smiles-datasphere@sha256:" + "b" * 64


def test_runtime_image_identity_rejects_mutable_registry_tags():
    assert require_runtime_image("b" + "1" * 19, registry=False) == "b" + "1" * 19
    assert require_runtime_image(OCI_IMAGE, registry=True) == OCI_IMAGE
    with pytest.raises(ValueError, match="pinned"):
        require_runtime_image(
            "ghcr.io/kondachello/hallu-smiles-datasphere:latest", registry=True
        )


def test_public_ghcr_tag_is_verified_and_resolved_to_body_digest(monkeypatch):
    body = b'{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json"}'
    digest = "sha256:" + hashlib.sha256(body).hexdigest()

    class Response:
        def __init__(self, payload: bytes, headers: dict[str, str] | None = None):
            self.payload = payload
            self.headers = headers or {}

        def read(self) -> bytes:
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):  # noqa: ARG001
        if "/token?" in request.full_url:
            return Response(json.dumps({"token": "anonymous"}).encode())
        assert request.headers["Authorization"] == "Bearer anonymous"
        assert request.full_url.endswith("/manifests/" + COMMIT)
        return Response(body, {"Docker-Content-Digest": digest})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert resolve(
        "ghcr.io/kondachello/hallu-smiles-datasphere", COMMIT
    ) == f"ghcr.io/kondachello/hallu-smiles-datasphere@{digest}"


def test_registry_digest_renders_and_validates_as_datasphere_docker_mapping(tmp_path):
    rendered = tmp_path / "preflight.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "render_datasphere_job.py"),
            "--kind",
            "preflight",
            "--commit",
            COMMIT,
            "--docker-image",
            OCI_IMAGE,
            "--run-id",
            "registry-runtime",
            "--output",
            str(rendered),
        ],
        check=True,
    )
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert document["env"] == {"docker": {"image": OCI_IMAGE}}
    assert f'DATASPHERE_DOCKER_IMAGE_ID="{OCI_IMAGE}"' in document["cmd"]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_datasphere_job.py"),
            "--job",
            str(rendered),
            "--repo-root",
            str(ROOT),
        ],
        check=True,
    )


def test_renderer_rejects_mutable_registry_tag(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "render_datasphere_job.py"),
            "--kind",
            "preflight",
            "--commit",
            COMMIT,
            "--docker-image",
            "ghcr.io/kondachello/hallu-smiles-datasphere:latest",
            "--run-id",
            "mutable-runtime",
            "--output",
            str(tmp_path / "bad.yaml"),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "pinned" in completed.stderr


def test_remote_build_workflow_never_packages_model_or_dataset():
    workflow = (ROOT / ".github/workflows/datasphere-runtime-image.yml").read_text(
        encoding="utf-8"
    )
    assert "docker buildx build" in workflow
    assert "--platform linux/amd64" in workflow
    assert "--provenance=false" in workflow
    assert "docker buildx imagetools inspect" in workflow
    assert "Meta-Llama" not in workflow
    assert "RAGTruth" not in workflow
