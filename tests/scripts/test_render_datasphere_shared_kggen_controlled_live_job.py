from __future__ import annotations
import subprocess, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
IMAGE="ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu@sha256:" + "a"*64
def test_rendered_controlled_live_job_is_cpu_only_and_has_no_secret(tmp_path):
 out=tmp_path/"job.yaml"
 subprocess.run([sys.executable, str(ROOT/"scripts/render_datasphere_shared_kggen_one_instance_controlled_live_job.py"), "--commit","a"*40,"--run-id","controlled-live-test","--response-id","6845","--gateway-url","https://gateway.example.test","--docker-image",IMAGE,"--output",str(out)],check=True)
 text=out.read_text(encoding="utf-8"); job=yaml.safe_load(text)
 assert job["cloud-instance-types"] == ["c1.4"]
 assert "HALLU_GATEWAY_URL" in text and "HALLU_GATEWAY_API_KEY" not in text
 assert "mock" not in text.lower() and "g1.1" not in text and "huggingface-cli" not in text.lower()
 assert 'DATASPHERE_DOCKER_IMAGE_ID="' + IMAGE + '"' in str(job["cmd"])
