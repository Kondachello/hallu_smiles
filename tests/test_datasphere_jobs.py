"""Offline contracts for shared-asset DataSphere plumbing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REVISION = "a" * 40


def _shared_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    shared = tmp_path / "shared"
    model = shared / "models" / "meta-llama-meta-llama-3-1-8b-instruct" / REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    files = [
        {"path": "config.json", "bytes": (model / "config.json").stat().st_size},
        {
            "path": "model-00001-of-00001.safetensors",
            "bytes": (model / "model-00001-of-00001.safetensors").stat().st_size,
        },
    ]
    (model / "model-manifest.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION, "files": files,
    }), encoding="utf-8")
    (model / ".hallu_smiles_model_ready").write_text("ready\n", encoding="utf-8")
    active = model.parent / "active-model.json"
    active.write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION, "model_dir": REVISION,
    }), encoding="utf-8")

    data = shared / "ragtruth"
    data.mkdir()
    for name in ("source_info.jsonl", "response.jsonl"):
        (data / name).write_text('{"id": 1}\n', encoding="utf-8")
    (data / "ragtruth-manifest.json").write_text(json.dumps({"files": [
        {"path": name, "bytes": (data / name).stat().st_size}
        for name in ("source_info.jsonl", "response.jsonl")
    ]}), encoding="utf-8")
    return shared, model, data


def test_shared_asset_check_and_active_model_resolution(tmp_path):
    shared, model, data = _shared_assets(tmp_path)
    report = tmp_path / "report.json"
    checked = subprocess.run([
        sys.executable, str(SCRIPTS / "check_datasphere_shared_assets.py"),
        "--model-path", str(model), "--data-dir", str(data),
        "--model-id", MODEL_ID, "--report", str(report),
    ], check=True, text=True, capture_output=True)
    assert json.loads(checked.stdout)["model_revision"] == REVISION
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "ready"

    resolved = subprocess.run([
        sys.executable, str(SCRIPTS / "resolve_datasphere_shared_model.py"),
        "--shared-root", str(shared), "--model-id", MODEL_ID,
    ], check=True, text=True, capture_output=True)
    assert Path(resolved.stdout.strip()) == model


def test_runtime_config_keeps_every_mutable_path_in_job_work_dir(tmp_path):
    output = tmp_path / "runtime.yaml"
    work_dir = tmp_path / "job-output"
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_runtime_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--output", str(output),
        "--model-id", MODEL_ID, "--api-base", "http://127.0.0.1:8000/v1",
        "--data-dir", "/read-only/ragtruth", "--work-dir", str(work_dir),
    ], check=True)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["data"]["dir"] == "/read-only/ragtruth"
    assert config["cache_dir"] == str(work_dir / "cache" / "kg")
    assert config["relation_verifier"]["cache_dir"] == str(work_dir / "cache" / "verdicts")
    assert config["llm"]["max_tokens"] == 256
    assert config["llm"]["concurrency"] == 1
    assert config["llm"]["request_timeout_s"] == 90
    assert config["llm"]["vllm_guided_json"] is False
    assert config["extraction"]["serial_chunking"] is True
    assert config["extraction"]["cluster_max_items"] is None

    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_runtime_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--output", str(output),
        "--model-id", MODEL_ID, "--api-base", "http://127.0.0.1:8000/v1",
        "--data-dir", "/read-only/ragtruth", "--work-dir", str(work_dir),
        "--disable-clustering",
    ], check=True)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["extraction"]["cluster"] is False

    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_runtime_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--output", str(output),
        "--model-id", MODEL_ID, "--api-base", "http://127.0.0.1:8000/v1",
        "--data-dir", "/read-only/ragtruth", "--work-dir", str(work_dir),
        "--explicit-clustering",
        "--vllm-guided-json",
    ], check=True)
    explicit = yaml.safe_load(output.read_text(encoding="utf-8"))["extraction"]
    assert explicit["cluster"] is True
    assert explicit["explicit_clustering"] is True
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["llm"]["vllm_guided_json"] is True


def test_gpu_job_template_is_pinned_and_has_no_gpu_time_download_or_pip(tmp_path):
    rendered = tmp_path / "qa-pilot.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
        "--kind", "qa-pilot-g1", "--commit", "f" * 40,
        "--run-id", "new-metrics-20260716", "--output", str(rendered),
    ], check=True)
    text = rendered.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "__" not in text
    assert config["cloud-instance-types"] == ["g1.1"]
    assert config["working-storage"]["size"] == "100Gb"
    assert "timeout --signal=TERM --kill-after=60s 10800" in config["cmd"]
    assert config["outputs"] == [{"qa-pilot-new-metrics-20260716.tar.gz": "ARTIFACT_ARCHIVE"}]
    assert "vllm==0.6.3.post1" in (ROOT / "requirements.datasphere.txt").read_text(encoding="utf-8")
    assert "transformers==4.45.2" in (ROOT / "requirements.datasphere.txt").read_text(encoding="utf-8")
    assert "lm-format-enforcer==0.10.6" in (ROOT / "requirements.datasphere.txt").read_text(encoding="utf-8")
    assert "outlines==0.0.46" in (ROOT / "requirements.datasphere.txt").read_text(encoding="utf-8")
    assert "pydantic==2.10.6" in (ROOT / "requirements.datasphere.txt").read_text(encoding="utf-8")

    runner = (SCRIPTS / "run_datasphere_qa_pilot.sh").read_text(encoding="utf-8")
    assert "huggingface-cli download" not in runner
    assert "pip install" not in runner
    assert "--relation-mode strict" in runner
    assert "--relation-mode support" in runner
    assert "check_datasphere_gpu_runtime.py" in runner
    assert "check_datasphere_vllm_completion.py" in runner
    assert "check_datasphere_vllm_guided_json.py" in runner
    assert "patch_datasphere_lmfe_bool_schema.py" in runner
    assert "check_datasphere_outlines_backend.py" in runner
    assert "check_datasphere_kggen_probe.py" in runner
    assert "check_datasphere_qa_reference_probe.py" in runner
    assert "KGGEN_MAX_TOKENS" in runner
    assert "KGGEN_CLUSTER_MAX_ITEMS" in runner
    assert "--disable-clustering" not in runner
    assert "--explicit-clustering" in runner
    assert "  --cluster \\" in runner
    assert "  --vllm-guided-json \\" in runner
    assert "KGGEN_CONCURRENCY" in runner
    assert "--serial-chunking" in runner
    assert "--guided-decoding-backend" in runner
    assert "--vllm-guided-json" in runner
    assert "require_complete_extraction" in runner
    assert 'GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-outlines}"' in runner
    assert "lm-format-enforcer" in runner
    assert "datasphere/runtime_shims" in runner
    assert "LITELLM_LOCAL_MODEL_COST_MAP" in runner
    assert "run_extraction_with_gpu_watchdog" in runner
    assert "GPU_IDLE_ABORT_SECONDS" in runner
    assert "--stage extract" in runner
    assert "--relation-mode strict --qa-pilot-manifest \"$MANIFEST\"" in runner
    assert "[extract] response:start" in (ROOT / "run.py").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_datasphere_qa_pilot.sh")], check=True)


def test_vllm_guided_json_adapter_inlines_nested_dspy_schema_without_response_format(monkeypatch):
    from src.dspy_adapter import inline_local_json_schema_refs, vllm_guided_json_adapter
    from dspy.adapters.chat_adapter import ChatAdapter

    nested_schema = {
        "$defs": {
            "Relation": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"relations": {"type": "array", "items": {"$ref": "#/$defs/Relation"}}},
        "required": ["relations"],
        "additionalProperties": False,
    }

    class OutputModel:
        @staticmethod
        def model_json_schema():
            return nested_schema

    monkeypatch.setattr(
        "dspy.adapters.json_adapter._get_structured_outputs_response_format",
        lambda signature: OutputModel,
    )

    captured = {}

    def fake_call(self, lm, lm_kwargs, signature, demos, inputs):
        captured.update(lm_kwargs)
        return [{"relations": []}]

    monkeypatch.setattr(ChatAdapter, "__call__", fake_call)
    result = vllm_guided_json_adapter()(object(), {}, object(), [], {"text": "x"})

    assert result == [{"relations": []}]
    assert "response_format" not in captured
    assert captured["extra_body"]["guided_json"]["required"] == ["relations"]
    schema = captured["extra_body"]["guided_json"]
    assert "$defs" not in schema
    assert schema["properties"]["relations"]["items"]["required"] == [
        "subject", "predicate", "object"
    ]
    assert schema["properties"]["relations"]["items"]["additionalProperties"] is False

    original = {
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/T"}},
    }
    flattened = inline_local_json_schema_refs(original)
    assert original["properties"]["value"] == {"$ref": "#/$defs/T"}
    assert flattened == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }


def test_cpu_preflight_uses_the_same_locked_runtime_and_import_check(tmp_path):
    rendered = tmp_path / "preflight.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
        "--kind", "preflight", "--commit", "f" * 40,
        "--run-id", "new-metrics-20260717", "--output", str(rendered),
    ], check=True)
    config = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert config["cloud-instance-types"] == ["c1.4"]
    assert config["working-storage"]["size"] == "100Gb"
    assert "check_datasphere_runtime_dependencies.py" in config["cmd"]
    assert "LITELLM_LOCAL_MODEL_COST_MAP=true" in config["cmd"]
    dependency_check = (SCRIPTS / "check_datasphere_runtime_dependencies.py").read_text(encoding="utf-8")
    assert '"lm-format-enforcer": "0.10.6"' in dependency_check
    assert "JsonSchemaParser" in dependency_check
    assert '"additionalProperties": False' in dependency_check
    assert "patch_datasphere_lmfe_bool_schema.py" in config["cmd"]
    assert "check_datasphere_outlines_backend.py" in config["cmd"]
    assert "datasphere/runtime_shims" in config["cmd"]
    assert "if printenv PYTHONPATH >/dev/null" in config["cmd"]
    assert (ROOT / "requirements.datasphere.preflight.txt").read_text(encoding="utf-8") == (
        ROOT / "requirements.datasphere.txt"
    ).read_text(encoding="utf-8")


def test_outlines_backend_uses_the_checked_in_pyairports_compatibility_shim():
    shim = ROOT / "datasphere" / "runtime_shims" / "pyairports" / "airports.py"
    checker = (SCRIPTS / "check_datasphere_outlines_backend.py").read_text(encoding="utf-8")

    assert shim.is_file()
    assert "AIRPORT_LIST" in shim.read_text(encoding="utf-8")
    assert 'EXPECTED_OUTLINES_VERSION = "0.0.46"' in checker
    assert "from pyairports.airports import AIRPORT_LIST" in checker
    assert "from outlines.integrations.vllm import JSONLogitsProcessor" in checker
    assert "compile flattened nested KGGen Relation schema" in checker


def test_vllm_guided_json_probe_uses_the_real_nested_relation_contract():
    checker = (SCRIPTS / "check_datasphere_vllm_guided_json.py").read_text(encoding="utf-8")

    assert '"$ref": "#/$defs/Relation"' in checker
    assert '"subject", "predicate", "object"' in checker
    assert "inline_local_json_schema_refs" in checker


def test_cluster_probe_is_bounded_but_keeps_kggen_clustering(tmp_path):
    rendered = tmp_path / "cluster-probe.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
        "--kind", "cluster-probe-g1", "--commit", "f" * 40,
        "--run-id", "cluster-probe-20260717", "--output", str(rendered),
    ], check=True)
    config = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert config["cloud-instance-types"] == ["g1.1"]
    assert "export QA_PILOT_LIMIT=3" in config["cmd"]
    assert "timeout --signal=TERM --kill-after=60s 3600" in config["cmd"]
    assert "--disable-clustering" not in config["cmd"]
    assert 'pilot.stdout.log' in config["cmd"]
    assert 'pilot.stderr.log' in config["cmd"]
    assert config["outputs"] == [{"cluster-probe-cluster-probe-20260717.tar.gz": "ARTIFACT_ARCHIVE"}]


def test_gpu_job_archives_artifacts_when_cancelled(tmp_path):
    rendered = tmp_path / "qa-pilot.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
        "--kind", "qa-pilot-g1", "--commit", "f" * 40,
        "--run-id", "new-metrics-20260717", "--output", str(rendered),
    ], check=True)
    command = yaml.safe_load(rendered.read_text(encoding="utf-8"))["cmd"]
    assert "trap archive_on_exit EXIT" in command
    assert "trap on_signal INT TERM" in command
    assert "export ARTIFACT_ARCHIVE" in command
    assert "tar -C \"$(dirname \"$RUN_ROOT\")\" -czf \"$ARTIFACT_ARCHIVE\"" in command
    assert 'pilot.stdout.log' in command
    assert 'pilot.stderr.log' in command


def test_rendered_jobs_pass_local_cli_guardrails(tmp_path):
    for kind in ("preflight", "cluster-probe-g1", "qa-pilot-g1"):
        rendered = tmp_path / f"{kind}.yaml"
        subprocess.run([
            sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
            "--kind", kind, "--commit", "f" * 40,
            "--run-id", "new-metrics-20260717", "--output", str(rendered),
        ], check=True)
        checked = subprocess.run([
            sys.executable, str(SCRIPTS / "validate_datasphere_job.py"),
            "--job", str(rendered), "--repo-root", str(ROOT),
        ], check=True, text=True, capture_output=True)
        assert "safe for DataSphere CLI submission" in checked.stdout

    subprocess.run(["bash", "-n", str(SCRIPTS / "submit_datasphere_job.sh")], check=True)


def test_stager_defaults_to_gated_llama_and_uses_model_specific_storage():
    text = (SCRIPTS / "stage_datasphere_shared_assets.py").read_text(encoding="utf-8")
    assert 'MODEL_ID_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"' in text
    assert 'shared_root / "models" / _model_family(model_id)' in text
    assert "optional for public models" in text
