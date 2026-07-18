"""Offline tests for the API probe gate and safe cache import."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.check_api_contract import run_contract_probe
from scripts.import_probe_cache import import_probe_cache
from scripts.run_api_job import _merge_provider_telemetry, _write_comparison
from scripts.validate_api_probe_artifact import _required_runtime_versions, validate_probe_artifact


COMMIT = "a" * 40
MODEL = "openai/qwen3-8b"
API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _manifest() -> dict:
    rows = []
    for split, count in (("train", 16), ("test", 4)):
        for index in range(count):
            rows.append({
                "source_id": f"{split}-s{index:02d}",
                "response_id": f"{split}-r{index:02d}",
                "split": split,
                "y": index % 2,
                "gen_model": "fixture",
            })
    rows.sort(key=lambda row: (row["split"], row["source_id"], row["response_id"]))
    return {
        "version": 1,
        "task": "QA",
        "seed": 42,
        "quotas": {"train_sources": 16, "test_sources": 4},
        "records": rows,
    }


def _score_fixture(mode: str) -> dict:
    support = mode == "support"
    return {
        "Vc": 2, "Ec": 1, "Vq": 1, "Eq": 0, "Va": 2, "Ea": 1,
        "EG": 1.0,
        "RP": 1.0, "RP_defined": True,
        "RP_strict": 1.0, "RP_strict_defined": True,
        "RP_grounded": 1.0, "RP_grounded_defined": True,
        "RP_entailed_cond": 1.0 if support else None,
        "RP_entailed_cond_defined": support,
        "RP_support": 1.0 if support else None,
        "RP_support_defined": support,
        "support_verified": support,
        "matched_entities": [],
        "ungrounded_entities": [],
        "supported_relations": [["alice", "works at", "acme"]],
        "unsupported_relations": [],
        "relation_audits": [{"status": "aligned"}],
        "unscorable": False,
        "ref_empty": False,
    }


def _audit_fixture(row: dict, mode: str, score: dict) -> dict:
    support = mode == "support"
    return {
        **row,
        "relation_mode": mode,
        "alpha": 0.7,
        "alpha_strict": 0.7,
        "alpha_support": 0.7,
        "EG": 1.0,
        "RP": 1.0,
        "RP_defined": True,
        "RP_strict": 1.0,
        "RP_grounded": 1.0,
        "RP_entailed_cond": 1.0 if support else None,
        "RP_support": 1.0 if support else None,
        "RP_support_defined": support,
        "CFI_strict": 1.0,
        "H_strict": 0.0,
        "CFI_support": 1.0 if support else None,
        "H_support": 0.0 if support else None,
        "graph_sizes": {field: score[field] for field in ("Vc", "Ec", "Vq", "Eq", "Va", "Ea")},
        "relation_audits": score["relation_audits"],
        "probe_diagnostic_alpha_not_tuned": True,
    }


def _fixture_files(*, gate_status: str = "success") -> dict[str, bytes]:
    manifest = _manifest()
    manifest_raw = (json.dumps(manifest, indent=2) + "\n").encode()
    prefix = [
        {key: row[key] for key in ("source_id", "response_id", "split", "y")}
        for row in manifest["records"][:3]
    ]
    gate = {
        "protocol": "hallu-api-probe-v1",
        "kind": "api-probe-c1",
        "status": gate_status,
        "source_commit": COMMIT,
        "model": MODEL,
        "api_base": API_BASE,
        "contract_passed": 3,
        "qa_completed": 3,
        "failed_extractions": 0,
        "cache_replay_provider_calls": 0,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "provider_calls": 4,
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "total_tokens": 60,
    }
    scored: dict[str, bytes] = {}
    for mode in ("strict", "support"):
        score = _score_fixture(mode)
        scored[f"{mode}/scored.jsonl"] = "".join(
            json.dumps({**row, "relation_mode": mode, "score": score}) + "\n"
            for row in prefix
        ).encode()
        for row in prefix:
            scored[f"{mode}/audit/{row['response_id']}.json"] = json.dumps(
                _audit_fixture(row, mode, score)
            ).encode()
    provider_record = {
        "outcome": "success",
        "request_id": "fixture",
        "latency_s": 0.1,
        "http_status": 200,
        "retry_index": 0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "error_type": None,
    }
    graph_payloads = {
        "context": {"entities": ["acme", "alice"], "relations": [["alice", "works at", "acme"]]},
        "query": {"entities": ["alice"], "relations": []},
        "answer": {"entities": ["acme", "alice"], "relations": [["alice", "works at", "acme"]]},
    }
    graph_digests = {
        name: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, payload in graph_payloads.items()
    }
    kg_files = {
        f".cache/kg/{hashlib.sha256(name.encode()).hexdigest()}.json":
            (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        for name, payload in graph_payloads.items()
        if name != "answer"
    }
    verdict_path = ".cache/verdicts/" + "c" * 64 + ".json"
    verdict_data = b'{"verdict":"entailed"}\n'
    cache_inventory = {
        "protocol": "hallu-api-cache-inventory-v1",
        "status": "ready",
        "entries": [
            {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in (*kg_files.items(), (verdict_path, verdict_data))
        ],
    }
    runtime_versions = {"python": "3.11.9", **_required_runtime_versions()}
    usage_summary = {
        "api_calls": 6,
        "requests_total": 10,
        "cache_hits": 4,
        "cache_hit_rate": 0.4,
        "estimated_cost_usd": 0.0,
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "total_tokens": 60,
        "provider_calls": 4,
        "provider_successes": 4,
        "provider_failures": 0,
        "provider_contract_errors": 0,
    }
    result = {
        key: gate[key]
        for key in (
            "contract_passed", "qa_completed", "failed_extractions",
            "cache_replay_provider_calls", "manifest_sha256", "provider_calls",
            "prompt_tokens", "completion_tokens", "total_tokens",
        )
    }
    result["kind"] = "api-probe-c1"
    run_metadata = {
        "protocol": "hallu-api-job-v1",
        "mode": "probe",
        "state": "completed",
        "status": "success",
        "kind": "api-probe-c1",
        "source_commit": COMMIT,
        "model": MODEL,
        "api_base": API_BASE,
        "api_key_env": "FIXTURE_API_KEY",
        "started_at_utc": "2026-07-18T00:00:00Z",
        "completed_at_utc": "2026-07-18T00:01:00Z",
        "elapsed_seconds": 60.0,
        "runtime_versions": runtime_versions,
        "llm_runtime_fingerprint": {
            "protocol": "dashscope-json-object-strict-local-schema-v1",
            "model": MODEL,
            "api_base": API_BASE,
            "temperature": 0.0,
            "max_tokens": 1024,
            "request_timeout_s": 180.0,
            "structured_output_transport": "json_object",
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
            "runtime_versions": runtime_versions,
        },
        "result": result,
    }
    files = {
        "gate_metadata.json": json.dumps(gate).encode(),
        "run_metadata.json": json.dumps(run_metadata).encode(),
        "usage_summary.json": json.dumps(usage_summary).encode(),
        "job.stdout.log": b"fixture probe completed\n",
        "job.stderr.log": b"",
        "qa_pilot_manifest.json": manifest_raw,
        "contract_probe.json": json.dumps({
            "protocol": "hallu-api-json-object-contract-v1",
            "status": "ready",
            "source_id": "15138",
            "transport": "json_object",
            "repair_allowed": False,
            "passed": 3,
            "attempts": [{
                "status": "ready",
                "attempt": n,
                "relations_count": 1,
                "relations": [["Swiss chard", "is similar to", "spinach"]],
            } for n in range(1, 4)],
        }).encode(),
        "synthetic_probe.json": json.dumps({
            "protocol": "hallu-api-synthetic-kggen-v1",
            "status": "ready",
            "cluster": True,
            "official_kggen_clustering": True,
            "entities": 4,
            "relations": 2,
            "semantic_anchors": {"ada_lovelace": True, "marie_curie_warsaw": True},
        }).encode(),
        "verifier_probe.json": json.dumps({
            "protocol": "hallu-api-verifier-probe-v1",
            "status": "ready",
            "verdict": "entailed",
            "evidence": [{"source": "context", "text": "Alice works at Acme."}],
        }).encode(),
        "cache_inventory.json": json.dumps(cache_inventory).encode(),
        "extraction_summary.json": json.dumps({
            "status": "ready",
            "expected_records": prefix,
            "completed_records": prefix,
            "pairs_completed": 3,
            "failures": [],
            "graph_records": [
                {
                    **row,
                    **{
                        name: {
                            "entities": len(payload["entities"]),
                            "relations": len(payload["relations"]),
                            "sha256": graph_digests[name],
                        }
                        for name, payload in graph_payloads.items()
                    },
                }
                for row in prefix
            ],
        }).encode(),
        "failed_extractions.jsonl": b"",
        "cache_replay.json": json.dumps({
            "status": "ready", "qa_completed": 3, "provider_calls": 0, "failures": 0,
        }).encode(),
        "provider_calls.jsonl": (json.dumps(provider_record) + "\n").encode() * 4,
        **kg_files,
        verdict_path: verdict_data,
        **scored,
    }
    return files


def _write_tar(path: Path, files: dict[str, bytes], *, unsafe_name: str | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"probe/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if unsafe_name:
            data = b"unsafe"
            info = tarfile.TarInfo(unsafe_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_validate_probe_artifact_accepts_complete_gate(tmp_path):
    artifact = _write_tar(tmp_path / "probe.tar.gz", _fixture_files())
    gate = validate_probe_artifact(
        artifact,
        expected_commit=COMMIT,
        expected_model=MODEL,
        expected_api_base=API_BASE,
    )
    assert gate["status"] == "success"
    assert gate["qa_completed"] == 3


@pytest.mark.parametrize("change", ["failed_gate", "nonempty_failures", "provider_replay"])
def test_validate_probe_artifact_rejects_incomplete_evidence(tmp_path, change):
    files = _fixture_files(gate_status="error" if change == "failed_gate" else "success")
    if change == "nonempty_failures":
        files["failed_extractions.jsonl"] = b'{"error":"bad relation schema"}\n'
    if change == "provider_replay":
        replay = json.loads(files["cache_replay.json"])
        replay["provider_calls"] = 1
        files["cache_replay.json"] = json.dumps(replay).encode()
    artifact = _write_tar(tmp_path / "probe.tar.gz", files)
    with pytest.raises(ValueError):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_rejects_shallow_or_wrong_mode_scores(tmp_path):
    files = _fixture_files()
    manifest = _manifest()
    prefix = manifest["records"][:3]
    files["strict/scored.jsonl"] = "".join(
        json.dumps({**row, "relation_mode": "strict", "score": {"EG": 1.0}}) + "\n"
        for row in prefix
    ).encode()
    shallow = _write_tar(tmp_path / "shallow.tar.gz", files)
    with pytest.raises(ValueError, match="ScoreResult fields mismatch"):
        validate_probe_artifact(shallow)

    files = _fixture_files()
    score = _score_fixture("support")
    score["support_verified"] = False
    files["support/scored.jsonl"] = "".join(
        json.dumps({**row, "relation_mode": "support", "score": score}) + "\n"
        for row in prefix
    ).encode()
    wrong_mode = _write_tar(tmp_path / "wrong-mode.tar.gz", files)
    with pytest.raises(ValueError, match="support verification"):
        validate_probe_artifact(wrong_mode)


def test_validate_probe_artifact_rejects_wrong_contract_shape(tmp_path):
    files = _fixture_files()
    contract = json.loads(files["contract_probe.json"])
    contract["attempts"][1]["relations"] = [{
        "subject": "Swiss chard", "predicate": "is similar to", "object": "spinach",
    }]
    files["contract_probe.json"] = json.dumps(contract).encode()
    artifact = _write_tar(tmp_path / "bad-contract.tar.gz", files)
    with pytest.raises(ValueError, match="three-field string triple"):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_requires_complete_graph_records(tmp_path):
    files = _fixture_files()
    extraction = json.loads(files["extraction_summary.json"])
    extraction["graph_records"] = []
    files["extraction_summary.json"] = json.dumps(extraction).encode()
    artifact = _write_tar(tmp_path / "missing-graphs.tar.gz", files)
    with pytest.raises(ValueError, match="exactly three graph records"):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_rejects_noncanonical_kg_cache(tmp_path):
    files = _fixture_files()
    kg_path = next(path for path in files if path.startswith(".cache/kg/"))
    payload = json.loads(files[kg_path])
    payload["relations"][0][2] = "entity absent from graph"
    raw = (json.dumps(payload) + "\n").encode()
    files[kg_path] = raw
    inventory = json.loads(files["cache_inventory.json"])
    entry = next(item for item in inventory["entries"] if item["path"] == kg_path)
    entry["bytes"] = len(raw)
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    files["cache_inventory.json"] = json.dumps(inventory).encode()
    artifact = _write_tar(tmp_path / "bad-kg-cache.tar.gz", files)
    with pytest.raises(ValueError, match="endpoints must belong to entities"):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_rejects_swallowed_provider_contract_error(tmp_path):
    files = _fixture_files()
    records = [json.loads(line) for line in files["provider_calls.jsonl"].splitlines()]
    records[0]["outcome"] = "contract_error"
    records[0]["error_type"] = "StructuredOutputSchemaError"
    files["provider_calls.jsonl"] = "".join(
        json.dumps(record) + "\n" for record in records
    ).encode()
    artifact = _write_tar(tmp_path / "contract-error.tar.gz", files)
    with pytest.raises(ValueError, match="cannot contain a swallowed.*contract_error"):
        validate_probe_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("http_status"), "telemetry fields mismatch"),
        (lambda row: row.update(request_id=None), "no request ID"),
        (lambda row: row.update(http_status=201), "not HTTP 200"),
        (lambda row: row.update(total_tokens=99), "inconsistent token usage"),
    ],
)
def test_validate_probe_artifact_requires_complete_provider_telemetry(
    tmp_path, mutation, message
):
    files = _fixture_files()
    records = [json.loads(line) for line in files["provider_calls.jsonl"].splitlines()]
    mutation(records[0])
    files["provider_calls.jsonl"] = "".join(
        json.dumps(record) + "\n" for record in records
    ).encode()
    artifact = _write_tar(tmp_path / "bad-provider-telemetry.tar.gz", files)
    with pytest.raises(ValueError, match=message):
        validate_probe_artifact(artifact)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("run_metadata.json", "run_metadata"),
        ("usage_summary.json", "usage_summary"),
        ("job.stdout.log", "job.stdout.log"),
        ("job.stderr.log", "job.stderr.log"),
    ],
)
def test_validate_probe_artifact_requires_metadata_usage_and_logs(tmp_path, missing, message):
    files = _fixture_files()
    del files[missing]
    artifact = _write_tar(tmp_path / f"missing-{missing.replace('/', '-')}.tar.gz", files)
    with pytest.raises(ValueError, match=message):
        validate_probe_artifact(artifact)


@pytest.mark.parametrize("change", ["state", "model", "version", "fingerprint", "usage"])
def test_validate_probe_artifact_rejects_inconsistent_runtime_evidence(tmp_path, change):
    files = _fixture_files()
    metadata = json.loads(files["run_metadata.json"])
    usage = json.loads(files["usage_summary.json"])
    if change == "state":
        metadata["state"] = "started"
    elif change == "model":
        metadata["model"] = "different/provider-model"
    elif change == "version":
        metadata["runtime_versions"]["kg-gen"] = "0.0.0"
    elif change == "fingerprint":
        metadata["llm_runtime_fingerprint"]["extra_body"]["enable_thinking"] = True
    else:
        usage["prompt_tokens"] += 1
    files["run_metadata.json"] = json.dumps(metadata).encode()
    files["usage_summary.json"] = json.dumps(usage).encode()
    artifact = _write_tar(tmp_path / f"inconsistent-{change}.tar.gz", files)
    with pytest.raises(ValueError):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_rejects_duplicate_log_suffix(tmp_path):
    artifact = _write_tar(
        tmp_path / "duplicate-log.tar.gz",
        _fixture_files(),
        unsafe_name="duplicate/job.stdout.log",
    )
    with pytest.raises(ValueError, match="found 2"):
        validate_probe_artifact(artifact)


def test_validate_probe_artifact_rejects_traversal_and_secret(tmp_path):
    unsafe = _write_tar(tmp_path / "unsafe.tar.gz", _fixture_files(), unsafe_name="../escape")
    with pytest.raises(ValueError, match="unsafe archive member"):
        validate_probe_artifact(unsafe)

    files = _fixture_files()
    files["leak.txt"] = b"top-secret-value"
    leaked = _write_tar(tmp_path / "leaked.tar.gz", files)
    with pytest.raises(ValueError, match="secret leaked"):
        validate_probe_artifact(leaked, secret="top-secret-value")


def test_import_probe_cache_copies_only_validated_cache_and_manifest(tmp_path):
    artifact = _write_tar(tmp_path / "probe.tar.gz", _fixture_files())
    destination = tmp_path / "pilot"
    report = import_probe_cache(
        artifact,
        destination,
        expected_commit=COMMIT,
        expected_model=MODEL,
        expected_api_base=API_BASE,
    )
    assert report["imported"] == {"kg": 2, "verdicts": 1}
    assert (destination / "qa_pilot_manifest.json").is_file()
    assert len(list((destination / ".cache/kg").glob("*.json"))) == 2
    assert len(list((destination / ".cache/verdicts").glob("*.json"))) == 1
    assert not (destination / "gate_metadata.json").exists()

    # Idempotent reuse is allowed, but a same-key/different-content cache is not.
    import_probe_cache(artifact, destination)
    cache = next((destination / ".cache/kg").glob("*.json"))
    cache.write_text("conflict", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting probe cache"):
        import_probe_cache(artifact, destination)


def test_import_probe_cache_rejects_destination_symlink_parents(tmp_path):
    artifact = _write_tar(tmp_path / "probe.tar.gz", _fixture_files())
    real = tmp_path / "real"
    real.mkdir()
    direct_link = tmp_path / "direct-link"
    direct_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        import_probe_cache(artifact, direct_link)

    destination = tmp_path / "pilot"
    destination.mkdir()
    cache_target = tmp_path / "outside-cache"
    cache_target.mkdir()
    (destination / ".cache").symlink_to(cache_target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink is forbidden"):
        import_probe_cache(artifact, destination)


def _swiss_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "source_info.jsonl").write_text(json.dumps({
        "source_id": "15138",
        "source_info": {"passages": "Swiss chard is similar to spinach and beetroot."},
    }) + "\n", encoding="utf-8")
    return data_dir


def test_contract_probe_uses_exact_path_three_times(tmp_path):
    class Extractor:
        def __init__(self):
            self.calls = []

        def relation_contract(self, text, entities):
            self.calls.append((text, entities))
            return {("Swiss chard", "is similar to", "spinach")}

    extractor = Extractor()
    report = run_contract_probe(extractor, _swiss_fixture(tmp_path))
    assert report["passed"] == 3
    assert report["repair_allowed"] is False
    assert [attempt["status"] for attempt in report["attempts"]] == ["ready"] * 3
    assert len(extractor.calls) == 3


def test_contract_probe_rejects_bare_or_semantically_wrong_relation(tmp_path):
    class BareRelationExtractor:
        def relation_contract(self, _text, _entities):
            # A dict is the historical bad root. Iterating it yields a string,
            # which must never be repaired or wrapped into {"relations": [...]}.
            return {"subject": "Swiss chard", "predicate": "is", "object": "spinach"}

    with pytest.raises(RuntimeError, match="attempt 1 failed"):
        run_contract_probe(BareRelationExtractor(), _swiss_fixture(tmp_path))


def test_provider_telemetry_merge_is_allowlist_only_and_counts_tokens(tmp_path):
    record = {
        "outcome": "success", "request_id": "r", "latency_s": 0.1,
        "http_status": 200, "retry_index": 0, "prompt_tokens": 7,
        "completion_tokens": 3, "total_tokens": 10, "error_type": None,
    }
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(record) + "\n", encoding="utf-8")
    second.write_text(json.dumps(record) + "\n", encoding="utf-8")
    destination = tmp_path / "provider_calls.jsonl"
    assert _merge_provider_telemetry([first, second], destination) == {
        "provider_calls": 2,
        "prompt_tokens": 14,
        "completion_tokens": 6,
        "total_tokens": 20,
    }
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 2
    leaked = {**record, "messages": [{"role": "user", "content": "secret prompt"}]}
    first.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="allowlist-only"):
        _merge_provider_telemetry([first], destination)


def test_strict_support_comparison_requires_same_twenty_records(tmp_path):
    fieldnames = [
        "source_id", "response_id", "split", "y", "H_strict", "H_support",
    ]
    import csv

    for mode, auc, f1 in (("strict", 0.5, 0.4), ("support", 0.75, 0.6)):
        directory = tmp_path / mode
        directory.mkdir()
        with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(20):
                writer.writerow({
                    "source_id": f"s{index}", "response_id": f"r{index}",
                    "split": "train" if index < 16 else "test", "y": index % 2,
                    "H_strict": 0.1, "H_support": 0.2,
                })
        with (directory / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["overall_AUC_exclude_unscorable", "overall_F1"]
            )
            writer.writeheader()
            writer.writerow({"overall_AUC_exclude_unscorable": auc, "overall_F1": f1})
        (directory / "tuning.json").write_text(json.dumps({
            "alpha": 0.7, "theta": 0.4, "tau_e": 0.9, "tau_r": 0.75,
        }), encoding="utf-8")
    comparison = _write_comparison(tmp_path, "d" * 64)
    assert comparison["records"] == 20
    assert comparison["delta_support_minus_strict"]["test_auc"] == 0.25
    assert json.loads((tmp_path / "comparison.json").read_text())["status"] == "ready"

    support_metrics = tmp_path / "support" / "metrics.csv"
    rows = list(csv.DictReader(support_metrics.open(encoding="utf-8")))
    rows[0]["response_id"] = "different"
    with support_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="same 20 QA"):
        _write_comparison(tmp_path, "d" * 64)
