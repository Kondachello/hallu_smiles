#!/usr/bin/env python3
"""Validate the immutable artifact that gates the 20-QA Alibaba API pilot.

The validator deliberately reads archive members without extracting them.  It
is used both by the submit-side CLI and by ``import_probe_cache.py`` inside a
pilot Job, so an untrusted or accidentally truncated tar cannot write outside
the selected working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROTOCOL = "hallu-api-probe-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSONL_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = {
    "Vc", "Ec", "Vq", "Eq", "Va", "Ea",
    "EG", "RP", "RP_defined", "RP_strict", "RP_strict_defined",
    "RP_grounded", "RP_grounded_defined", "RP_entailed_cond",
    "RP_entailed_cond_defined", "RP_support", "RP_support_defined",
    "support_verified", "matched_entities", "ungrounded_entities",
    "supported_relations", "unsupported_relations", "relation_audits",
    "unscorable", "ref_empty",
}


def _required_runtime_versions() -> dict[str, str]:
    pins: dict[str, str] = {}
    path = ROOT / "requirements.datasphere.api.txt"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        if "==" not in line:
            raise ValueError(f"unpinned DataSphere API requirement: {line}")
        distribution, version = line.split("==", 1)
        if not distribution or not version or distribution in pins:
            raise ValueError(f"invalid or duplicate DataSphere API requirement: {line}")
        pins[distribution] = version
    if not pins:
        raise ValueError("DataSphere API requirements contain no pinned distributions")
    return pins


class ProbeArtifact:
    """Read-only, traversal-safe view of a probe tarball."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"probe artifact does not exist: {self.path}")
        if self.path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError(f"probe artifact is unexpectedly large: {self.path.stat().st_size}")
        self.tar = tarfile.open(self.path, mode="r:*")
        self._members: dict[str, tarfile.TarInfo] = {}
        total = 0
        try:
            for member in self.tar.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise ValueError(f"unsafe archive member: {member.name!r}")
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"links and devices are forbidden in probe artifacts: {member.name!r}")
                if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValueError(f"archive member has invalid size: {member.name!r}")
                if member.name in self._members:
                    raise ValueError(f"duplicate archive member: {member.name!r}")
                self._members[member.name] = member
                if member.isfile():
                    total += member.size
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("unpacked probe artifact is unexpectedly large")
        except Exception:
            self.tar.close()
            raise

    def close(self) -> None:
        self.tar.close()

    def __enter__(self) -> "ProbeArtifact":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def one(self, suffix: str) -> tarfile.TarInfo:
        suffix = suffix.lstrip("/")
        matches = [
            member
            for name, member in self._members.items()
            if member.isfile() and (name == suffix or name.endswith("/" + suffix))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one regular {suffix!r} in {self.path}, found {len(matches)}"
            )
        return matches[0]

    def raw(self, suffix: str, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
        member = self.one(suffix)
        if member.size > max_bytes:
            raise ValueError(f"archive member {suffix!r} is unexpectedly large: {member.size}")
        handle = self.tar.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read archive member {suffix!r}")
        data = handle.read(max_bytes + 1)
        if len(data) != member.size:
            raise ValueError(f"archive member {suffix!r} is truncated")
        return data

    def json(self, suffix: str) -> dict[str, Any]:
        value = json.loads(self.raw(suffix).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{suffix!r} must contain a JSON object")
        return value

    def jsonl(self, suffix: str) -> list[dict[str, Any]]:
        raw = self.raw(suffix, max_bytes=MAX_JSONL_BYTES)
        records: list[dict[str, Any]] = []
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{suffix}:{number} must be a JSON object")
            records.append(value)
        return records

    def regular_members(self) -> Iterable[tarfile.TarInfo]:
        return (member for member in self._members.values() if member.isfile())


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def _stream_contains(handle: Any, needle: bytes) -> bool:
    overlap = b""
    keep = max(0, len(needle) - 1)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return False
        combined = overlap + chunk
        if needle in combined:
            return True
        overlap = combined[-keep:] if keep else b""


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    _expect(manifest.get("version"), 1, "manifest version")
    _expect(manifest.get("task"), "QA", "manifest task")
    _expect(manifest.get("seed"), 42, "manifest seed")
    _expect(
        manifest.get("quotas"),
        {"train_sources": 16, "test_sources": 4},
        "manifest quotas",
    )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("QA manifest must contain exactly 20 records")
    sources: set[str] = set()
    responses: set[str] = set()
    counts: Counter[tuple[str, int]] = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("QA manifest records must be JSON objects")
        source_id = str(record.get("source_id", ""))
        response_id = str(record.get("response_id", ""))
        split = record.get("split")
        label = record.get("y")
        if (
            not source_id
            or source_id in {".", ".."}
            or "/" in source_id
            or "\\" in source_id
            or source_id in sources
        ):
            raise ValueError("QA manifest source IDs must be unique and non-empty")
        if (
            not response_id
            or response_id in {".", ".."}
            or "/" in response_id
            or "\\" in response_id
            or response_id in responses
        ):
            raise ValueError("QA manifest response IDs must be unique and non-empty")
        if split not in {"train", "test"} or label not in {0, 1}:
            raise ValueError("QA manifest records require split=train|test and y=0|1")
        sources.add(source_id)
        responses.add(response_id)
        counts[(str(split), int(label))] += 1
    expected = Counter({
        ("train", 0): 8,
        ("train", 1): 8,
        ("test", 0): 2,
        ("test", 1): 2,
    })
    if counts != expected:
        raise ValueError(f"QA manifest split/label quotas mismatch: {dict(counts)}")
    sorted_records = sorted(
        records,
        key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])),
    )
    if records != sorted_records:
        raise ValueError("QA manifest records are not in deterministic order")
    return records


def _unit_interval(value: Any, label: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric value")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be between zero and one")


def _validate_score(score: dict[str, Any], relation_mode: str, label: str) -> None:
    if set(score) != SCORE_FIELDS:
        missing = sorted(SCORE_FIELDS - set(score))
        extra = sorted(set(score) - SCORE_FIELDS)
        raise ValueError(f"{label} ScoreResult fields mismatch: missing={missing} extra={extra}")
    for field in ("Vc", "Ec", "Vq", "Eq", "Va", "Ea"):
        if type(score[field]) is not int or score[field] < 0:  # bool is not an integer count
            raise ValueError(f"{label} {field} must be a nonnegative integer")
    for field in (
        "RP_defined", "RP_strict_defined", "RP_grounded_defined",
        "RP_entailed_cond_defined", "RP_support_defined", "support_verified",
        "unscorable", "ref_empty",
    ):
        if type(score[field]) is not bool:
            raise ValueError(f"{label} {field} must be boolean")
    for field in (
        "matched_entities", "ungrounded_entities", "supported_relations",
        "unsupported_relations", "relation_audits",
    ):
        if not isinstance(score[field], list):
            raise ValueError(f"{label} {field} must be a list")
    if len(score["relation_audits"]) != score["Ea"] or any(
        not isinstance(audit, dict) for audit in score["relation_audits"]
    ):
        raise ValueError(f"{label} must have one relation audit per answer edge")

    if score["Va"] == 0:
        _expect(score["EG"], None, f"{label} empty-answer EG")
        _expect(score["unscorable"], True, f"{label} empty-answer flag")
    else:
        _unit_interval(score["EG"], f"{label} EG")
        _expect(score["unscorable"], False, f"{label} answer scorable flag")

    expected_verified = relation_mode == "support"
    _expect(score["support_verified"], expected_verified, f"{label} support verification")
    if score["Ea"] == 0:
        for metric in ("RP", "RP_strict", "RP_grounded", "RP_entailed_cond", "RP_support"):
            _expect(score[metric], None, f"{label} edge-empty {metric}")
        for flag in (
            "RP_defined", "RP_strict_defined", "RP_grounded_defined",
            "RP_entailed_cond_defined", "RP_support_defined",
        ):
            _expect(score[flag], False, f"{label} edge-empty {flag}")
        return

    for metric in ("RP", "RP_strict", "RP_grounded"):
        _unit_interval(score[metric], f"{label} {metric}")
    for flag in ("RP_defined", "RP_strict_defined", "RP_grounded_defined"):
        _expect(score[flag], True, f"{label} {flag}")
    if float(score["RP"]) != float(score["RP_strict"]):
        raise ValueError(f"{label} legacy RP must equal RP_strict")
    if relation_mode == "strict":
        _expect(score["RP_support"], None, f"{label} strict RP_support")
        _expect(score["RP_support_defined"], False, f"{label} strict RP_support_defined")
        _expect(score["RP_entailed_cond"], None, f"{label} strict RP_entailed_cond")
        _expect(
            score["RP_entailed_cond_defined"], False, f"{label} strict entailed flag"
        )
    else:
        _unit_interval(score["RP_support"], f"{label} RP_support")
        _expect(score["RP_support_defined"], True, f"{label} RP_support_defined")
        if float(score["RP_grounded"]) > 0:
            _unit_interval(score["RP_entailed_cond"], f"{label} RP_entailed_cond")
            _expect(
                score["RP_entailed_cond_defined"], True, f"{label} entailed conditional flag"
            )
        else:
            _expect(score["RP_entailed_cond"], None, f"{label} ungrounded conditional RP")
            _expect(
                score["RP_entailed_cond_defined"], False, f"{label} ungrounded conditional flag"
            )


def _validate_scored(
    records: list[dict[str, Any]], expected: list[dict[str, Any]], relation_mode: str
) -> dict[str, dict[str, Any]]:
    if len(records) != 3:
        raise ValueError(f"{relation_mode} scored output must contain exactly three rows")
    identities = []
    scores: dict[str, dict[str, Any]] = {}
    for record in records:
        _expect(record.get("relation_mode"), relation_mode, f"{relation_mode} relation mode")
        score = record.get("score")
        if not isinstance(score, dict):
            raise ValueError(f"{relation_mode} scored row has no score object")
        response_id = str(record.get("response_id", ""))
        _validate_score(score, relation_mode, f"{relation_mode}/{response_id}")
        scores[response_id] = score
        identities.append({
            "source_id": str(record.get("source_id", "")),
            "response_id": str(record.get("response_id", "")),
            "split": record.get("split"),
            "y": int(record.get("y", -1)),
        })
    _expect(identities, expected, f"{relation_mode} scored identities")
    return scores


def _validate_provider_calls(records: list[dict[str, Any]]) -> dict[str, int]:
    if len(records) <= 3:
        raise ValueError(
            "provider_calls.jsonl must include live calls beyond the three contract attempts"
        )
    allowed = {
        "outcome",
        "request_id",
        "latency_s",
        "http_status",
        "retry_index",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error_type",
    }
    totals = {
        "provider_calls": len(records),
        "provider_successes": 0,
        "provider_failures": 0,
        "provider_contract_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for number, record in enumerate(records, start=1):
        unexpected = set(record) - allowed
        missing = allowed - set(record)
        if unexpected or missing:
            raise ValueError(
                f"provider_calls.jsonl:{number} telemetry fields mismatch: "
                f"missing={sorted(missing)} extra={sorted(unexpected)}"
            )
        if record.get("outcome") not in {"success", "failure", "contract_error"}:
            raise ValueError(f"provider_calls.jsonl:{number} has an invalid outcome")
        if record["outcome"] == "contract_error":
            raise ValueError(
                "successful probe cannot contain a swallowed provider contract_error"
            )
        totals["provider_successes"] += int(record["outcome"] == "success")
        totals["provider_failures"] += int(record["outcome"] == "failure")
        for field in ("retry_index", "prompt_tokens", "completion_tokens", "total_tokens"):
            if not isinstance(record.get(field), int) or int(record[field]) < 0:
                raise ValueError(f"provider_calls.jsonl:{number} has invalid {field}")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[field] += int(record[field])
        if not isinstance(record.get("latency_s"), (int, float)) or record["latency_s"] < 0:
            raise ValueError(f"provider_calls.jsonl:{number} has invalid latency_s")
        status = record["http_status"]
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError(f"provider_calls.jsonl:{number} has invalid http_status")
        if record["outcome"] == "success":
            if status != 200:
                raise ValueError(f"provider_calls.jsonl:{number} successful call is not HTTP 200")
            if not isinstance(record["request_id"], str) or not record["request_id"].strip():
                raise ValueError(f"provider_calls.jsonl:{number} successful call has no request ID")
            if record["error_type"] is not None:
                raise ValueError(f"provider_calls.jsonl:{number} successful call has an error type")
            if record["total_tokens"] <= 0 or record["total_tokens"] != (
                record["prompt_tokens"] + record["completion_tokens"]
            ):
                raise ValueError(f"provider_calls.jsonl:{number} has inconsistent token usage")
        else:
            if status is not None and status != 429 and not 500 <= status < 600:
                raise ValueError(
                    f"provider_calls.jsonl:{number} contains a non-retryable HTTP failure"
                )
            if not isinstance(record["error_type"], str) or not record["error_type"].strip():
                raise ValueError(f"provider_calls.jsonl:{number} failure has no error type")
    if totals["provider_successes"] <= 3:
        raise ValueError("successful probe requires live provider calls beyond the contract")
    return totals


def _rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def _validate_audit(
    audit: dict[str, Any],
    score: dict[str, Any],
    expected: dict[str, Any],
    relation_mode: str,
) -> None:
    required = {
        "response_id", "source_id", "relation_mode", "alpha", "alpha_strict",
        "alpha_support", "EG", "RP", "RP_defined", "RP_strict", "RP_grounded",
        "RP_entailed_cond", "RP_support", "RP_support_defined", "CFI_strict",
        "H_strict", "CFI_support", "H_support", "graph_sizes", "relation_audits",
        "probe_diagnostic_alpha_not_tuned",
    }
    missing = required - set(audit)
    if missing:
        raise ValueError(f"{relation_mode} audit is missing fields: {sorted(missing)}")
    _expect(audit["response_id"], expected["response_id"], f"{relation_mode} audit response")
    _expect(audit["source_id"], expected["source_id"], f"{relation_mode} audit source")
    _expect(audit["relation_mode"], relation_mode, f"{relation_mode} audit mode")
    _expect(
        audit["probe_diagnostic_alpha_not_tuned"],
        True,
        f"{relation_mode} audit diagnostic-alpha marker",
    )
    for alpha_field in ("alpha", "alpha_strict", "alpha_support"):
        _expect(float(audit[alpha_field]), 0.7, f"{relation_mode} audit {alpha_field}")
    metric_mapping = {
        "EG": "EG",
        "RP": "RP",
        "RP_strict": "RP_strict",
        "RP_grounded": "RP_grounded",
        "RP_entailed_cond": "RP_entailed_cond",
        "RP_support": "RP_support",
    }
    for audit_field, score_field in metric_mapping.items():
        _expect(
            audit[audit_field],
            _rounded(score[score_field]),
            f"{relation_mode} audit {audit_field}",
        )
    _expect(audit["RP_defined"], score["RP_defined"], f"{relation_mode} audit RP_defined")
    _expect(
        audit["RP_support_defined"],
        score["RP_support_defined"],
        f"{relation_mode} audit RP_support_defined",
    )
    graph_sizes = {field: score[field] for field in ("Vc", "Ec", "Vq", "Eq", "Va", "Ea")}
    _expect(audit["graph_sizes"], graph_sizes, f"{relation_mode} audit graph sizes")
    _expect(
        audit["relation_audits"], score["relation_audits"], f"{relation_mode} edge audits"
    )

    cfi_strict = None
    if not score["unscorable"] and score["EG"] is not None:
        if score["Ea"] == 0:
            cfi_strict = float(score["EG"])
        else:
            cfi_strict = 0.7 * float(score["EG"]) + 0.3 * float(score["RP_strict"])
    _expect(audit["CFI_strict"], _rounded(cfi_strict), f"{relation_mode} CFI_strict")
    _expect(
        audit["H_strict"],
        _rounded(None if cfi_strict is None else 1.0 - cfi_strict),
        f"{relation_mode} H_strict",
    )
    if relation_mode == "strict":
        _expect(audit["CFI_support"], None, "strict audit CFI_support")
        _expect(audit["H_support"], None, "strict audit H_support")
    else:
        cfi_support = None
        if not score["unscorable"] and score["EG"] is not None:
            if score["Ea"] == 0:
                cfi_support = float(score["EG"])
            else:
                cfi_support = 0.7 * float(score["EG"]) + 0.3 * float(score["RP_support"])
        _expect(audit["CFI_support"], _rounded(cfi_support), "support audit CFI_support")
        _expect(
            audit["H_support"],
            _rounded(None if cfi_support is None else 1.0 - cfi_support),
            "support audit H_support",
        )


def _validate_run_metadata(metadata: dict[str, Any], gate: dict[str, Any]) -> None:
    _expect(metadata.get("protocol"), "hallu-api-job-v1", "run metadata protocol")
    _expect(metadata.get("mode"), "probe", "run metadata mode")
    _expect(metadata.get("state"), "completed", "run metadata state")
    _expect(metadata.get("status"), "success", "run metadata status")
    _expect(metadata.get("kind"), "api-probe-c1", "run metadata kind")
    _expect(metadata.get("source_commit"), gate.get("source_commit"), "run metadata commit")
    _expect(metadata.get("model"), gate.get("model"), "run metadata model")
    _expect(metadata.get("api_base"), gate.get("api_base"), "run metadata API base")
    if "run_id" in metadata or "run_id" in gate:
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run metadata run_id must be a nonempty string when present")
        _expect(run_id, gate.get("run_id"), "run ID")
    if not isinstance(metadata.get("completed_at_utc"), str) or not metadata["completed_at_utc"]:
        raise ValueError("run metadata has no completion timestamp")
    elapsed = metadata.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError("run metadata elapsed_seconds must be nonnegative")

    required_versions = _required_runtime_versions()
    runtime_versions = metadata.get("runtime_versions")
    if not isinstance(runtime_versions, dict):
        raise ValueError("run metadata has no runtime_versions object")
    expected_version_keys = {"python", *required_versions}
    _expect(set(runtime_versions), expected_version_keys, "runtime version keys")
    python_version = runtime_versions.get("python")
    if not isinstance(python_version, str) or not re.fullmatch(r"3\.11(?:\.\d+)?", python_version):
        raise ValueError(f"run metadata Python is not 3.11: {python_version!r}")
    for distribution, expected in required_versions.items():
        _expect(
            runtime_versions.get(distribution),
            expected,
            f"runtime version {distribution}",
        )

    fingerprint = metadata.get("llm_runtime_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("run metadata has no LLM runtime fingerprint")
    _expect(
        fingerprint.get("protocol"),
        "dashscope-json-object-strict-local-schema-v1",
        "LLM runtime protocol",
    )
    _expect(fingerprint.get("model"), gate.get("model"), "fingerprint model")
    _expect(fingerprint.get("api_base"), gate.get("api_base"), "fingerprint API base")
    _expect(fingerprint.get("temperature"), 0.0, "fingerprint temperature")
    max_tokens = fingerprint.get("max_tokens")
    timeout = fingerprint.get("request_timeout_s")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("fingerprint max_tokens must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("fingerprint request_timeout_s must be positive")
    _expect(
        fingerprint.get("structured_output_transport"),
        "json_object",
        "fingerprint structured output",
    )
    _expect(
        fingerprint.get("response_format"),
        {"type": "json_object"},
        "fingerprint response format",
    )
    extra_body = fingerprint.get("extra_body")
    if not isinstance(extra_body, dict) or extra_body.get("enable_thinking") is not False:
        raise ValueError("fingerprint must disable provider-side thinking")
    _expect(fingerprint.get("runtime_versions"), runtime_versions, "fingerprint versions")
    forbidden_fingerprint_keys = {"api_key", "authorization", "secret"}
    if any(str(key).casefold() in forbidden_fingerprint_keys for key in fingerprint):
        raise ValueError("LLM runtime fingerprint contains a secret-bearing field")

    result = metadata.get("result")
    if not isinstance(result, dict):
        raise ValueError("completed run metadata has no result object")
    _expect(result.get("kind"), gate.get("kind"), "run result kind")
    for field in (
        "contract_passed", "qa_completed", "failed_extractions",
        "cache_replay_provider_calls", "manifest_sha256", "provider_calls",
        "prompt_tokens", "completion_tokens", "total_tokens",
    ):
        _expect(result.get(field), gate.get(field), f"run result {field}")


def _validate_usage_summary(
    usage: dict[str, Any], provider_totals: dict[str, int], gate: dict[str, Any]
) -> None:
    required = {
        "api_calls", "requests_total", "cache_hits", "cache_hit_rate",
        "estimated_cost_usd", "prompt_tokens", "completion_tokens", "total_tokens",
        "provider_calls", "provider_successes", "provider_failures",
        "provider_contract_errors",
    }
    missing = required - set(usage)
    if missing:
        raise ValueError(f"usage summary is missing fields: {sorted(missing)}")
    for field in (
        "api_calls", "requests_total", "cache_hits", "prompt_tokens",
        "completion_tokens", "total_tokens", "provider_calls", "provider_successes",
        "provider_failures", "provider_contract_errors",
    ):
        if type(usage[field]) is not int or usage[field] < 0:
            raise ValueError(f"usage summary {field} must be a nonnegative integer")
    for field in ("cache_hit_rate", "estimated_cost_usd"):
        value = usage[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"usage summary {field} must be nonnegative")
    if float(usage["cache_hit_rate"]) > 1:
        raise ValueError("usage summary cache_hit_rate cannot exceed one")
    _expect(usage["provider_contract_errors"], 0, "usage contract errors")
    for field, expected in provider_totals.items():
        _expect(usage.get(field), expected, f"usage {field}")
    for field in ("provider_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
        _expect(usage[field], gate.get(field), f"usage/gate {field}")


def _manifest_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_cached_graph(payload: dict[str, Any], label: str) -> tuple[str, int, int]:
    if set(payload) != {"entities", "relations"}:
        raise ValueError(f"{label} must contain only entities and relations")
    entities = payload["entities"]
    relations = payload["relations"]
    if (
        not isinstance(entities, list)
        or any(not isinstance(entity, str) or not entity.strip() for entity in entities)
        or entities != sorted(set(entities))
    ):
        raise ValueError(f"{label} entities must be unique sorted nonempty strings")
    if not isinstance(relations, list):
        raise ValueError(f"{label} relations must be a list")
    canonical_relations: list[list[str]] = []
    entity_set = set(entities)
    for relation in relations:
        if (
            not isinstance(relation, list)
            or len(relation) != 3
            or any(not isinstance(value, str) or not value.strip() for value in relation)
        ):
            raise ValueError(f"{label} relations must be three-field string lists")
        if relation[0] not in entity_set or relation[2] not in entity_set:
            raise ValueError(f"{label} relation endpoints must belong to entities")
        canonical_relations.append(relation)
    if canonical_relations != sorted(canonical_relations) or len(
        {tuple(relation) for relation in canonical_relations}
    ) != len(canonical_relations):
        raise ValueError(f"{label} relations must be unique and sorted")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), len(entities), len(relations)


def _validate_graph_records(
    records: Any,
    expected: list[dict[str, Any]],
    cache_graphs: dict[str, tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("extraction summary must contain exactly three graph records")
    identities: list[dict[str, Any]] = []
    by_response: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("extraction graph records must be JSON objects")
        identity = {
            "source_id": str(record.get("source_id", "")),
            "response_id": str(record.get("response_id", "")),
            "split": record.get("split"),
            "y": record.get("y"),
        }
        identities.append(identity)
        summaries: dict[str, Any] = {}
        for name in ("context", "query", "answer"):
            summary = record.get(name)
            if not isinstance(summary, dict) or set(summary) != {
                "entities", "relations", "sha256"
            }:
                raise ValueError(f"graph record {identity['response_id']}/{name} has invalid fields")
            for field in ("entities", "relations"):
                if type(summary[field]) is not int or summary[field] < 0:
                    raise ValueError(
                        f"graph record {identity['response_id']}/{name}/{field} is invalid"
                    )
            digest = summary["sha256"]
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"graph record {identity['response_id']}/{name} has invalid SHA")
            if digest not in cache_graphs:
                raise ValueError(f"graph record {identity['response_id']}/{name} is absent from KG cache")
            _expect(
                (summary["entities"], summary["relations"]),
                cache_graphs[digest],
                f"graph record {identity['response_id']}/{name} cache sizes",
            )
            summaries[name] = summary
        if summaries["context"]["entities"] == 0 or summaries["answer"]["entities"] == 0:
            raise ValueError(
                f"graph record {identity['response_id']} lacks a scorable context or answer graph"
            )
        by_response[identity["response_id"]] = summaries
    _expect(identities, expected, "3-QA graph record identities")
    return by_response


def validate_probe_artifact(
    path: str | Path,
    *,
    expected_commit: str | None = None,
    expected_model: str | None = None,
    expected_api_base: str | None = None,
    secret: str | None = None,
) -> dict[str, Any]:
    """Validate a successful 3-QA probe and return its gate metadata."""
    if expected_commit is not None and not SHA40_RE.fullmatch(expected_commit):
        raise ValueError("expected commit must be a full lowercase 40-character Git SHA")
    secret_bytes = secret.encode("utf-8") if secret else None
    with ProbeArtifact(path) as artifact:
        if secret_bytes:
            for member in artifact.regular_members():
                handle = artifact.tar.extractfile(member)
                if handle is not None and _stream_contains(handle, secret_bytes):
                    raise ValueError(f"API secret leaked into archive member {member.name!r}")

        gate = artifact.json("gate_metadata.json")
        _expect(gate.get("protocol"), PROTOCOL, "gate protocol")
        _expect(gate.get("kind"), "api-probe-c1", "gate kind")
        _expect(gate.get("status"), "success", "gate status")
        if expected_commit is not None:
            _expect(gate.get("source_commit"), expected_commit, "gate source commit")
        if expected_model is not None:
            _expect(gate.get("model"), expected_model, "gate model")
        if expected_api_base is not None:
            _expect(gate.get("api_base"), expected_api_base, "gate API base")
        _expect(gate.get("contract_passed"), 3, "contract pass count")
        _expect(gate.get("qa_completed"), 3, "completed QA count")
        _expect(gate.get("failed_extractions"), 0, "failed extraction count")
        _expect(gate.get("cache_replay_provider_calls"), 0, "cache replay provider calls")
        commit = str(gate.get("source_commit", ""))
        if not SHA40_RE.fullmatch(commit):
            raise ValueError("gate source_commit is not a full lowercase Git SHA")
        metadata = artifact.json("run_metadata.json")
        _validate_run_metadata(metadata, gate)
        usage_summary = artifact.json("usage_summary.json")
        artifact.one("job.stdout.log")
        artifact.one("job.stderr.log")

        manifest_raw = artifact.raw("qa_pilot_manifest.json")
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("qa_pilot_manifest.json must contain a JSON object")
        manifest_records = _validate_manifest(manifest)
        digest = _manifest_digest(manifest_raw)
        _expect(gate.get("manifest_sha256"), digest, "manifest digest")
        prefix = [
            {
                "source_id": str(row["source_id"]),
                "response_id": str(row["response_id"]),
                "split": row["split"],
                "y": int(row["y"]),
            }
            for row in manifest_records[:3]
        ]

        contract = artifact.json("contract_probe.json")
        _expect(
            contract.get("protocol"),
            "hallu-api-json-object-contract-v1",
            "contract probe protocol",
        )
        _expect(contract.get("status"), "ready", "contract probe status")
        _expect(contract.get("source_id"), "15138", "contract probe source")
        _expect(contract.get("transport"), "json_object", "contract probe transport")
        _expect(contract.get("repair_allowed"), False, "contract repair policy")
        _expect(contract.get("passed"), 3, "contract probe passes")
        attempts = contract.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 3:
            raise ValueError("contract probe must record exactly three attempts")
        if any(
            not isinstance(attempt, dict) or attempt.get("status") != "ready"
            for attempt in attempts
        ):
            raise ValueError("contract probe contains a failed attempt")
        for expected_number, attempt in enumerate(attempts, start=1):
            _expect(attempt.get("attempt"), expected_number, "contract attempt number")
            relations = attempt.get("relations")
            if not isinstance(relations, list) or not relations:
                raise ValueError("contract attempt must contain nonempty relations")
            _expect(attempt.get("relations_count"), len(relations), "contract relation count")
            for relation in relations:
                if (
                    not isinstance(relation, list)
                    or len(relation) != 3
                    or any(not isinstance(value, str) or not value.strip() for value in relation)
                ):
                    raise ValueError("contract relation must be a three-field string triple")
            if not any(
                (
                    "chard" in relation[0].casefold()
                    and ("spinach" in relation[2].casefold() or "beet" in relation[2].casefold())
                )
                or (
                    "chard" in relation[2].casefold()
                    and ("spinach" in relation[0].casefold() or "beet" in relation[0].casefold())
                )
                for relation in relations
            ):
                raise ValueError("contract relation omitted the Swiss-chard semantic anchor")

        synthetic = artifact.json("synthetic_probe.json")
        _expect(
            synthetic.get("protocol"),
            "hallu-api-synthetic-kggen-v1",
            "synthetic KGGen protocol",
        )
        _expect(synthetic.get("status"), "ready", "synthetic KGGen probe status")
        _expect(synthetic.get("cluster"), True, "synthetic KGGen clustering")
        _expect(
            synthetic.get("official_kggen_clustering"),
            True,
            "official KGGen clustering marker",
        )
        if type(synthetic.get("entities")) is not int or synthetic["entities"] < 4:
            raise ValueError("synthetic KGGen probe has too few entities")
        if type(synthetic.get("relations")) is not int or synthetic["relations"] < 2:
            raise ValueError("synthetic KGGen probe has too few relations")
        _expect(
            synthetic.get("semantic_anchors"),
            {"ada_lovelace": True, "marie_curie_warsaw": True},
            "synthetic KGGen semantic anchors",
        )
        verifier = artifact.json("verifier_probe.json")
        _expect(
            verifier.get("protocol"),
            "hallu-api-verifier-probe-v1",
            "verifier probe protocol",
        )
        _expect(verifier.get("status"), "ready", "verifier probe status")
        _expect(verifier.get("verdict"), "entailed", "verifier probe verdict")
        evidence = verifier.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(span, dict) for span in evidence
        ):
            raise ValueError("verifier probe must retain nonempty evidence")

        inventory = artifact.json("cache_inventory.json")
        _expect(inventory.get("protocol"), "hallu-api-cache-inventory-v1", "cache inventory protocol")
        _expect(inventory.get("status"), "ready", "cache inventory status")
        entries = inventory.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("cache inventory has no entries")
        expected_cache_paths: set[str] = set()
        cached_graphs: dict[str, tuple[int, int]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("cache inventory entries must be JSON objects")
            relative = str(entry.get("path", ""))
            parts = PurePosixPath(relative).parts
            if (
                len(parts) != 3
                or parts[0] != ".cache"
                or parts[1] not in {"kg", "verdicts"}
                or not re.fullmatch(r"[0-9a-f]{64}\.json", parts[2])
                or relative in expected_cache_paths
            ):
                raise ValueError(f"invalid cache inventory path: {relative!r}")
            raw = artifact.raw(relative)
            _expect(entry.get("bytes"), len(raw), f"cache size for {relative}")
            _expect(
                entry.get("sha256"), hashlib.sha256(raw).hexdigest(), f"cache digest for {relative}"
            )
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"cache entry {relative!r} is not a JSON object")
            if parts[1] == "kg":
                graph_digest, entity_count, relation_count = _validate_cached_graph(
                    payload, f"KG cache entry {relative!r}"
                )
                previous = cached_graphs.setdefault(
                    graph_digest, (entity_count, relation_count)
                )
                _expect(
                    previous,
                    (entity_count, relation_count),
                    f"KG semantic digest {graph_digest}",
                )
            elif (
                set(payload) != {"verdict"}
                or payload.get("verdict") not in {"entailed", "contradicted", "unknown"}
            ):
                raise ValueError(f"verifier cache entry {relative!r} has an invalid verdict")
            expected_cache_paths.add(relative)
        if not any(path.startswith(".cache/kg/") for path in expected_cache_paths):
            raise ValueError("cache inventory contains no KG cache entries")
        archive_cache_paths: set[str] = set()
        for member in artifact.regular_members():
            parts = PurePosixPath(member.name).parts
            for index in range(len(parts) - 2):
                candidate = parts[index : index + 3]
                if (
                    candidate[0] == ".cache"
                    and candidate[1] in {"kg", "verdicts"}
                    and re.fullmatch(r"[0-9a-f]{64}\.json", candidate[2])
                ):
                    archive_cache_paths.add("/".join(candidate))
        _expect(archive_cache_paths, expected_cache_paths, "cache inventory membership")

        extraction = artifact.json("extraction_summary.json")
        _expect(extraction.get("status"), "ready", "extraction summary status")
        _expect(extraction.get("expected_records"), prefix, "3-QA expected records")
        _expect(extraction.get("completed_records"), prefix, "3-QA completed records")
        _expect(extraction.get("pairs_completed"), 3, "3-QA complete pairs")
        _expect(extraction.get("failures"), [], "3-QA extraction failures")
        graph_records = _validate_graph_records(
            extraction.get("graph_records"), prefix, cached_graphs
        )
        if artifact.raw("failed_extractions.jsonl", max_bytes=MAX_JSONL_BYTES).strip():
            raise ValueError("failed_extractions.jsonl is not empty")

        strict_scores = _validate_scored(
            artifact.jsonl("strict/scored.jsonl"), prefix, "strict"
        )
        support_scores = _validate_scored(
            artifact.jsonl("support/scored.jsonl"), prefix, "support"
        )
        invariant_fields = (
            "Vc", "Ec", "Vq", "Eq", "Va", "Ea", "EG", "RP", "RP_defined",
            "RP_strict", "RP_strict_defined", "RP_grounded", "RP_grounded_defined",
            "matched_entities", "ungrounded_entities", "supported_relations",
            "unsupported_relations", "unscorable", "ref_empty",
        )
        for expected in prefix:
            response_id = expected["response_id"]
            summaries = graph_records[response_id]
            _expect(
                tuple(strict_scores[response_id][field] for field in ("Vc", "Ec")),
                (summaries["context"]["entities"], summaries["context"]["relations"]),
                f"score/context graph sizes {response_id}",
            )
            _expect(
                tuple(strict_scores[response_id][field] for field in ("Vq", "Eq")),
                (summaries["query"]["entities"], summaries["query"]["relations"]),
                f"score/query graph sizes {response_id}",
            )
            _expect(
                tuple(strict_scores[response_id][field] for field in ("Va", "Ea")),
                (summaries["answer"]["entities"], summaries["answer"]["relations"]),
                f"score/answer graph sizes {response_id}",
            )
            for field in invariant_fields:
                _expect(
                    support_scores[response_id][field],
                    strict_scores[response_id][field],
                    f"strict/support invariant {response_id}/{field}",
                )
        for relation_mode in ("strict", "support"):
            for expected in prefix:
                audit = artifact.json(
                    f"{relation_mode}/audit/{expected['response_id']}.json"
                )
                scores = strict_scores if relation_mode == "strict" else support_scores
                _validate_audit(
                    audit,
                    scores[expected["response_id"]],
                    expected,
                    relation_mode,
                )
        provider_totals = _validate_provider_calls(artifact.jsonl("provider_calls.jsonl"))
        for field in ("provider_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            _expect(gate.get(field), provider_totals[field], f"gate {field}")
        _validate_usage_summary(usage_summary, provider_totals, gate)
        replay = artifact.json("cache_replay.json")
        _expect(replay.get("status"), "ready", "cache replay status")
        _expect(replay.get("qa_completed"), 3, "cache replay QA count")
        _expect(replay.get("provider_calls"), 0, "cache replay provider calls")
        _expect(replay.get("failures"), 0, "cache replay failures")
        return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-api-base")
    parser.add_argument(
        "--config",
        help="derive expected model, endpoint, and secret environment name from this YAML",
    )
    parser.add_argument(
        "--secret-env",
        help="also reject an archive containing the current value of this environment variable",
    )
    args = parser.parse_args()
    if args.config:
        import yaml

        document = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        llm = document.get("llm") if isinstance(document, dict) else None
        if not isinstance(llm, dict):
            raise SystemExit(f"config has no llm mapping: {args.config}")
        config_model = str(llm.get("model", ""))
        config_api_base = str(llm.get("api_base", ""))
        config_secret_env = str(llm.get("api_key_env", ""))
        if args.expected_model is not None and args.expected_model != config_model:
            raise SystemExit("--expected-model differs from --config")
        if args.expected_api_base is not None and args.expected_api_base != config_api_base:
            raise SystemExit("--expected-api-base differs from --config")
        if args.secret_env is not None and args.secret_env != config_secret_env:
            raise SystemExit("--secret-env differs from --config")
        args.expected_model = config_model
        args.expected_api_base = config_api_base
        args.secret_env = config_secret_env
    secret = None
    if args.secret_env:
        import os

        secret = os.environ.get(args.secret_env)
    gate = validate_probe_artifact(
        args.artifact,
        expected_commit=args.expected_commit,
        expected_model=args.expected_model,
        expected_api_base=args.expected_api_base,
        secret=secret,
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
