#!/usr/bin/env python3
"""CPU preflight for the split, immutable DataSphere Docker runtime."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SERVER_VERSIONS = {
    "vllm": "0.8.5.post1+cu118",
    "torch": "2.6.0+cu118",
    "torchvision": "0.21.0+cu118",
    "torchaudio": "2.6.0+cu118",
    "xformers": "0.0.29.post2",
    "transformers": "4.51.3",
    "xgrammar": "0.1.18",
}
CLIENT_VERSIONS = {
    "torch": "2.6.0+cpu",
    "kg-gen": "0.4.0",
    "dspy": "2.6.27",
    "litellm": "1.60.4",
    "pydantic": "2.10.6",
    "sentence-transformers": "5.6.0",
    "jsonschema": "4.23.0",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _inspection_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Return the checker payload while tolerating dependency logs on stdout."""
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("versions"), dict)
            and isinstance(payload.get("modules"), list)
            and "torch_cuda" in payload
        ):
            return payload
    stdout_tail = completed.stdout[-4000:]
    stderr_tail = completed.stderr[-4000:]
    raise RuntimeError(
        "runtime inspection emitted no valid JSON payload; "
        f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
    )


def _inspect(
    python: str,
    expected: dict[str, str],
    modules: list[str],
    *,
    expected_torch_cuda: str | None,
) -> dict[str, Any]:
    program = (
        "import importlib,json,torch; from importlib import metadata; "
        f"expected={expected!r}; modules={modules!r}; "
        "versions={k:metadata.version(k) for k in expected}; "
        "[importlib.import_module(m) for m in modules]; "
        "print(json.dumps({'versions':versions,'modules':modules,"
        "'torch_cuda':torch.version.cuda},sort_keys=True))"
    )
    completed = subprocess.run(
        [python, "-c", program], check=True, text=True, capture_output=True, timeout=120
    )
    payload = _inspection_payload(completed)
    mismatches = {
        name: {"expected": version, "installed": payload["versions"].get(name)}
        for name, version in expected.items()
        if str(payload["versions"].get(name, "")) != version
    }
    if mismatches:
        raise RuntimeError(f"runtime dependency mismatch: {mismatches}")
    if payload["torch_cuda"] != expected_torch_cuda:
        raise RuntimeError(
            "runtime torch CUDA mismatch: "
            f"expected {expected_torch_cuda!r}, installed {payload['torch_cuda']!r}"
        )
    return payload


def _compile_xgrammar(server_python: str, schemas_path: Path) -> None:
    program = (
        "import json,sys,xgrammar as xgr; "
        "schemas=json.load(open(sys.argv[1],encoding='utf-8')); "
        "[xgr.Grammar.from_json_schema(json.dumps(schema)) for schema in schemas.values()]"
    )
    subprocess.run(
        [server_python, "-c", program, str(schemas_path)],
        check=True,
        timeout=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    from scripts.check_datasphere_vllm_guided_json import kggen_fallback_relation_schema
    from src.dspy_adapter import validate_json_document
    from src.verifier import VERDICT_SCHEMA

    relation_schema = kggen_fallback_relation_schema()
    enum_schema = {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Swiss chard", "chard"]},
                },
            }
        },
        "required": ["clusters"],
        "additionalProperties": False,
    }
    validate_json_document({"relations": []}, relation_schema)
    validate_json_document({"verdict": "unknown"}, VERDICT_SCHEMA)
    validate_json_document({"clusters": [["Swiss chard", "chard"]]}, enum_schema)
    report_path = Path(args.report)
    schemas_path = report_path.with_name("preflight-schemas.json")
    _atomic_json(
        schemas_path,
        {"relations": relation_schema, "verifier": VERDICT_SCHEMA, "clustering_enum": enum_schema},
    )

    server = _inspect(
        args.server_python,
        SERVER_VERSIONS,
        ["torch", "torchvision", "torchaudio", "xformers", "transformers", "xgrammar", "vllm"],
        expected_torch_cuda="11.8",
    )
    client = _inspect(
        args.client_python,
        CLIENT_VERSIONS,
        ["torch", "kg_gen", "dspy", "litellm", "pydantic", "sentence_transformers", "jsonschema"],
        expected_torch_cuda=None,
    )
    _compile_xgrammar(args.server_python, schemas_path)
    embedding_program = (
        "import sys; from sentence_transformers import SentenceTransformer; "
        "m=SentenceTransformer(sys.argv[1],device='cpu',local_files_only=True); "
        "v=m.encode(['offline preflight'],convert_to_numpy=True); "
        "assert v.shape[0]==1"
    )
    subprocess.run(
        [args.client_python, "-c", embedding_program, args.embedding_path],
        check=True,
        timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
    )

    manifest = json.loads(Path(args.runtime_manifest).read_text(encoding="utf-8"))
    if manifest.get("source_commit") != args.expected_source_commit:
        raise RuntimeError("Docker runtime was not built from the selected Git commit")
    if manifest.get("embedding_path") != args.embedding_path:
        raise RuntimeError("Docker runtime embedding path differs from preflight")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "runtime_manifest": manifest,
        "server": server,
        "client": client,
        "schemas": str(schemas_path),
        "checks": [
            "split server/client dependency imports",
            "exact CUDA 11.8 server and CPU-only client PyTorch builds",
            "XGrammar compilation of relation, verifier and enum schemas",
            "offline CPU SBERT embedding",
            "runtime source commit and embedded asset identity",
        ],
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
