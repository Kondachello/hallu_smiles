#!/usr/bin/env python3
"""CPU preflight for the split, immutable DataSphere Docker runtime."""
from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _compile_xgrammar(
    server_python: str, schemas_path: Path, request_backend: str
) -> dict[str, Any]:
    program = (
        "import json,sys,xgrammar as xgr; "
        "from vllm.sampling_params import GuidedDecodingParams; "
        "params=GuidedDecodingParams(backend=sys.argv[2]); "
        "assert params.backend_name=='xgrammar'; "
        "assert set(params.backend_options())=="
        "{'disable-any-whitespace','no-fallback'}; "
        "schemas=json.load(open(sys.argv[1],encoding='utf-8')); "
        "[xgr.Grammar.from_json_schema(json.dumps(schema),any_whitespace=False) "
        "for schema in schemas.values()]; "
        "print(json.dumps({'request_backend':params.backend,"
        "'backend_name':params.backend_name,'backend_options':"
        "sorted(params.backend_options()),'any_whitespace':False},sort_keys=True))"
    )
    completed = subprocess.run(
        [server_python, "-c", program, str(schemas_path), request_backend],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("backend_name") == "xgrammar":
            return payload
    raise RuntimeError(
        "XGrammar contract check emitted no JSON payload: "
        f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-2000:]!r}"
    )


def _check_native_build_toolchain(server_python: str) -> dict[str, str]:
    """Verify the headers/toolchain needed by XGrammar's first Triton request."""
    gcc = shutil.which("gcc")
    if gcc is None:
        raise RuntimeError("Docker runtime is missing gcc required by Triton JIT")
    program = (
        "import json,sysconfig; from pathlib import Path; "
        "include=Path(sysconfig.get_paths()['include']); header=include/'Python.h'; "
        "assert header.is_file(), f'missing Python development header: {header}'; "
        "print(json.dumps({'python_include':str(include),'python_header':str(header)},sort_keys=True))"
    )
    completed = subprocess.run(
        [server_python, "-c", program],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    return {"gcc": gcc, **payload}


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
    import dspy
    from kg_gen.steps._3_cluster_graph import (
        Cluster,
        choose_rep,
        get_check_existing_clusters_sig,
        get_extract_cluster_sig,
        get_validate_cluster_sig,
    )
    from src.dspy_adapter import (
        STRUCTURED_OUTPUT_PROTOCOL_VERSION,
        StructuredOutputSchemaError,
        XGRAMMAR_STRICT_REQUEST_BACKEND,
        dspy_output_schema,
        specialize_dspy_signature,
        validate_json_document,
    )
    from src.verifier import VERDICT_SCHEMA

    relation_schema = kggen_fallback_relation_schema()
    empty_relation_schema = kggen_fallback_relation_schema([])
    empty_relation_array = empty_relation_schema["properties"]["relations"]
    if (
        empty_relation_array.get("minItems") != 0
        or empty_relation_array.get("maxItems") != 0
    ):
        raise RuntimeError("empty entity input does not force an empty relation array")
    validate_json_document({"relations": []}, empty_relation_schema)
    validate_json_document({"relations": []}, relation_schema)
    relation_model = next(iter(relation_schema.get("$defs", {}).values()))
    endpoint_enum = relation_model["properties"]["subject"].get("enum")
    if not endpoint_enum or endpoint_enum != relation_model["properties"]["object"].get("enum"):
        raise RuntimeError("runtime relation endpoints are not bound to one entity enum")
    valid_relation = {
        "relations": [
            {
                "subject": endpoint_enum[0],
                "predicate": "is distinct from",
                "object": endpoint_enum[-1],
            }
        ]
    }
    validate_json_document(valid_relation, relation_schema)
    try:
        validate_json_document(
            {
                "relations": [
                    {
                        "subject": "out-of-contract endpoint",
                        "predicate": "is distinct from",
                        "object": endpoint_enum[0],
                    }
                ]
            },
            relation_schema,
        )
    except StructuredOutputSchemaError:
        pass
    else:
        raise RuntimeError("relation endpoint enum accepted an out-of-contract value")

    cluster_candidates = {
        "Swiss chard",
        "chard",
        "spinach",
        "Beta vulgaris subsp. maritima",
        "sea beet",
        "pizzoccheri",
        'quoted "entity"',
        "München",
        "line\nbreak",
        "is cultivated descendants of",
        "is similar to",
        "is extremely perishable",
    }
    current_candidates = cluster_candidates - {"spinach", "pizzoccheri"}
    extract_cluster, _ = get_extract_cluster_sig(cluster_candidates)
    extract_cluster_schema = dspy_output_schema(
        specialize_dspy_signature(
            extract_cluster,
            {"items": current_candidates},
        )
    )
    validate_cluster, _ = get_validate_cluster_sig(current_candidates)
    validate_cluster_schema = dspy_output_schema(
        specialize_dspy_signature(
            validate_cluster,
            {"cluster": current_candidates},
        )
    )
    representative_schema = dspy_output_schema(
        specialize_dspy_signature(
            choose_rep.signature,
            {"cluster": current_candidates},
        )
    )
    existing_clusters = [
        Cluster(representative="Swiss chard", members={"Swiss chard", "chard"}),
        Cluster(representative="sea beet", members={"sea beet"}),
    ]
    check_existing = get_check_existing_clusters_sig(
        {"München", 'quoted "entity"'}, existing_clusters
    )
    check_existing_schema = dspy_output_schema(
        specialize_dspy_signature(
            dspy.ChainOfThought(check_existing).predict.signature,
            {
                "items": ["München", 'quoted "entity"'],
                "clusters": existing_clusters,
            },
        )
    )
    validate_json_document({"verdict": "unknown"}, VERDICT_SCHEMA)
    report_path = Path(args.report)
    schemas_path = report_path.with_name("preflight-schemas.json")
    _atomic_json(
        schemas_path,
        {
            "relations": relation_schema,
            "relations_empty_entities": empty_relation_schema,
            "verifier": VERDICT_SCHEMA,
            "cluster_extract": extract_cluster_schema,
            "cluster_validate": validate_cluster_schema,
            "cluster_representative": representative_schema,
            "cluster_existing_assignment": check_existing_schema,
        },
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
    native_build_toolchain = _check_native_build_toolchain(args.server_python)
    xgrammar_contract = _compile_xgrammar(
        args.server_python, schemas_path, XGRAMMAR_STRICT_REQUEST_BACKEND
    )
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
        "native_build_toolchain": native_build_toolchain,
        "xgrammar_contract": xgrammar_contract,
        "structured_output_protocol": STRUCTURED_OUTPUT_PROTOCOL_VERSION,
        "schemas": str(schemas_path),
        "checks": [
            "split server/client dependency imports",
            "exact CUDA 11.8 server and CPU-only client PyTorch builds",
            "gcc and Python development headers for XGrammar/Triton JIT",
            "XGrammar bounded-whitespace compilation of relation, verifier and all dynamic KGGen clustering schemas",
            "runtime-bound relation endpoints and current-item clustering contracts",
            "offline CPU SBERT embedding",
            "runtime source commit and embedded asset identity",
        ],
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
