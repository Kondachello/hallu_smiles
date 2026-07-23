from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from hallugraph_dynamic_typing.graph_spec import NODE_SPECS, NodeKind


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "prompts" / "v1"
MANIFEST = PROMPT_ROOT / "manifest.json"
VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _prompt_set_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in PROMPT_ROOT.rglob("*") if item.is_file()):
        digest.update(path.relative_to(PROMPT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_strict_objects(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        node_type = value.get("type")
        is_object = node_type == "object" or (
            isinstance(node_type, list) and "object" in node_type
        )
        if is_object:
            assert value.get("additionalProperties") is False, path
        for key, child in value.items():
            _assert_strict_objects(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_strict_objects(child, path=f"{path}[{index}]")


def test_manifest_covers_every_unique_model_prompt() -> None:
    manifest = _manifest()
    manifest_ids = {entry["prompt_id"] for entry in manifest["entries"]}
    graph_prompt_ids = {
        item.prompt_id
        for item in NODE_SPECS
        if item.kind in {NodeKind.MODEL, NodeKind.NLI}
    }
    assert manifest_ids == graph_prompt_ids
    assert len(manifest["entries"]) == len(manifest_ids)


def test_complete_prompt_set_hash_is_deterministic() -> None:
    first = _prompt_set_sha256()
    second = _prompt_set_sha256()
    assert first == second
    assert len(first) == 64


def test_prompt_paths_variables_and_schemas_are_valid() -> None:
    root = PROMPT_ROOT.resolve()
    for entry in _manifest()["entries"]:
        paths = {
            name: (PROMPT_ROOT / entry[name]).resolve()
            for name in ("system", "user", "schema")
        }
        assert all(path.is_relative_to(root) for path in paths.values())
        assert all(path.is_file() for path in paths.values())

        system_text = paths["system"].read_text(encoding="utf-8").strip()
        user_text = paths["user"].read_text(encoding="utf-8").strip()
        assert len(system_text) >= 200
        assert len(user_text) >= 100
        assert set(VARIABLE.findall(user_text)) == set(entry["required_variables"])

        schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
        _assert_strict_objects(schema)


def test_source_prompts_do_not_accept_answer_variables() -> None:
    forbidden = {"answer", "answer_text", "response", "response_text", "answer_graph_json"}
    for entry in _manifest()["entries"]:
        if entry["phase"] == "source":
            assert set(entry["required_variables"]).isdisjoint(forbidden), entry["prompt_id"]


def test_schema_overview_accepts_localized_internal_ids_without_relaxing_output_shape() -> None:
    schema = json.loads((PROMPT_ROOT / "schemas" / "schema_overview.schema.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "schema-overview-v1",
        "source_summary": "Русский источник описывает коммерческий банк.",
        "draft_types": [{
            "candidate_type_id": "CT-коммерческий-банк",
            "label": "коммерческий банк",
            "definition": "Финансовая организация, указанная в источнике.",
            "parent_candidate_ids": [], "aliases": [], "distinctions": [], "role_signatures": [],
            "evidence_span_ids": ["context:span:0"], "evidence_level": "source_entailed",
        }],
        "contextual_roles": [{
            "role_id": "CR-заёмщик", "label": "заёмщик", "relation_pattern": "получает кредит",
            "evidence_span_ids": ["context:span:0"],
        }],
        "open_questions": [],
    }
    Draft202012Validator(schema).validate(payload)


def test_prompts_state_closed_world_and_injection_boundaries() -> None:
    system_documents = [
        (PROMPT_ROOT / entry["system"]).read_text(encoding="utf-8").lower()
        for entry in _manifest()["entries"]
    ]
    assert all("only" in document for document in system_documents)
    assert all("instructions" in document or "instruction" in document for document in system_documents)
