#!/usr/bin/env python3
"""Verify the checked-in Outlines JSON guided-decoding compatibility shim.

vLLM 0.6.3.post1 installs Outlines 0.0.46.  That release imports
``pyairports.airports`` at module import time, while its declared PyPI
dependency does not supply that module.  The DataSphere runtime uses the
checked-in JSON-only shim via ``PYTHONPATH``.  Verify that exact path on CPU
before allocating a V100.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dspy_adapter import canonicalize_vllm_guided_json_schema


EXPECTED_OUTLINES_VERSION = "0.0.46"


KGGEN_RELATION_SCHEMA: dict[str, Any] = {
    "$defs": {
        "Relation": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "__dspy_field_type": "input", "desc": "subject"},
                "predicate": {"type": "string", "examples": ["is related to"]},
                "object": {"type": "string", "title": "Object"},
            },
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False,
        },
    },
    "type": "object",
    "properties": {"relations": {"type": "array", "items": {"$ref": "#/$defs/Relation"}}},
    "required": ["relations"],
    "additionalProperties": False,
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check() -> dict[str, Any]:
    version = metadata.version("outlines")
    if version != EXPECTED_OUTLINES_VERSION:
        raise RuntimeError(
            f"expected outlines=={EXPECTED_OUTLINES_VERSION}, found {version}; "
            "do not run an untested guided-decoding backend"
        )
    from pyairports.airports import AIRPORT_LIST
    from outlines.fsm.json_schema import build_regex_from_schema
    from outlines.integrations.vllm import JSONLogitsProcessor

    if not isinstance(AIRPORT_LIST, list):
        raise RuntimeError("pyairports compatibility shim did not expose a list")
    if JSONLogitsProcessor is None:  # Defensive: import success must be meaningful.
        raise RuntimeError("Outlines vLLM JSON logits processor is unavailable")
    flattened_schema = canonicalize_vllm_guided_json_schema(KGGEN_RELATION_SCHEMA)
    serialised_schema = json.dumps(flattened_schema, sort_keys=True)
    if "$defs" in flattened_schema or "\"$ref\"" in serialised_schema or "__dspy_" in serialised_schema:
        raise RuntimeError("KGGen relation schema was not fully canonicalised before Outlines compilation")
    # The processor needs a live tokenizer, but the schema->regex compiler does
    # not.  Compile the *actual nested KGGen relation grammar* on CPU before a
    # V100 is allocated, rather than merely checking that Outlines imports.
    relation_regex = build_regex_from_schema(json.dumps(flattened_schema, sort_keys=True))
    if not isinstance(relation_regex, str) or not relation_regex:
        raise RuntimeError("Outlines did not compile the flattened KGGen relation schema")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "outlines_version": version,
        "guided_decoding_backend": "outlines",
        "pyairports_shim_entries": len(AIRPORT_LIST),
        "checks": [
            "import pyairports.airports from checked-in runtime shim",
            "import outlines.integrations.vllm.JSONLogitsProcessor",
            "compile canonical nested KGGen Relation schema to an Outlines regex",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = check()
    except Exception as exc:
        _atomic_json(
            report_path,
            {
                "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
