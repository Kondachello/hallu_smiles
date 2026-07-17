#!/usr/bin/env python3
"""Verify the exact vLLM guided-decoding dependency chain on a CPU Job.

This deliberately imports the lm-format-enforcer Transformers integration.
It catches the otherwise late ``LogitsWarper`` incompatibility before a V100
is allocated. It does not load model weights and does not require a GPU.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


EXPECTED_VERSIONS = {
    "kg-gen": "0.4.0",
    "dspy": "2.6.27",
    "litellm": "1.60.4",
    "vllm": "0.6.3.post1",
    "transformers": "4.45.2",
    "lm-format-enforcer": "0.10.6",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _import(module: str) -> None:
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise RuntimeError(f"runtime dependency import failed: {module}: {exc}") from exc


def check() -> dict[str, Any]:
    versions = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "installed": versions[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"runtime dependency versions differ from the tested lock: {mismatches}")

    _import("vllm")
    _import("transformers")
    _import("lmformatenforcer")
    if os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() != "true":
        raise RuntimeError("LITELLM_LOCAL_MODEL_COST_MAP=true is required for offline-safe Job startup")
    _import("litellm")
    _import("dspy")
    _import("kg_gen")
    # lm-format-enforcer 0.10.6 imports LogitsWarper here. Transformers 4.57
    # removed that symbol and previously caused a paid GPU HTTP 500.
    integration = importlib.import_module("lmformatenforcer.integrations.transformers")
    if not hasattr(integration, "build_token_enforcer_tokenizer_data"):
        raise RuntimeError("lm-format-enforcer Transformers integration has no tokenizer adapter")

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "versions": versions,
        "checks": [
            "import vllm",
            "import transformers",
            "import lmformatenforcer",
            "import lmformatenforcer.integrations.transformers",
            "import litellm with local model-cost map",
            "import dspy",
            "import kg_gen",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="Optional writable JSON report path.")
    args = parser.parse_args()
    report = check()
    if args.report:
        _atomic_json(Path(args.report), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
