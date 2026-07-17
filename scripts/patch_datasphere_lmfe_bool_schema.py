#!/usr/bin/env python3
"""Backport lm-format-enforcer's boolean JSON-schema parser fix in a Job venv.

vLLM 0.6.3.post1 pins ``lm-format-enforcer==0.10.6`` exactly.  That package
has a five-line upstream defect: its Pydantic validator calls ``.get`` on a
boolean when a normal closed JSON schema contains
``additionalProperties: false``.  Newer lm-format-enforcer releases contain
the fix, but cannot be installed together with this vLLM release.

This script applies only that upstream five-line backport to the *ephemeral
manual Job virtualenv*, before vLLM imports the package.  It never touches
shared Project storage, model files, or the repository checkout.  The patch is
idempotent, version-checked and recorded in JSON so the CPU preflight can
prove the exact live configuration before a GPU is allocated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


EXPECTED_VERSION = "0.10.6"
PATCH_MARKER = "# Hallu Smiles backport: accept boolean JSON Schema nodes"
BUGGY_FRAGMENT = """    ) -> Any:
        exclusive_maximum: Union[float, bool, None] = values.get('exclusiveMaximum')
"""
PATCHED_FRAGMENT = """    ) -> Any:
        # Hallu Smiles backport: accept boolean JSON Schema nodes
        if isinstance(values, bool):
            return values

        exclusive_maximum: Union[float, bool, None] = values.get('exclusiveMaximum')
"""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def patch_source(path: Path) -> str:
    """Apply the narrow upstream backport and return ``patched`` or ``present``."""
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return "present"
    if source.count(BUGGY_FRAGMENT) != 1:
        raise RuntimeError(
            "lm-format-enforcer source does not match the audited 0.10.6 validator; "
            "refusing an unsafe patch"
        )
    patched = source.replace(BUGGY_FRAGMENT, PATCHED_FRAGMENT, 1)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(patched, encoding="utf-8")
    py_compile.compile(str(temporary), doraise=True)
    os.replace(temporary, path)
    return "patched"


def apply_patch() -> dict[str, Any]:
    installed_version = metadata.version("lm-format-enforcer")
    if installed_version != EXPECTED_VERSION:
        raise RuntimeError(
            f"expected lm-format-enforcer=={EXPECTED_VERSION}, found {installed_version}; "
            "do not patch an unreviewed version"
        )
    import lmformatenforcer.external.jsonschemaobject as jsonschemaobject

    source_path = Path(jsonschemaobject.__file__).resolve()
    state = patch_source(source_path)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "installed_version": installed_version,
        "patch_state": state,
        "module_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "patch_marker": PATCH_MARKER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Writable JSON report path.")
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = apply_patch()
    except Exception as exc:
        _atomic_json(
            report_path,
            {
                "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "expected_version": EXPECTED_VERSION,
            },
        )
        raise
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
