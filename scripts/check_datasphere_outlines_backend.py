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
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


EXPECTED_OUTLINES_VERSION = "0.0.46"


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
    from outlines.integrations.vllm import JSONLogitsProcessor

    if not isinstance(AIRPORT_LIST, list):
        raise RuntimeError("pyairports compatibility shim did not expose a list")
    if JSONLogitsProcessor is None:  # Defensive: import success must be meaningful.
        raise RuntimeError("Outlines vLLM JSON logits processor is unavailable")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "outlines_version": version,
        "guided_decoding_backend": "outlines",
        "pyairports_shim_entries": len(AIRPORT_LIST),
        "checks": [
            "import pyairports.airports from checked-in runtime shim",
            "import outlines.integrations.vllm.JSONLogitsProcessor",
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
