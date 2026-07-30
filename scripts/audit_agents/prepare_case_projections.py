#!/usr/bin/env python3
"""Split exported audit-case packages into single-method projections.

The auditor agent for HalluGraph must not be able to anchor on GraphEval's verdict,
and vice versa.  The audit system prompt asks for that in words; this command makes
it structural by removing the other method from the package the agent ever sees.

Analysis-only: reads packages written by ``export_historical_replay_audit_case.py``
and writes new files beside them.  It never touches the sealed archive and never
invokes KGGen, HalluGraph, GraphEval, an LLM, or a gateway.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

METHODS = ("hallugraph", "grapheval")
SUFFIX = {"hallugraph": "hg", "grapheval": "ge"}

# Substrings that must never survive into a projection for the *other* method.
LEAK_PATTERNS = {
    "hallugraph": re.compile(r"grapheval|graph[ _-]?eval", re.IGNORECASE),
    "grapheval": re.compile(r"hallugraph|hallu[ _-]?graph", re.IGNORECASE),
}


class LeakError(RuntimeError):
    """Raised when a projection still mentions the method it is meant to hide."""


def project(package: dict[str, Any], keep: str) -> dict[str, Any]:
    """Return a copy of ``package`` containing only method ``keep``."""
    if keep not in METHODS:
        raise ValueError(f"unknown method {keep!r}")
    drop = [m for m in METHODS if m != keep]

    out = json.loads(json.dumps(package))  # deep copy, packages are plain JSON
    out["schema_version"] = f"{package['schema_version']}+single-method"
    # The withheld method is deliberately NOT named: knowing which comparison
    # detector exists is itself an anchor. Only the count is disclosed.
    out["audit_scope"] = {
        "method_under_audit": keep,
        "withheld_comparison_methods": len(drop),
        "rationale": (
            "Verdicts of any comparison detector are removed so that this audit cannot "
            "be anchored on them. Cross-method comparison happens in a separate later stage."
        ),
    }

    methods = out.get("methods") or {}
    for method in drop:
        methods.pop(method, None)
    out["methods"] = methods

    classification = out.get("classification") or {}
    for method in drop:
        classification.pop(f"{method}_outcome", None)
    # paired_score_available is a statement about both methods; meaningless here.
    classification.pop("paired_score_available", None)
    out["classification"] = classification

    return out


def assert_no_leak(projection: dict[str, Any], keep: str) -> None:
    """Fail loudly if the hidden method is still mentioned anywhere in the payload."""
    pattern = LEAK_PATTERNS[keep]
    blob = json.dumps(projection, ensure_ascii=False)
    hit = pattern.search(blob)
    if hit is None:
        return
    start = max(0, hit.start() - 120)
    raise LeakError(
        f"projection for {keep!r} (case {projection.get('case_id')!r}) still mentions "
        f"the hidden method at offset {hit.start()}: ...{blob[start:hit.end() + 120]}..."
    )


def iter_packages(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("audit-case-*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="directory of audit-case-<id>.json packages")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for single-method projections")
    parser.add_argument(
        "--method",
        choices=(*METHODS, "both"),
        default="both",
        help="which projection(s) to emit (default: both)",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing projections")
    args = parser.parse_args()

    packages = iter_packages(args.input_dir)
    if not packages:
        raise SystemExit(f"no audit-case-*.json found in {args.input_dir}")

    wanted = METHODS if args.method == "both" else (args.method,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in packages:
        package = json.loads(path.read_text(encoding="utf-8"))
        case_id = package["case_id"]
        for method in wanted:
            projection = project(package, method)
            assert_no_leak(projection, method)
            target = args.output_dir / f"case-{case_id}.{SUFFIX[method]}.json"
            if target.exists() and not args.overwrite:
                raise SystemExit(f"refusing to overwrite {target}; pass --overwrite")
            target.write_text(
                json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written += 1

    print(f"wrote {written} projection(s) for {len(packages)} case(s) into {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
