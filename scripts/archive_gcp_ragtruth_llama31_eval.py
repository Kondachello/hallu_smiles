#!/usr/bin/env python3
"""Create the redacted terminal archive for the controlled GCP evaluation."""
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_ROOT_FILES = {
    "input_provenance.json",
    "llama31_manifest.json",
    "frozen_reference_provenance.json",
    "gateway_manifest.json",
    "runtime_identity.json",
    "config_identity.json",
    "cache_before_replay.json",
    "cache_after_replay.json",
    "replay_verification.json",
    "controlled_summary.json",
    "run_metadata.json",
    "run.log",
}
ALLOWED_METHOD_FILES = {
    "metrics.csv",
    "summary_metrics.csv",
    "tuning.json",
    "report.md",
    "extraction_summary_redacted.json",
}


def _redacted_extraction(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    failures = payload.get("failures", [])
    failure_counts = Counter(
        str(item.get("stage", "unknown"))
        for item in failures
        if isinstance(item, dict)
    )
    redacted = {
        "protocol": "hallu-extraction-summary-redacted-v1",
        "status": payload.get("status"),
        "expected_sources": payload.get("expected_sources"),
        "analysis_expected_sources": payload.get("analysis_expected_sources"),
        "references_completed": payload.get("references_completed"),
        "responses_completed": payload.get("responses_completed"),
        "analysis_expected_responses": payload.get("analysis_expected_responses"),
        "pairs_completed": payload.get("pairs_completed"),
        "excluded_source_ids": payload.get("excluded_source_ids"),
        "failure_counts_by_stage": dict(sorted(failure_counts.items())),
        "reference_graph_provenance": payload.get("reference_graph_provenance"),
    }
    output.write_text(json.dumps(redacted, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _included(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if len(relative.parts) == 1:
        return relative.name in ALLOWED_ROOT_FILES
    return len(relative.parts) == 2 and relative.parts[0] in {"strict", "support-critical"} \
        and relative.name in ALLOWED_METHOD_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if not root.is_dir() or not root.name.startswith("gcp-ragtruth-llama31-"):
        raise SystemExit("run root is not a controlled GCP Llama artifact directory")
    for method in ("strict", "support-critical"):
        source = root / method / "extraction_summary.json"
        if not source.is_file():
            if args.allow_partial:
                continue
            raise SystemExit(f"missing {method} extraction summary")
        _redacted_extraction(source, root / method / "extraction_summary_redacted.json")
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and _included(root, path):
                handle.add(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
    with tarfile.open(temporary, "r:gz") as handle:
        members = [member.name for member in handle.getmembers()]
    if not members or any(
        "cache" in Path(name).parts
        or Path(name).name in {"scored.jsonl", "usage.jsonl", "extraction_summary.json"}
        or Path(name).name.endswith(".raw.json")
        for name in members
    ):
        temporary.unlink(missing_ok=True)
        raise SystemExit("refusing to write a non-redacted controlled evaluation archive")
    temporary.replace(archive)
    print(json.dumps({"archive": str(archive), "members": len(members)}, sort_keys=True))


if __name__ == "__main__":
    main()
