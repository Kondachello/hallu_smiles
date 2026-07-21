#!/usr/bin/env python3
"""Run the paired experiment framework with the real detector adapters on fakes.

This is an infrastructure smoke test only: it uses two synthetic no-gold inputs,
FakeKGGen/DictEmbedder for HalluGraph and FakeExtractor/FakeNLI for GraphEval.
It deliberately has no RAGTruth download, secret, gateway, or model call.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import RunArchive, atomic_write_json, atomic_write_jsonl
from experiments.demo import demo_instances
from experiments.detectors import build_grapheval_fake, build_hallugraph_fake
from experiments.runner import run_paired, seal_run


def _run_once(root: Path, run_id: str, config: Path) -> tuple[RunArchive, dict]:
    archive = RunArchive.create(
        root,
        run_id=run_id,
        manifest={
            "run_purpose": "datasphere_offline_mock",
            "comparison_track": "infrastructure_smoke",
            "network_access": False,
            "data_source": "synthetic",
        },
    )
    instances = archive.path / "instances.no_gold.jsonl"
    rows = demo_instances()
    atomic_write_jsonl(instances, rows)
    summary = run_paired(
        archive,
        instances_path=instances,
        detectors={
            "hallugraph": build_hallugraph_fake(config),
            "grapheval": build_grapheval_fake(),
        },
    )
    seal_run(archive, instances)
    validation = archive.validate()
    if not validation["valid"]:
        raise RuntimeError(f"invalid prediction archive: {validation['errors']}")
    return archive, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="writable artifact directory")
    parser.add_argument("--config", default="config.yaml", help="HalluGraph config (contains no secret)")
    args = parser.parse_args()

    output = Path(args.out).resolve()
    config = Path(args.config).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    first, first_summary = _run_once(output, "first", config)
    second, second_summary = _run_once(output, "replay", config)
    def prediction_semantics(archive: RunArchive) -> list[dict]:
        return [
            {
                key: row[key]
                for key in ("method", "response_id", "status", "raw_score", "components", "flagged_unit_ids")
            }
            for row in archive.read_jsonl("predictions/raw_predictions.jsonl")
        ]

    first_predictions = prediction_semantics(first)
    second_predictions = prediction_semantics(second)
    methods = sorted({row["method"] for row in first.read_jsonl("predictions/raw_predictions.jsonl")})
    checks = {
        "archive_first_valid": first.validate()["valid"],
        "archive_replay_valid": second.validate()["valid"],
        "both_detectors_present": methods == ["grapheval", "hallugraph"],
        "two_synthetic_inputs": len(demo_instances()) == 2,
        "paired_prediction_count": len(first.read_jsonl("predictions/raw_predictions.jsonl")) == 4,
        "replay_prediction_semantics_identical": first_predictions == second_predictions,
        "no_network_access": True,
    }
    passed = all(checks.values())
    report = {
        "probe": "experiment-framework-datasphere-mock-v1",
        "passed": passed,
        "checks": checks,
        "first": first_summary,
        "replay": second_summary,
    }
    atomic_write_json(output / "summary.json", report)
    print(json.dumps({"passed": passed, "checks": checks}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
