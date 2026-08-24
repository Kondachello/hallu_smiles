#!/usr/bin/env python3
"""Summarise the controlled Llama run without exposing prompts, graphs or keys."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llama31_eval import QUARANTINED_SOURCE_ID  # noqa: E402


def _method(directory: Path) -> tuple[dict, pd.DataFrame]:
    metrics = pd.read_csv(directory / "metrics.csv")
    summary = pd.read_csv(directory / "summary_metrics.csv").iloc[0].to_dict()
    tuning = json.loads((directory / "tuning.json").read_text(encoding="utf-8"))
    extraction = json.loads((directory / "extraction_summary.json").read_text(encoding="utf-8"))
    if extraction.get("status") != "ready_with_explicit_exclusions":
        raise SystemExit(f"{directory.name} extraction is incomplete")
    if extraction.get("excluded_source_ids") != [QUARANTINED_SOURCE_ID]:
        raise SystemExit(f"{directory.name} extraction quarantine changed")
    test = metrics[metrics["split"] == "test"].copy()
    scorable = test[~test["unscorable"].astype(bool)].copy()
    return {
        "relation_mode": str(summary["relation_mode"]),
        "test_scored": int(len(test)),
        "test_scorable": int(len(scorable)),
        "test_scorable_positive": int(scorable["y"].sum()),
        "test_scorable_negative": int(len(scorable) - scorable["y"].sum()),
        "test_unscorable_Va0": int(test["unscorable"].astype(bool).sum()),
        "roc_auc": float(summary["overall_AUC_exclude_unscorable"]),
        "roc_auc_ci95": [
            float(summary["overall_AUC_ci95_lo"]),
            float(summary["overall_AUC_ci95_hi"]),
        ],
        "f1": float(summary["overall_F1"]),
        "threshold": float(tuning["theta"]),
        "parameters": {
            key: tuning[key]
            for key in ("alpha", "beta", "top_k", "unknown_risk", "tau_e", "tau_r")
            if key in tuning
        },
        "extraction_failures": len(extraction.get("failures", [])),
        "explicit_exclusions": len(extraction.get("excluded_records", [])),
    }, scorable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict-dir", required=True)
    parser.add_argument("--support-critical-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 750:
        raise SystemExit("controlled Llama manifest must contain 750 records")
    test_labels = Counter(
        int(record["y"]) for record in records if record.get("split") == "test"
    )
    if test_labels != Counter({0: 100, 1: 50}):
        raise SystemExit(f"unexpected pre-graph test balance: {dict(test_labels)}")
    analysed_train = [
        record for record in records
        if record.get("split") == "train" and record.get("source_id") != QUARANTINED_SOURCE_ID
    ]
    if len(analysed_train) != 599:
        raise SystemExit("controlled train analysis count must be 599 after quarantine")
    strict, strict_scorable = _method(Path(args.strict_dir))
    critical, critical_scorable = _method(Path(args.support_critical_dir))
    strict_ids = set(strict_scorable["response_id"])
    critical_ids = set(critical_scorable["response_id"])
    if strict_ids != critical_ids:
        raise SystemExit("strict and support-critical have different scorable test denominators")
    payload = {
        "protocol": "hallu-ragtruth-llama31-controlled-result-v1",
        "scientific_scope": (
            "controlled single-model evaluation; not a replacement for the published "
            "historical 750-QA result"
        ),
        "mixed_identity": (
            "historical frozen C/Q graphs plus current-gateway answer graphs and "
            "support-critical verification artifacts"
        ),
        "pre_graph_test_balance": {"positive": 50, "negative": 100, "total": 150},
        "analysed_training_sources": len(analysed_train),
        "quarantined_source_id": QUARANTINED_SOURCE_ID,
        "common_scorable_test_denominator": len(strict_ids),
        "methods": {"strict": strict, "support_critical": critical},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "common_scorable_test_denominator": payload["common_scorable_test_denominator"],
        "strict_auc": strict["roc_auc"],
        "support_critical_auc": critical["roc_auc"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
