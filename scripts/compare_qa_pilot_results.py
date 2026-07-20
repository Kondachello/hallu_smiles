#!/usr/bin/env python3
"""Write one compact strict/support/support-critical comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_summary(run_dir: Path) -> dict[str, str]:
    path = run_dir / "summary_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", required=True)
    parser.add_argument("--support-dir", required=True)
    parser.add_argument("--critical-dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    strict_dir, support_dir = Path(args.strict_dir), Path(args.support_dir)
    strict, support = _read_summary(strict_dir), _read_summary(support_dir)
    comparison: dict[str, object] = {
        "strict": {
            "alpha": strict.get("alpha"), "theta": strict.get("theta"),
            "test_auc": strict.get("overall_AUC_exclude_unscorable"),
            "test_f1": strict.get("overall_F1"),
        },
        "support": {
            "alpha": support.get("alpha"), "theta": support.get("theta"),
            "test_auc": support.get("overall_AUC_exclude_unscorable"),
            "test_f1": support.get("overall_F1"),
        },
        "strict_dir": str(strict_dir), "support_dir": str(support_dir),
    }
    if args.critical_dir:
        critical_dir = Path(args.critical_dir)
        critical = _read_summary(critical_dir)
        comparison["support_critical"] = {
            "alpha": critical.get("alpha"), "theta": critical.get("theta"),
            "beta": critical.get("beta"), "top_k": critical.get("top_k"),
            "unknown_risk": critical.get("unknown_risk"),
            "test_auc": critical.get("overall_AUC_exclude_unscorable"),
            "test_f1": critical.get("overall_F1"),
        }
        comparison["critical_dir"] = str(critical_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    rows = (
        "# QA pilot comparison\n\n"
        "| detector | alpha | theta | test ROC-AUC | test F1 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| strict | {comparison['strict']['alpha']} | {comparison['strict']['theta']} | "
        f"{comparison['strict']['test_auc']} | {comparison['strict']['test_f1']} |\n"
        f"| support | {comparison['support']['alpha']} | {comparison['support']['theta']} | "
        f"{comparison['support']['test_auc']} | {comparison['support']['test_f1']} |\n"
    )
    critical = comparison.get("support_critical")
    if isinstance(critical, dict):
        rows += (
            f"| support-critical | {critical['alpha']} | {critical['theta']} | "
            f"{critical['test_auc']} | {critical['test_f1']} |\n"
        )
    output.with_suffix(".md").write_text(rows, encoding="utf-8")


if __name__ == "__main__":
    main()
