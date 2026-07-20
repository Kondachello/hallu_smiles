#!/usr/bin/env python3
"""Summarize paired strict/support/support-critical diagnostic artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        return {row["response_id"]: row for row in csv.DictReader(handle)}


def _theta(path: Path) -> float:
    with (path / "summary_metrics.csv").open(encoding="utf-8", newline="") as handle:
        return float(next(csv.DictReader(handle))["theta"])


def _prediction(row: dict[str, str], theta: float) -> int | None:
    if row.get("unscorable", "False").lower() == "true" or not row.get("H"):
        return None
    return int(float(row["H"]) >= theta)


def _status_counts(rows: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows.values():
        try:
            counts.update(json.loads(row.get("critical_claim_statuses", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError):
            counts["invalid_audit_payload"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", required=True)
    parser.add_argument("--support-dir", required=True)
    parser.add_argument("--critical-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    strict, support, critical = (_rows(Path(value)) for value in (
        args.strict_dir, args.support_dir, args.critical_dir
    ))
    strict_theta, support_theta, critical_theta = (_theta(Path(value)) for value in (
        args.strict_dir, args.support_dir, args.critical_dir
    ))
    changes: Counter[str] = Counter()
    records = []
    for response_id in sorted(set(strict) & set(support) & set(critical)):
        s, u, c = strict[response_id], support[response_id], critical[response_id]
        y = int(c["y"])
        ps, pu, pc = _prediction(s, strict_theta), _prediction(u, support_theta), _prediction(c, critical_theta)
        if ps is None or pc is None:
            continue
        key = (
            "strict_correct" if ps == y else "strict_error",
            "critical_correct" if pc == y else "critical_error",
        )
        changes["_to_".join(key)] += 1
        if ps != pc or pu != pc:
            records.append({
                "response_id": response_id, "y": y,
                "strict_prediction": ps, "support_prediction": pu, "critical_prediction": pc,
                "strict_H": s.get("H"), "support_H": u.get("H"), "critical_H": c.get("H"),
                "critical_claim_statuses": c.get("critical_claim_statuses", "[]"),
            })
    payload = {
        "status": "ready",
        "thresholds": {"strict": strict_theta, "support": support_theta, "support_critical": critical_theta},
        "claim_verdict_counts": _status_counts(critical),
        "strict_to_critical_change_counts": dict(sorted(changes.items())),
        "changed_predictions": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Support-critical diagnostic\n",
        "## Claim verdicts\n",
        "| verdict | count |\n|---|---:|\n",
    ]
    lines.extend(f"| {name} | {count} |\n" for name, count in payload["claim_verdict_counts"].items())
    lines.extend(["\n## Strict → support-critical changes\n", "| transition | count |\n|---|---:|\n"])
    lines.extend(f"| {name} | {count} |\n" for name, count in payload["strict_to_critical_change_counts"].items())
    lines.extend(["\n## Changed predictions\n", "| response | y | strict | support | critical | critical claim verdicts |\n|---:|---:|---:|---:|---:|---|\n"])
    lines.extend(
        f"| {row['response_id']} | {row['y']} | {row['strict_prediction']} | {row['support_prediction']} | "
        f"{row['critical_prediction']} | `{row['critical_claim_statuses']}` |\n"
        for row in records
    )
    output.with_suffix(".md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
