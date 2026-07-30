#!/usr/bin/env python3
"""Assemble per-case agent audits into FILE-1, the dry per-case audit report.

FILE-1 is deliberately conclusion-free: it is the raw material that the
aggregation stage clusters into FILE-2.  This command only concatenates and
indexes what the auditor agents wrote; it adds no analysis of its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MISSING = "—"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["case_id"]): row for row in rows}


def load_summaries(path: Path) -> dict[str, dict[str, Any]]:
    """Structured worker returns, one JSON object per line, keyed by case_id."""
    if not path or not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["case_id"]): row for row in rows}


def _index_table(case_ids: list[str], manifest: dict[str, Any], summaries: dict[str, Any]) -> str:
    head = (
        "| Кейс | Split | Gold | Исход HG | Балл | Порог | Первопричина | Тег | Вклад KGGen |\n"
        "|---|---|---:|---|---:|---:|---|---|---|\n"
    )
    rows = []
    for case_id in case_ids:
        m = manifest.get(case_id, {})
        s = summaries.get(case_id, {})
        cause = (s.get("primary_root_cause") or {})
        score = m.get("hallugraph_score")
        threshold = m.get("hallugraph_threshold")
        rows.append(
            f"| `{case_id}` "
            f"| {m.get('split', MISSING)} "
            f"| {m.get('gold_response_label', MISSING)} "
            f"| {m.get('hallugraph_outcome', MISSING)} "
            f"| {MISSING if score is None else f'{float(score):.4f}'} "
            f"| {MISSING if threshold is None else f'{float(threshold):.4f}'} "
            f"| {cause.get('coarse_class', MISSING)} "
            f"| `{cause.get('fine_tag', MISSING)}` "
            f"| {s.get('kggen_contribution', MISSING)} |"
        )
    return head + "\n".join(rows) + "\n"


def build(
    *,
    audits_dir: Path,
    manifest_path: Path,
    summaries_path: Path | None,
    suffix: str,
    method_label: str,
) -> tuple[str, list[str], list[str]]:
    manifest = load_manifest(manifest_path)
    summaries = load_summaries(summaries_path) if summaries_path else {}

    found = sorted(audits_dir.glob(f"*.{suffix}.md"), key=lambda p: p.name)
    case_ids = [p.name[: -len(f".{suffix}.md")] for p in found]
    expected = sorted(manifest)
    missing = [c for c in expected if c not in set(case_ids)]

    parts = [
        f"# FILE-1 — покейсовый аудит ошибок {method_label}\n\n",
        "Сухой разбор каждого кейса, собранный из отчётов агентов-аудиторов.\n"
        "Выводов и рекомендаций здесь нет — они собираются в FILE-2.\n\n",
        f"- Кейсов в отчёте: **{len(case_ids)}**\n",
        f"- Кейсов в манифесте: **{len(expected)}**\n",
    ]
    if missing:
        parts.append(f"- **Отсутствуют отчёты по {len(missing)} кейсам:** {', '.join(f'`{c}`' for c in missing)}\n")
    parts.append("\n## Индекс\n\n")
    parts.append(_index_table(case_ids, manifest, summaries))
    parts.append("\n---\n\n")

    for path, case_id in zip(found, case_ids):
        body = path.read_text(encoding="utf-8").strip()
        parts.append(f"<a id=\"case-{case_id}\"></a>\n\n{body}\n\n---\n\n")

    return "".join(parts), case_ids, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audits-dir", type=Path, required=True, help="directory of <case>.<suffix>.md agent reports")
    parser.add_argument("--manifest", type=Path, required=True, help="audit-manifest.jsonl from the exporter")
    parser.add_argument("--summaries", type=Path, help="JSONL of structured worker returns (optional, enriches the index)")
    parser.add_argument("--output", type=Path, required=True, help="path of FILE-1")
    parser.add_argument("--suffix", default="hg", help="per-case file suffix: hg or ge (default: hg)")
    parser.add_argument("--method-label", default="HalluGraph")
    args = parser.parse_args()

    report, case_ids, missing = build(
        audits_dir=args.audits_dir,
        manifest_path=args.manifest,
        summaries_path=args.summaries,
        suffix=args.suffix,
        method_label=args.method_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"FILE-1: {len(case_ids)} case(s) -> {args.output}")
    if missing:
        print(f"WARNING: {len(missing)} case(s) in the manifest have no audit: {', '.join(missing[:10])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
