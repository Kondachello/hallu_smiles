#!/usr/bin/env python3
"""Turn a typed-metric-pass artifact into the per-record ledger table.

A run that dies on quota or wall-clock still archives everything it finished, so
the interesting question after any run is "which records actually completed, and
with what scores". This reads that straight out of the artifact -- tarball or
already-extracted directory -- and prints a Markdown table ready for the ledger.

Entity/type counts come from the frozen typing registries, so the table also
shows how much typing work each record took.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _run_dir(source: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a directory holding the run artifacts, extracting a tarball if needed."""
    if source.is_dir():
        return _locate(source), None
    holder = tempfile.TemporaryDirectory()
    with tarfile.open(source) as archive:
        # Skip the vendored runtime deps: they are most of the archive and none of the data.
        members = [m for m in archive.getmembers() if "/pydeps/" not in m.name]
        archive.extractall(holder.name, members=members, filter="data")
    return _locate(Path(holder.name)), holder


def _locate(root: Path) -> Path:
    if (root / "typed_metrics.jsonl").exists():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "typed_metrics.jsonl").exists():
            return child
    raise SystemExit(f"no typed_metrics.jsonl under {root}")


def _typing_effort(run: Path) -> dict[str, tuple[int, int, int]]:
    """response_id -> (entities typed, types invented, NLI checks)."""
    effort: dict[str, tuple[int, int, int]] = {}
    registry_dir = run / "typing-cache" / "frozen_registry"
    if not registry_dir.is_dir():
        return effort
    for path in registry_dir.glob("*.json"):
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_id = str(registry.get("source_id", ""))
        if source_id:
            effort[source_id] = (
                len(registry.get("assignments", [])),
                len(registry.get("types", [])),
                len(registry.get("nli_results", [])),
            )
    return effort


def _cell(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="run tarball or extracted run directory")
    args = parser.parse_args()

    run, holder = _run_dir(Path(args.artifact))
    try:
        rows = [json.loads(line) for line in (run / "typed_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        effort = _typing_effort(run)
        summary_path = run / "typed_metric_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        statuses: dict[str, int] = {}
        for row in rows:
            key = str(row.get("status"))
            statuses[key] = statuses.get(key, 0) + 1

        print(f"Прогон: `{run.name}`")
        print(f"Записей в артефакте: **{len(rows)}** — " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))
        if summary:
            print(f"Заявлено к обработке (`selected`): {summary.get('selected')}, батчей: {summary.get('batches')}, alpha: {summary.get('alpha')}")
        print()
        print("| response_id | status | eg_type | cfi_type | raw_score | вершин | рёбер | сущностей | типов | NLI |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for row in sorted(rows, key=lambda r: str(r.get("response_id"))):
            comp = row.get("components") or {}
            entities, types, nli = effort.get(str(row.get("response_id")), (0, 0, 0))
            print(
                f"| {row.get('response_id')} | {row.get('status')} "
                f"| {_cell(comp.get('eg_type'))} | {_cell(comp.get('cfi_type'))} | {_cell(comp.get('raw_score') or row.get('raw_score'))} "
                f"| {comp.get('total_vertices', '—')} | {comp.get('total_edges', '—')} "
                f"| {entities or '—'} | {types or '—'} | {nli or '—'} |"
            )

        ok_scores = [r.get("raw_score") for r in rows if r.get("status") == "ok" and isinstance(r.get("raw_score"), (int, float))]
        if ok_scores:
            print()
            print(f"Средний raw_score по ok: **{sum(ok_scores) / len(ok_scores):.4f}** "
                  f"(мин {min(ok_scores):.4f}, макс {max(ok_scores):.4f})")
        print()
        print("Список ID одной строкой для прогона другими методами:")
        print("`" + ", ".join(str(r.get("response_id")) for r in sorted(rows, key=lambda r: str(r.get("response_id")))) + "`")
    finally:
        if holder is not None:
            holder.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
