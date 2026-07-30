#!/usr/bin/env python3
"""Growing registry of audit aspects discovered by the case auditor agents.

The base checklist lives in the audit system prompt and never changes during a run.
This registry is the *dynamic* half: after auditing a case, each agent proposes an
aspect the checklist did not ask for.  Accepted aspects are appended here and are
injected into the prompt of every later wave, so the run gets sharper as it goes.

Concurrency contract: worker agents NEVER write here.  They return proposals; the
orchestrator merges them between waves.  That keeps appends single-writer and makes
the provenance log exact.

Storage:
  aspects.jsonl       accepted aspects, append-only, one JSON object per line
  aspects-log.jsonl   every proposal ever seen, accepted or not, with provenance
  aspects.md          human/agent-readable rendering of aspects.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

STATUSES = ("accepted", "duplicate", "refinement", "rejected")


@dataclass(frozen=True)
class Aspect:
    aspect_id: str
    title: str
    definition: str
    how_to_check: str
    why_it_matters: str
    proposed_by_case: str
    proposed_by_agent: str
    wave: int

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Aspect":
        return Aspect(**{f: row[f] for f in Aspect.__dataclass_fields__})


def _slug(title: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", title.upper()).strip("_")
    return slug or "UNNAMED_ASPECT"


def make_aspect_id(title: str, existing: Iterable[str]) -> str:
    """Stable, collision-free English identifier, usable as a tag in the audit output."""
    base = _slug(title)
    taken = set(existing)
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def load_registry(path: Path) -> list[Aspect]:
    if not path.exists():
        return []
    return [
        Aspect.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_markdown(aspects: list[Aspect], *, method: str) -> str:
    header = (
        f"# Динамический реестр аспектов аудита — {method}\n\n"
        "Этот файл пополняется автоматически. Базовый чек-лист живёт в системном\n"
        "промпте и здесь не дублируется. Ниже — только аспекты, которые предложили\n"
        "агенты-аудиторы на предыдущих кейсах.\n\n"
        "**Как использовать:** после прохождения базового чек-листа пройдись по\n"
        "каждому аспекту ниже и укажи, релевантен ли он этому кейсу. Если аспект\n"
        "неприменим — так и напиши, одной строкой. Не растягивай разбор ради объёма.\n\n"
    )
    if not aspects:
        return header + "_Реестр пока пуст: это одна из первых волн прогона._\n"
    parts = [header, f"Всего аспектов: **{len(aspects)}**\n\n---\n\n"]
    for aspect in aspects:
        parts.append(
            f"## {aspect.aspect_id}\n\n"
            f"**{aspect.title}**\n\n"
            f"- **Определение:** {aspect.definition}\n"
            f"- **Как проверять:** {aspect.how_to_check}\n"
            f"- **Почему важно:** {aspect.why_it_matters}\n"
            f"- _Предложен на кейсе `{aspect.proposed_by_case}` (волна {aspect.wave})_\n\n"
        )
    return "".join(parts)


def append_accepted(registry_path: Path, aspect: Aspect) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(aspect), ensure_ascii=False, sort_keys=True) + "\n")


def log_event(log_path: Path, event: dict[str, Any]) -> None:
    if event.get("status") not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {event.get('status')!r}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def sync_markdown(registry_path: Path, markdown_path: Path, *, method: str) -> int:
    aspects = load_registry(registry_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(aspects, method=method), encoding="utf-8")
    return len(aspects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="regenerate aspects.md from aspects.jsonl")
    render.add_argument("--registry", type=Path, required=True)
    render.add_argument("--markdown", type=Path, required=True)
    render.add_argument("--method", default="HalluGraph")

    add = sub.add_parser("add", help="append one accepted aspect and log it")
    add.add_argument("--registry", type=Path, required=True)
    add.add_argument("--log", type=Path, required=True)
    add.add_argument("--markdown", type=Path, required=True)
    add.add_argument("--method", default="HalluGraph")
    for field in ("title", "definition", "how-to-check", "why-it-matters", "case", "agent"):
        add.add_argument(f"--{field}", required=True)
    add.add_argument("--wave", type=int, required=True)

    args = parser.parse_args()

    if args.command == "render":
        count = sync_markdown(args.registry, args.markdown, method=args.method)
        print(f"rendered {count} aspect(s) into {args.markdown}")
        return

    existing = load_registry(args.registry)
    aspect = Aspect(
        aspect_id=make_aspect_id(args.title, (a.aspect_id for a in existing)),
        title=args.title,
        definition=args.definition,
        how_to_check=getattr(args, "how_to_check"),
        why_it_matters=getattr(args, "why_it_matters"),
        proposed_by_case=args.case,
        proposed_by_agent=args.agent,
        wave=args.wave,
    )
    append_accepted(args.registry, aspect)
    log_event(args.log, {**asdict(aspect), "status": "accepted"})
    sync_markdown(args.registry, args.markdown, method=args.method)
    print(aspect.aspect_id)


if __name__ == "__main__":
    main()
