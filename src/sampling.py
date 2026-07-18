"""Deterministic, source-level QA pilot selection and manifest handling.

RAGTruth stores several model responses for each QA source.  A pilot needs one
response per source so a source cannot appear in both training and test data.
The manifest is the contract between strict and support runs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .data import Instance


MANIFEST_VERSION = 1


def select_qa_pilot(
    instances: Iterable[Instance],
    *,
    seed: int = 42,
    train_sources: int = 16,
    test_sources: int = 4,
) -> list[Instance]:
    """Select one QA response per source with fixed split/label quotas.

    The default is 8 factual + 8 hallucinated train responses and 2 + 2 test
    responses.  Candidate models are selected round-robin where possible; an
    SHA256 ordering makes the result independent of input file ordering.
    """
    if train_sources % 2 or test_sources % 2:
        raise ValueError("QA pilot split sizes must be even to balance labels")

    qa = [i for i in instances if i.task == "QA" and i.split in {"train", "test"}]
    quotas = {
        ("train", 0): train_sources // 2,
        ("train", 1): train_sources // 2,
        ("test", 0): test_sources // 2,
        ("test", 1): test_sources // 2,
    }
    used_sources: set[str] = set()
    model_counts: Counter[str] = Counter()
    selected: list[Instance] = []

    # Select positives first within a split so the rarer class cannot be
    # exhausted by a source chosen for a factual response.
    for split in ("train", "test"):
        for y in (1, 0):
            quota = quotas[(split, y)]
            bucket = [i for i in qa if i.split == split and i.y == y]
            selected.extend(
                _pick_bucket(bucket, quota, used_sources, model_counts, seed, split, y)
            )

    selected.sort(key=lambda i: (i.split, i.source_id, i.response_id))
    expected = train_sources + test_sources
    if len(selected) != expected:  # defensive; _pick_bucket raises first
        raise RuntimeError(f"selected {len(selected)} QA responses, expected {expected}")
    return selected


def _pick_bucket(
    candidates: list[Instance],
    quota: int,
    used_sources: set[str],
    model_counts: Counter[str],
    seed: int,
    split: str,
    y: int,
) -> list[Instance]:
    by_model: dict[str, list[Instance]] = defaultdict(list)
    for inst in candidates:
        by_model[inst.gen_model].append(inst)
    for model, rows in by_model.items():
        rows.sort(key=lambda i: _stable_order(seed, split, y, model, i.source_id, i.response_id))

    picked: list[Instance] = []
    while len(picked) < quota:
        available = [
            model for model, rows in by_model.items()
            if any(row.source_id not in used_sources for row in rows)
        ]
        if not available:
            raise ValueError(
                f"cannot select {quota} unique QA sources for split={split!r}, y={y}; "
                f"only {len(picked)} are available"
            )
        # Prefer under-represented generator models, then a stable seeded order.
        model = min(
            available,
            key=lambda m: (model_counts[m], _stable_order(seed, split, y, "model", m)),
        )
        row = next(row for row in by_model[model] if row.source_id not in used_sources)
        picked.append(row)
        used_sources.add(row.source_id)
        model_counts[row.gen_model] += 1
    return picked


def _stable_order(seed: int, *parts: str) -> str:
    payload = "\x00".join([str(seed), *map(str, parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_dict(
    instances: Iterable[Instance], *, seed: int, train_sources: int, test_sources: int
) -> dict:
    rows = sorted(instances, key=lambda i: (i.split, i.source_id, i.response_id))
    return {
        "version": MANIFEST_VERSION,
        "task": "QA",
        "seed": int(seed),
        "quotas": {"train_sources": int(train_sources), "test_sources": int(test_sources)},
        "records": [
            {
                "source_id": i.source_id,
                "response_id": i.response_id,
                "split": i.split,
                "y": int(i.y),
                "gen_model": i.gen_model,
            }
            for i in rows
        ],
    }


def write_manifest(
    path: str | Path,
    instances: Iterable[Instance],
    *,
    seed: int,
    train_sources: int,
    test_sources: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            manifest_dict(
                instances, seed=seed, train_sources=train_sources, test_sources=test_sources
            ),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest_instances(path: str | Path, instances: Iterable[Instance]) -> list[Instance]:
    """Resolve and validate the exact response IDs in a QA pilot manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION or payload.get("task") != "QA":
        raise ValueError(f"unsupported QA pilot manifest: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"manifest {path} has no records")

    by_id = {i.response_id: i for i in instances}
    out: list[Instance] = []
    sources: set[str] = set()
    for rec in records:
        rid = str(rec.get("response_id", ""))
        inst = by_id.get(rid)
        if inst is None:
            raise ValueError(f"manifest response_id={rid!r} does not exist in current data")
        if inst.task != "QA" or inst.source_id != str(rec.get("source_id", "")):
            raise ValueError(f"manifest record for {rid!r} is not the expected QA source")
        if inst.split != rec.get("split") or int(inst.y) != int(rec.get("y")):
            raise ValueError(f"manifest record for {rid!r} no longer matches split/label")
        if inst.source_id in sources:
            raise ValueError(f"manifest selects more than one response for source {inst.source_id}")
        sources.add(inst.source_id)
        out.append(inst)

    _validate_manifest_quotas(payload, out)
    return sorted(out, key=lambda i: (i.split, i.source_id, i.response_id))


def _validate_manifest_quotas(payload: dict, instances: list[Instance]) -> None:
    quotas = payload.get("quotas", {})
    expected = {
        ("train", 0): int(quotas.get("train_sources", 16)) // 2,
        ("train", 1): int(quotas.get("train_sources", 16)) // 2,
        ("test", 0): int(quotas.get("test_sources", 4)) // 2,
        ("test", 1): int(quotas.get("test_sources", 4)) // 2,
    }
    actual = Counter((i.split, int(i.y)) for i in instances)
    if actual != Counter(expected):
        raise ValueError(f"manifest split/label quotas mismatch: expected {expected}, got {dict(actual)}")
