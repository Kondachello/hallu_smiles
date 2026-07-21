"""Pinned-file RAGTruth audit, deterministic source sampling and gold-safe materialization."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from src.data import build_context_query

from ..artifacts import atomic_write_json, atomic_write_jsonl
from ..contracts import assert_no_gold

DATASET_NAME = "RAGTruth"
SAMPLER_VERSION = "ragtruth-source-sampler-v1"
MATERIALIZER_VERSION = "ragtruth-materializer-v1"
ONE_INSTANCE_MATERIALIZER_VERSION = "ragtruth-one-instance-materializer-v1"
OFFICIAL_RAW_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_info(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": sha256_bytes(content),
        "bytes": len(content),
        "line_count": content.count(b"\n") + (1 if content and not content.endswith(b"\n") else 0),
    }


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Yield source line text and parsed JSON without mutating the raw representation."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.rstrip("\r\n")
            if stripped:
                yield line_number, stripped, json.loads(stripped)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line, parsed in iter_jsonl(path):
        records.append(
            {
                "line_number": line_number,
                "raw_line": raw_line,
                "raw_record_sha256": sha256_bytes(raw_line.encode("utf-8")),
                "record": parsed,
            }
        )
    return records


def audit_dataset(source_info_path: str | Path, response_path: str | Path, *, revision: str) -> dict[str, Any]:
    """Perform a non-mutating data audit required before sample materialization."""
    sources = load_records(source_info_path)
    responses = load_records(response_path)
    source_ids = [str(item["record"].get("source_id", "")) for item in sources]
    response_ids = [str(item["record"].get("id", "")) for item in responses]
    source_set = set(source_ids)
    response_source_ids = [str(item["record"].get("source_id", "")) for item in responses]
    response_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in responses:
        response_by_source[str(item["record"].get("source_id", ""))].append(item["record"])

    invalid_offsets = 0
    text_mismatches = 0
    label_types: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    models: Counter[str] = Counter()
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for item in sources:
        tasks[str(item["record"].get("task_type", "unknown"))] += 1
    for item in responses:
        response = item["record"]
        qualities[str(response.get("quality", "unknown"))] += 1
        models[str(response.get("model", "unknown"))] += 1
        splits_by_source[str(response.get("source_id", ""))].add(str(response.get("split", "")))
        answer = str(response.get("response", ""))
        for label in response.get("labels") or []:
            label_types[str(label.get("label_type", "unknown"))] += 1
            try:
                start, end = int(label["start"]), int(label["end"])
            except (KeyError, TypeError, ValueError):
                invalid_offsets += 1
                continue
            if start < 0 or end < start or end > len(answer):
                invalid_offsets += 1
            elif answer[start:end] != str(label.get("text", "")):
                text_mismatches += 1

    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_revision": revision,
        "ragtruth_adapter_version": MATERIALIZER_VERSION,
        "source_info": _file_info(Path(source_info_path)),
        "response": _file_info(Path(response_path)),
        "audit": {
            "n_unique_source_ids": len(source_set),
            "n_unique_response_ids": len(set(response_ids)),
            "n_orphan_responses": sum(source_id not in source_set for source_id in response_source_ids),
            "n_duplicate_source_ids": len(source_ids) - len(set(source_ids)),
            "n_duplicate_response_ids": len(response_ids) - len(set(response_ids)),
            "n_responses_per_source_distribution": dict(sorted(Counter(map(len, response_by_source.values())).items())),
            "n_source_ids_crossing_splits": sum(len(splits) > 1 for splits in splits_by_source.values()),
            "n_invalid_offsets": invalid_offsets,
            "n_label_text_offset_mismatches": text_mismatches,
            "task_counts": dict(sorted(tasks.items())),
            "model_counts": dict(sorted(models.items())),
            "quality_counts": dict(sorted(qualities.items())),
            "label_type_counts": dict(sorted(label_types.items())),
            "encoding": "utf-8",
            "newline_convention": "preserved_in_raw_files",
        },
    }
    manifest["dataset_manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def write_data_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(manifest))


def _download_atomic(url: str, destination: Path) -> None:
    """Download a single immutable object to a unique temporary file then rename it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".download", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"unexpected HTTP status for {url}: {response.status}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not Path(temporary).stat().st_size:
            raise RuntimeError(f"downloaded empty file from {url}")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def fetch_dataset(*, data_root: str | Path, revision: str, raw_base: str = OFFICIAL_RAW_BASE) -> dict[str, Any]:
    """Fetch official RAGTruth files at an immutable commit; never accepts floating ``main``.

    This function performs network I/O only when explicitly called by ``data fetch``.  The
    framework itself and all tests remain offline.
    """
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("RAGTruth revision must be an exact 40-character lowercase Git commit SHA")
    destination = Path(data_root) / "raw" / revision
    source_path = destination / "source_info.jsonl"
    response_path = destination / "response.jsonl"
    if not source_path.exists():
        _download_atomic(f"{raw_base.rstrip('/')}/{revision}/dataset/source_info.jsonl", source_path)
    if not response_path.exists():
        _download_atomic(f"{raw_base.rstrip('/')}/{revision}/dataset/response.jsonl", response_path)
    manifest = audit_dataset(source_path, response_path, revision=revision)
    manifest["dataset_repository"] = "https://github.com/ParticleMedia/RAGTruth"
    manifest["dataset_download_url"] = f"{raw_base.rstrip('/')}/{revision}/dataset"
    manifest["license"] = "MIT"
    manifest.pop("dataset_manifest_sha256", None)
    manifest["dataset_manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    write_data_manifest(destination / "data_manifest.json", manifest)
    return manifest


def _seeded_key(seed: int, *parts: str) -> str:
    return sha256_bytes("\x00".join([str(seed), *parts]).encode("utf-8"))


def create_source_sample_manifest(
    source_info_path: str | Path,
    response_path: str | Path,
    *,
    dataset_manifest: Mapping[str, Any],
    split: str,
    seed: int,
    n_sources: int | None = None,
    tasks: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    include_all_responses_per_source: bool = True,
    purpose: str = "development",
) -> dict[str, Any]:
    """Create a deterministic no-gold source-level sample manifest.

    Sampling strata only use task and generator model.  Gold labels are intentionally not
    read for selection, even though the raw response records contain them.
    """
    source_rows = {str(row["record"]["source_id"]): row["record"] for row in load_records(source_info_path)}
    wanted_tasks = set(tasks or ())
    wanted_models = set(models or ())
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in load_records(response_path):
        response = item["record"]
        source_id = str(response.get("source_id", ""))
        source = source_rows.get(source_id)
        if source is None or str(response.get("split", "")) != split:
            continue
        if wanted_tasks and str(source.get("task_type", "")) not in wanted_tasks:
            continue
        if wanted_models and str(response.get("model", "")) not in wanted_models:
            continue
        eligible[source_id].append(response)

    ordered_sources = sorted(eligible, key=lambda source_id: _seeded_key(seed, split, source_id))
    if n_sources is not None:
        if n_sources <= 0:
            raise ValueError("n_sources must be positive")
        ordered_sources = ordered_sources[:n_sources]
    selected_responses: list[dict[str, Any]] = []
    for source_id in ordered_sources:
        rows = sorted(eligible[source_id], key=lambda row: str(row.get("id", "")))
        selected_responses.extend(rows if include_all_responses_per_source else rows[:1])

    manifest = {
        "sample_manifest_version": 1,
        "sampler_version": SAMPLER_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_revision": dataset_manifest["dataset_revision"],
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
        "purpose": purpose,
        "split": split,
        "sampling_unit": "source_id",
        "seed": int(seed),
        "filters": {"tasks": sorted(wanted_tasks), "models": sorted(wanted_models)},
        "include_all_responses_per_source": bool(include_all_responses_per_source),
        "gold_used_for_selection": False,
        "prediction_used_for_selection": False,
        "selected_source_ids": ordered_sources,
        "selected_response_ids": [str(row["id"]) for row in selected_responses],
        "selection_order": ordered_sources,
        "counts": {
            "n_sources": len(ordered_sources),
            "n_responses": len(selected_responses),
            "task": dict(sorted(Counter(str(source_rows[s]["task_type"]) for s in ordered_sources).items())),
            "generator_model": dict(sorted(Counter(str(row.get("model", "unknown")) for row in selected_responses).items())),
        },
    }
    manifest["sample_manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def write_sample_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(manifest))


def materialize_one_response_no_gold(
    source_info_path: str | Path,
    response_path: str | Path,
    *,
    response_id: str,
) -> dict[str, Any]:
    """Build one detector-safe RAGTruth record selected by an explicit response id.

    This is intentionally different from the source-level sampler.  It is for a
    plumbing probe, not a scientific sample: the caller names exactly one response
    and this function copies only detector-allowed fields from the raw source and
    response records.  In particular, neither ``labels`` nor ``quality`` is read
    into the returned object, logged, or used for selection.
    """
    requested_id = str(response_id).strip()
    if not requested_id:
        raise ValueError("response_id must be a non-empty string")

    response_row: dict[str, Any] | None = None
    for line_number, raw_line, response in iter_jsonl(response_path):
        if str(response.get("id", "")) == requested_id:
            response_row = {
                "line_number": line_number,
                "raw_record_sha256": sha256_bytes(raw_line.encode("utf-8")),
                "record": response,
            }
            break
    if response_row is None:
        raise ValueError(f"RAGTruth response_id was not found: {requested_id!r}")

    response = response_row["record"]
    source_id = str(response.get("source_id", ""))
    source_row: dict[str, Any] | None = None
    for line_number, raw_line, source in iter_jsonl(source_info_path):
        if str(source.get("source_id", "")) == source_id:
            source_row = {
                "line_number": line_number,
                "raw_record_sha256": sha256_bytes(raw_line.encode("utf-8")),
                "record": source,
            }
            break
    if source_row is None:
        raise ValueError(f"RAGTruth response {requested_id!r} has no source record {source_id!r}")

    source = source_row["record"]
    context, query = build_context_query(source)
    record_id_material = ":".join(
        (
            ONE_INSTANCE_MATERIALIZER_VERSION,
            source_row["raw_record_sha256"],
            response_row["raw_record_sha256"],
        )
    )
    dataset_record_id = sha256_bytes(record_id_material.encode("utf-8"))[:24]
    input_record = {
        "dataset_record_id": dataset_record_id,
        "source_id": source_id,
        "response_id": requested_id,
        "split": str(response.get("split", "")),
        "context_raw": context,
        "query_raw": query,
        "response_raw": str(response.get("response", "")),
        "original_prompt_raw": str(source.get("prompt", "")),
        "context_hash": sha256_bytes(context.encode("utf-8")),
        "query_hash": sha256_bytes((query or "").encode("utf-8")),
        "response_hash": sha256_bytes(str(response.get("response", "")).encode("utf-8")),
        "source_record_sha256": source_row["raw_record_sha256"],
        "response_record_sha256": response_row["raw_record_sha256"],
        "source_line_number": source_row["line_number"],
        "response_line_number": response_row["line_number"],
        "context_construction_policy": "ragtruth-task-native-v1",
        "query_construction_policy": "ragtruth-task-native-v1",
        "context_document_ids": [f"source:{source_id}"],
        "context_document_order": [f"source:{source_id}"],
        "metadata": {
            "dataset_record_id": dataset_record_id,
            "task": str(source.get("task_type", "unknown")),
            "source_dataset": str(source.get("source", "unknown")),
            "generator_model": str(response.get("model", "unknown")),
            "generator_temperature": response.get("temperature"),
            "context_document_ids": [f"source:{source_id}"],
            "context_document_order": [f"source:{source_id}"],
        },
        "gold_access_state": "hidden",
    }
    assert_no_gold(input_record)
    return input_record


def materialize_subset(
    source_info_path: str | Path,
    response_path: str | Path,
    *,
    dataset_manifest: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Materialize immutable no-gold inputs and separate evaluation-only gold files."""
    selected_ids = set(map(str, sample_manifest["selected_response_ids"]))
    source_rows = {str(row["record"]["source_id"]): row for row in load_records(source_info_path)}
    no_gold: list[dict[str, Any]] = []
    response_gold: list[dict[str, Any]] = []
    gold_spans: list[dict[str, Any]] = []

    for response_row in load_records(response_path):
        response = response_row["record"]
        response_id = str(response.get("id", ""))
        if response_id not in selected_ids:
            continue
        source_id = str(response.get("source_id", ""))
        source_row = source_rows.get(source_id)
        if source_row is None:
            raise ValueError(f"selected response {response_id!r} has no source record")
        source = source_row["record"]
        context, query = build_context_query(source)
        dataset_record_id = sha256_bytes(f"{dataset_manifest['dataset_manifest_sha256']}:{response_id}".encode("utf-8"))[:24]
        input_record = {
            "dataset_record_id": dataset_record_id,
            "source_id": source_id,
            "response_id": response_id,
            "split": str(response.get("split", "")),
            "context_raw": context,
            "query_raw": query,
            "response_raw": str(response.get("response", "")),
            "original_prompt_raw": str(source.get("prompt", "")),
            "context_hash": sha256_bytes(context.encode("utf-8")),
            "query_hash": sha256_bytes((query or "").encode("utf-8")),
            "response_hash": sha256_bytes(str(response.get("response", "")).encode("utf-8")),
            "source_record_sha256": source_row["raw_record_sha256"],
            "response_record_sha256": response_row["raw_record_sha256"],
            "context_construction_policy": "ragtruth-task-native-v1",
            "query_construction_policy": "ragtruth-task-native-v1",
            "context_document_ids": [f"source:{source_id}"],
            "context_document_order": [f"source:{source_id}"],
            "metadata": {
                "dataset_record_id": dataset_record_id,
                "task": str(source.get("task_type", "unknown")),
                "source_dataset": str(source.get("source", "unknown")),
                "generator_model": str(response.get("model", "unknown")),
                "generator_temperature": response.get("temperature"),
                "context_document_ids": [f"source:{source_id}"],
                "context_document_order": [f"source:{source_id}"],
            },
            "gold_access_state": "hidden",
        }
        assert_no_gold(input_record)
        no_gold.append(input_record)

        labels = list(response.get("labels") or [])
        response_gold.append(
            {
                "response_id": response_id,
                "source_id": source_id,
                "gold_response_label": int(bool(labels)),
                "n_gold_spans_raw": len(labels),
                "quality_raw": str(response.get("quality", "")),
                "gold_access_state": "joined_for_evaluation",
            }
        )
        for index, label in enumerate(labels):
            gold_spans.append(
                {
                    "gold_span_id": f"{response_id}:gold:{index}",
                    "response_id": response_id,
                    "source_id": source_id,
                    "start": label.get("start"),
                    "end": label.get("end"),
                    "text": label.get("text"),
                    "label_type": label.get("label_type"),
                    "due_to_null": label.get("due_to_null"),
                    "implicit_true": label.get("implicit_true"),
                    "meta": label.get("meta"),
                    "raw_label_json": canonical_json(label),
                    "gold_access_state": "joined_for_evaluation",
                }
            )

    if len(no_gold) != len(selected_ids):
        found = {row["response_id"] for row in no_gold}
        raise ValueError(f"sample references absent response ids: {sorted(selected_ids - found)}")
    output = Path(output_dir)
    gold_dir = output / "gold"
    inputs_path = output / "instances.no_gold.jsonl"
    response_gold_path = gold_dir / "response_gold.jsonl"
    gold_spans_path = gold_dir / "gold_spans.jsonl"
    atomic_write_jsonl(inputs_path, sorted(no_gold, key=lambda row: row["response_id"]))
    atomic_write_jsonl(response_gold_path, sorted(response_gold, key=lambda row: row["response_id"]))
    atomic_write_jsonl(gold_spans_path, sorted(gold_spans, key=lambda row: row["gold_span_id"]))
    atomic_write_json(output / "sample_manifest.json", dict(sample_manifest))
    atomic_write_json(output / "data_manifest.json", dict(dataset_manifest))
    return {"instances": inputs_path, "response_gold": response_gold_path, "gold_spans": gold_spans_path}
