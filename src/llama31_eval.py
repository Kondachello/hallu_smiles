"""Inputs and immutable reference artifacts for the controlled Llama-3.1-8B run.

This module deliberately keeps the new response labels and the historical
context/query graphs outside the normal RAGTruth cache contract.  The latter
were produced through a different gateway identity, so treating them as a
read-through cache would make a current-run cache hit look scientifically
equivalent to a current extraction.  A frozen artifact makes that mixed
identity explicit instead.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import Instance, build_context_query, load_sources
from .extract import Graph


LLAMA31_MANIFEST_PROTOCOL = "hallu-ragtruth-llama31-controlled-manifest-v1"
FROZEN_REFERENCE_PROTOCOL = "hallu-ragtruth-frozen-reference-graphs-v1"
HISTORICAL_GATEWAY_MANIFEST_SHA256 = (
    "9407591410b215ba41478290526acd3a4ea32f3dd70a63076c6394c95e37c845"
)
HISTORICAL_LLM_RUNTIME_FINGERPRINT = (
    "vertex-gateway:9ba169c4f2de8a246c756948b24a3860a54cc419957004d9ed351c2ad538b3bd"
)
DEFAULT_RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
LLAMA31_ID_PREFIX = "llama31_8b_"
QUARANTINED_SOURCE_ID = "12448"


@dataclass(frozen=True)
class Llama31Response:
    """A validated row from the supplied, frozen annotation CSV."""

    response_id: str
    source_id: str
    response: str
    prompt: str
    y: int
    annotation_model: str


def canonical_json_sha256(value: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible object."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _prompt_normalize(text: str | None) -> str:
    """Whitespace-normalise only for prompt containment validation."""
    return re.sub(r"\s+", " ", text or "").strip()


def _require_prompt_contains(prompt: str, text: str | None, *, label: str, source_id: str) -> None:
    canonical = _prompt_normalize(text)
    if canonical and canonical not in _prompt_normalize(prompt):
        raise ValueError(
            f"Llama-3.1 CSV prompt for source_id={source_id!r} does not contain canonical {label}"
        )


def load_llama31_csv(path: str | Path) -> dict[str, Llama31Response]:
    """Read and fully validate the supplied response-level annotation CSV.

    The source-id suffix is intentionally part of the external contract: no
    order-dependent pairing is allowed between the CSV and RAGTruth.
    """
    path = Path(path)
    required = {
        "id",
        "generated_response",
        "prompt",
        "hallucination",
        "annotation_reason",
        "annotation_raw",
        "annotation_model",
    }
    by_source: dict[str, Llama31Response] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Llama-3.1 CSV is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            response_id = str(row.get("id") or "").strip()
            if not response_id.startswith(LLAMA31_ID_PREFIX):
                raise ValueError(
                    f"CSV row {row_number} id must start with {LLAMA31_ID_PREFIX!r}"
                )
            source_id = response_id.removeprefix(LLAMA31_ID_PREFIX)
            if not source_id or not source_id.isdigit():
                raise ValueError(f"CSV row {row_number} has invalid Llama source id")
            raw_label = str(row.get("hallucination") or "").strip()
            if raw_label not in {"0", "1"}:
                raise ValueError(f"CSV row {row_number} hallucination must be binary 0/1")
            response = str(row.get("generated_response") or "").strip()
            prompt = str(row.get("prompt") or "").strip()
            annotation_model = str(row.get("annotation_model") or "").strip()
            if not response:
                raise ValueError(f"CSV row {row_number} has an empty generated_response")
            if not prompt:
                raise ValueError(f"CSV row {row_number} has an empty prompt")
            if not annotation_model:
                raise ValueError(f"CSV row {row_number} has no annotation_model")
            if source_id in by_source:
                raise ValueError(f"CSV contains duplicate source_id={source_id!r}")
            by_source[source_id] = Llama31Response(
                response_id=response_id,
                source_id=source_id,
                response=response,
                prompt=prompt,
                y=int(raw_label),
                annotation_model=annotation_model,
            )
    if len(by_source) != 750:
        raise ValueError(f"Llama-3.1 CSV must contain exactly 750 rows, got {len(by_source)}")
    labels = Counter(row.y for row in by_source.values())
    if labels != Counter({0: 494, 1: 256}):
        raise ValueError(
            "Llama-3.1 CSV label counts must be 494 non-hallucinated and 256 hallucinated, "
            f"got {dict(labels)}"
        )
    models = Counter(row.annotation_model for row in by_source.values())
    if models != Counter({"openai/gpt-4o": 750}):
        raise ValueError(
            "Llama-3.1 CSV annotation_model must be openai/gpt-4o for all rows, "
            f"got {dict(models)}"
        )
    return by_source


def _load_historical_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if payload.get("version") != 1 or payload.get("task") != "QA" or not isinstance(records, list):
        raise ValueError(f"unsupported historical QA manifest: {path}")
    if len(records) != 750:
        raise ValueError(f"historical QA manifest must contain 750 rows, got {len(records)}")
    source_ids = [str(record.get("source_id") or "") for record in records]
    if len(set(source_ids)) != len(source_ids) or any(not source_id for source_id in source_ids):
        raise ValueError("historical QA manifest must select exactly one response per source")
    splits = Counter(str(record.get("split") or "") for record in records)
    if splits != Counter({"train": 600, "test": 150}):
        raise ValueError(f"historical QA manifest split sizes changed: {dict(splits)}")
    return records


def build_llama31_instances(
    csv_path: str | Path,
    data_dir: str | Path,
    historical_manifest: str | Path,
) -> tuple[list[Instance], dict[str, Any]]:
    """Build the controlled one-response-per-source Llama instance set.

    Historical labels and answer text are never used.  The historical manifest
    contributes only immutable source membership and split membership.
    """
    csv_rows = load_llama31_csv(csv_path)
    historical_records = _load_historical_records(historical_manifest)
    historical_sources = {str(record["source_id"]) for record in historical_records}
    if set(csv_rows) != historical_sources:
        missing = sorted(historical_sources - set(csv_rows))
        extra = sorted(set(csv_rows) - historical_sources)
        raise ValueError(
            "Llama-3.1 CSV source coverage does not match historical 750-source manifest; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    sources = load_sources(data_dir)
    instances: list[Instance] = []
    for record in historical_records:
        source_id = str(record["source_id"])
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"RAGTruth source_id={source_id!r} is absent from the pinned source snapshot")
        if source.get("task_type") != "QA":
            raise ValueError(f"historical source_id={source_id!r} is not a QA source")
        context, query = build_context_query(source)
        csv_row = csv_rows[source_id]
        _require_prompt_contains(csv_row.prompt, context, label="context", source_id=source_id)
        _require_prompt_contains(csv_row.prompt, query, label="query", source_id=source_id)
        instances.append(
            Instance(
                response_id=csv_row.response_id,
                source_id=source_id,
                task="QA",
                gen_model="meta-llama/Llama-3.1-8B-Instruct",
                split=str(record["split"]),
                context=context,
                query=query,
                response=csv_row.response,
                y=csv_row.y,
                gt_span_types=[],
                quality="controlled-llama31-gpt4o-annotation",
                prompt=csv_row.prompt,
            )
        )
    instances.sort(key=lambda inst: (inst.split, inst.source_id, inst.response_id))
    provenance = {
        "protocol": LLAMA31_MANIFEST_PROTOCOL,
        "ragtruth_commit": DEFAULT_RAGTRUTH_COMMIT,
        "annotation_csv_sha256": file_sha256(csv_path),
        "source_info_jsonl_sha256": file_sha256(Path(data_dir) / "source_info.jsonl"),
        "response_jsonl_sha256": file_sha256(Path(data_dir) / "response.jsonl"),
        "historical_manifest_sha256": file_sha256(historical_manifest),
        "annotation_model_counts": dict(sorted(Counter(
            row.annotation_model for row in csv_rows.values()
        ).items())),
        "label_counts": {str(label): count for label, count in sorted(
            Counter(inst.y for inst in instances).items()
        )},
        "source_counts": dict(sorted(Counter(inst.split for inst in instances).items())),
    }
    return instances, provenance


def llama31_manifest_dict(instances: Iterable[Instance], provenance: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(instances, key=lambda inst: (inst.split, inst.source_id, inst.response_id))
    if len(rows) != 750:
        raise ValueError(f"controlled Llama manifest must contain 750 rows, got {len(rows)}")
    return {
        "protocol": LLAMA31_MANIFEST_PROTOCOL,
        "task": "QA",
        "generator_model": "meta-llama/Llama-3.1-8B-Instruct",
        "provenance": provenance,
        "quarantine": {
            "source_id": QUARANTINED_SOURCE_ID,
            "reason": "historical deterministic source-level quarantine",
        },
        "records": [
            {
                "source_id": inst.source_id,
                "response_id": inst.response_id,
                "split": inst.split,
                "y": int(inst.y),
                "gen_model": inst.gen_model,
                "context_sha256": normalized_text_sha256(inst.context),
                "query_sha256": normalized_text_sha256(inst.query),
                "answer_sha256": normalized_text_sha256(inst.response),
            }
            for inst in rows
        ],
    }


def write_llama31_manifest(
    path: str | Path, instances: Iterable[Instance], provenance: dict[str, Any]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = llama31_manifest_dict(instances, provenance)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_llama31_manifest_instances_with_historical_manifest(
    path: str | Path,
    csv_path: str | Path,
    data_dir: str | Path,
    historical_manifest: str | Path,
) -> list[Instance]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != LLAMA31_MANIFEST_PROTOCOL or payload.get("task") != "QA":
        raise ValueError(f"unsupported controlled Llama manifest: {path}")
    instances, provenance = build_llama31_instances(csv_path, data_dir, historical_manifest)
    expected = llama31_manifest_dict(instances, provenance)
    if payload != expected:
        raise ValueError("controlled Llama manifest records or provenance differ from immutable inputs")
    return instances


def graph_sha256(graph: Graph) -> str:
    return canonical_json_sha256(graph.to_dict())


def _validated_graph_payload(value: Any, *, label: str) -> Graph:
    if not isinstance(value, dict) or set(value) != {"entities", "relations"}:
        raise ValueError(f"{label} graph has an invalid schema")
    entities = value.get("entities")
    relations = value.get("relations")
    if not isinstance(entities, list) or not all(isinstance(entity, str) for entity in entities):
        raise ValueError(f"{label} graph entities must be a string list")
    if not isinstance(relations, list) or not all(
        isinstance(relation, list)
        and len(relation) == 3
        and all(isinstance(part, str) for part in relation)
        for relation in relations
    ):
        raise ValueError(f"{label} graph relations must be a list of string triples")
    graph = Graph.from_dict(value)
    if graph.to_dict() != {
        "entities": sorted(entities),
        "relations": sorted(relations),
    }:
        raise ValueError(f"{label} graph is not canonical")
    return graph


def _historical_cache_index(cache_root: str | Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """Index a complete cache export without importing it as a cache namespace."""
    root = Path(cache_root)
    if not root.is_dir():
        raise ValueError("historical cache root is not a directory")
    index: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in root.rglob("*.json"):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(envelope, dict) or set(envelope) != {
            "protocol", "cache_key", "graph", "graph_sha256"
        }:
            continue
        key = envelope.get("cache_key")
        if isinstance(key, str):
            index.setdefault(key, []).append((path, envelope))
    return index


def _load_historical_cache_graph(
    cache_index: dict[str, list[tuple[Path, dict[str, Any]]]],
    cache_record: dict[str, Any],
    graph_summary: dict[str, Any],
    *,
    label: str,
) -> Graph:
    cache_file = str(cache_record.get("cache_file") or "")
    cache_key = str(cache_record.get("cache_key") or "")
    if not cache_file or not cache_key or Path(cache_file).name != cache_file:
        raise ValueError(f"historical {label} cache record is malformed")
    candidates = [
        (path, envelope)
        for path, envelope in cache_index.get(cache_key, [])
        if path.name == cache_file
    ]
    if not candidates:
        raise ValueError(f"historical {label} graph cache is missing or corrupt")
    if len(candidates) > 1:
        digests = {canonical_json_sha256(envelope) for _, envelope in candidates}
        if len(digests) != 1:
            raise ValueError(f"historical {label} graph cache key is ambiguous")
    _, envelope = candidates[0]
    if not isinstance(envelope, dict) or set(envelope) != {
        "protocol", "cache_key", "graph", "graph_sha256"
    }:
        raise ValueError(f"historical {label} graph cache envelope is invalid")
    if envelope.get("protocol") != "hallu-kg-cache-v2" or envelope.get("cache_key") != cache_key:
        raise ValueError(f"historical {label} graph cache identity mismatch")
    graph = _validated_graph_payload(envelope.get("graph"), label=f"historical {label}")
    if envelope.get("graph_sha256") != graph_sha256(graph):
        raise ValueError(f"historical {label} graph cache hash mismatch")
    if graph_summary.get("sha256") != graph_sha256(graph):
        raise ValueError(f"historical {label} graph does not match the verified extraction summary")
    return graph


def build_frozen_reference_artifact_from_historical_cache(
    *,
    instances: Iterable[Instance],
    historical_extraction_summary: str | Path,
    historical_run_metadata: str | Path,
    historical_runtime_identity: str | Path,
    historical_cache_root: str | Path,
    historical_cache_export: str | Path,
    historical_cache_export_sha256_file: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn verified historical C/Q cache envelopes into a standalone artifact.

    The import is intentionally one-way: it validates historical cache keys to
    read their graph payloads, then writes no historical key into the frozen
    artifact.  Therefore the new evaluation cannot accidentally read through
    the old cache namespace.
    """
    summary_path = Path(historical_extraction_summary)
    metadata_path = Path(historical_run_metadata)
    runtime_path = Path(historical_runtime_identity)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("historical reference metadata is missing or corrupt") from exc
    if summary.get("protocol") != "hallu-extraction-summary-v2":
        raise ValueError("historical extraction summary protocol is unsupported")
    if summary.get("status") != "ready_with_explicit_exclusions":
        raise ValueError("historical extraction summary is not a verified quarantined completion")
    if summary.get("excluded_source_ids") != [QUARANTINED_SOURCE_ID]:
        raise ValueError("historical extraction summary quarantine differs from source 12448")
    if metadata.get("gateway_manifest_sha256") != HISTORICAL_GATEWAY_MANIFEST_SHA256:
        raise ValueError("historical run metadata gateway identity is not the verified R12 identity")
    if runtime.get("gateway_manifest_sha256") != HISTORICAL_GATEWAY_MANIFEST_SHA256:
        raise ValueError("historical runtime gateway identity is not the verified R12 identity")
    if runtime.get("runtime_fingerprint") != HISTORICAL_LLM_RUNTIME_FINGERPRINT:
        raise ValueError("historical runtime fingerprint is not the verified R12 fingerprint")
    sha_line = Path(historical_cache_export_sha256_file).read_text(encoding="utf-8").strip()
    expected_export_sha = sha_line.split(maxsplit=1)[0] if sha_line else ""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_export_sha):
        raise ValueError("historical cache-export SHA-256 sidecar is malformed")
    actual_export_sha = file_sha256(historical_cache_export)
    if actual_export_sha != expected_export_sha:
        raise ValueError("historical cache-export archive SHA-256 does not match its sidecar")

    rows = list(instances)
    by_source = {inst.source_id: inst for inst in rows}
    expected_sources = set(by_source) - {QUARANTINED_SOURCE_ID}
    graph_records = summary.get("graph_records")
    if not isinstance(graph_records, list):
        raise ValueError("historical extraction summary graph_records is invalid")
    cache_index = _historical_cache_index(historical_cache_root)
    refs: dict[str, tuple[Graph, Graph]] = {}
    for record in graph_records:
        if not isinstance(record, dict):
            raise ValueError("historical extraction summary graph record is invalid")
        source_id = str(record.get("source_id") or "")
        if source_id in refs:
            raise ValueError(f"historical extraction summary repeats source_id={source_id!r}")
        inst = by_source.get(source_id)
        if inst is None:
            raise ValueError(f"historical extraction summary source_id={source_id!r} is not selected")
        cache = record.get("cache")
        if not isinstance(cache, dict):
            raise ValueError(f"historical extraction cache record missing for source_id={source_id}")
        for kind, text in (("context", inst.context), ("query", inst.query)):
            cache_record = cache.get(kind)
            graph_summary = record.get(kind)
            if not isinstance(cache_record, dict) or not isinstance(graph_summary, dict):
                raise ValueError(f"historical {kind} graph summary missing for source_id={source_id}")
            if cache_record.get("text_sha256") != normalized_text_sha256(text):
                raise ValueError(f"historical {kind} text hash mismatch for source_id={source_id}")
        context_graph = _load_historical_cache_graph(
            cache_index, cache["context"], record["context"], label="context"
        )
        query_graph = _load_historical_cache_graph(
            cache_index, cache["query"], record["query"], label="query"
        )
        refs[source_id] = (context_graph, query_graph)
    if set(refs) != expected_sources:
        missing = sorted(expected_sources - set(refs))
        extra = sorted(set(refs) - expected_sources)
        raise ValueError(
            "historical reference coverage is not exactly the 749 non-quarantined sources; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    historical_provenance = {
        "historical_run_source_commit": str(metadata.get("source_commit") or ""),
        "gateway_manifest_sha256": HISTORICAL_GATEWAY_MANIFEST_SHA256,
        "llm_runtime_fingerprint": HISTORICAL_LLM_RUNTIME_FINGERPRINT,
        "extraction_summary_sha256": file_sha256(summary_path),
        "run_metadata_sha256": file_sha256(metadata_path),
        "runtime_identity_sha256": file_sha256(runtime_path),
        "cache_export_sha256": actual_export_sha,
        "cache_export_sidecar_sha256": file_sha256(historical_cache_export_sha256_file),
        "cache_export_graph_file_count": sum(len(paths) for paths in cache_index.values()),
        "source_count": len(refs),
    }
    artifact = frozen_reference_artifact_dict(
        rows, refs, historical_provenance=historical_provenance
    )
    return artifact, historical_provenance


def frozen_reference_artifact_dict(
    instances: Iterable[Instance],
    references: dict[str, tuple[Graph, Graph]],
    *,
    historical_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Create a versioned C/Q graph artifact with no cache keys or answer graphs."""
    rows = list(instances)
    expected_sources = {inst.source_id for inst in rows} - {QUARANTINED_SOURCE_ID}
    if set(references) != expected_sources:
        missing = sorted(expected_sources - set(references))
        extra = sorted(set(references) - expected_sources)
        raise ValueError(f"frozen reference coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    by_source = {inst.source_id: inst for inst in rows}
    records = []
    for source_id in sorted(expected_sources):
        context_graph, query_graph = references[source_id]
        inst = by_source[source_id]
        records.append(
            {
                "source_id": source_id,
                "context_sha256": normalized_text_sha256(inst.context),
                "query_sha256": normalized_text_sha256(inst.query),
                "context_graph": context_graph.to_dict(),
                "context_graph_sha256": graph_sha256(context_graph),
                "query_graph": query_graph.to_dict(),
                "query_graph_sha256": graph_sha256(query_graph),
            }
        )
    return {
        "protocol": FROZEN_REFERENCE_PROTOCOL,
        "historical_provenance": historical_provenance,
        "quarantined_source_id": QUARANTINED_SOURCE_ID,
        "records": records,
    }


def write_frozen_reference_artifact(
    path: str | Path,
    instances: Iterable[Instance],
    references: dict[str, tuple[Graph, Graph]],
    *,
    historical_provenance: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frozen_reference_artifact_dict(
        instances, references, historical_provenance=historical_provenance
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_frozen_reference_graphs(
    path: str | Path,
    instances: Iterable[Instance],
    *,
    excluded_source_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[Graph, Graph]], dict[str, Any]]:
    """Load a frozen C/Q artifact and reject every identity or graph mismatch."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != FROZEN_REFERENCE_PROTOCOL:
        raise ValueError(f"unsupported frozen reference artifact: {path}")
    historical = payload.get("historical_provenance")
    if not isinstance(historical, dict):
        raise ValueError("frozen reference artifact has no historical provenance")
    if historical.get("gateway_manifest_sha256") != HISTORICAL_GATEWAY_MANIFEST_SHA256:
        raise ValueError("frozen reference artifact historical gateway identity is not the verified R12 identity")
    if historical.get("llm_runtime_fingerprint") != HISTORICAL_LLM_RUNTIME_FINGERPRINT:
        raise ValueError("frozen reference artifact historical runtime identity is not the verified R12 identity")
    excluded = {str(value) for value in (excluded_source_ids or set())}
    if payload.get("quarantined_source_id") != QUARANTINED_SOURCE_ID:
        raise ValueError("frozen reference artifact quarantine identity changed")
    if QUARANTINED_SOURCE_ID not in excluded:
        raise ValueError("frozen reference artifact requires source 12448 to remain quarantined")
    source_instances = {inst.source_id: inst for inst in instances}
    expected_sources = set(source_instances) - excluded
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("frozen reference artifact records must be a list")
    references: dict[str, tuple[Graph, Graph]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("frozen reference artifact record is not an object")
        source_id = str(record.get("source_id") or "")
        if source_id in references:
            raise ValueError(f"frozen reference artifact duplicates source_id={source_id!r}")
        inst = source_instances.get(source_id)
        if inst is None:
            raise ValueError(f"frozen reference artifact source_id={source_id!r} is outside the manifest")
        if normalized_text_sha256(inst.context) != record.get("context_sha256"):
            raise ValueError(f"frozen reference artifact context hash mismatch for source_id={source_id}")
        if normalized_text_sha256(inst.query) != record.get("query_sha256"):
            raise ValueError(f"frozen reference artifact query hash mismatch for source_id={source_id}")
        try:
            context_graph = Graph.from_dict(record["context_graph"])
            query_graph = Graph.from_dict(record["query_graph"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"frozen reference artifact graph is malformed for source_id={source_id}") from exc
        if graph_sha256(context_graph) != record.get("context_graph_sha256"):
            raise ValueError(f"frozen reference artifact context graph hash mismatch for source_id={source_id}")
        if graph_sha256(query_graph) != record.get("query_graph_sha256"):
            raise ValueError(f"frozen reference artifact query graph hash mismatch for source_id={source_id}")
        references[source_id] = (context_graph, query_graph)
    if set(references) != expected_sources:
        missing = sorted(expected_sources - set(references))
        extra = sorted(set(references) - expected_sources)
        raise ValueError(f"frozen reference artifact coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    provenance = {
        "reference_origin": "frozen_historical_artifact",
        "artifact_sha256": file_sha256(path),
        "historical_provenance": historical,
        "sources": len(references),
    }
    return references, provenance
