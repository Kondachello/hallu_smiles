"""Framework-facing detector contract and gold-leakage guards.

The first vertical slice intentionally reuses the proven shared contract in
``graph_eval.types``.  It is wrapped here so all framework modules import one neutral
module; moving the dataclasses to ``detector_contracts`` later will not change callers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ._compat import ensure_graph_eval_importable

ensure_graph_eval_importable()
from graph_eval.types import (  # noqa: E402
    DetectionInput,
    DetectionResult,
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
)

ALLOWED_METADATA_FIELDS = frozenset(
    {
        "dataset_record_id",
        "task",
        "source_dataset",
        "generator_model",
        "generator_temperature",
        "context_document_ids",
        "context_document_order",
    }
)

_GOLD_KEY_FRAGMENTS = frozenset(
    {
        "label",
        "gold",
        "span",
        "due_to_null",
        "implicit_true",
        "quality",
        "annotator",
        "target",
        "response_label",
        "ground_truth",
    }
)
_SAFE_GOLDISH_KEYS = frozenset({"gold_access_state"})


class GoldLeakageError(ValueError):
    """Raised before a detector can receive an evaluation-only field."""


@runtime_checkable
class DetectorProtocol(Protocol):
    """All method adapters exposed to the generic runner implement this protocol."""

    method_name: str
    variant_name: str

    def predict(self, item: DetectionInput) -> DetectionResult:
        """Return a score whose larger value means higher hallucination risk."""


def _is_gold_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _SAFE_GOLDISH_KEYS:
        return False
    return any(fragment in normalized for fragment in _GOLD_KEY_FRAGMENTS)


def assert_no_gold(value: Any, *, path: str = "record") -> None:
    """Recursively reject evaluation-only keys before inference.

    This rejects keys rather than values so the guard remains safe for future datasets.
    It is intentionally applied to an entire no-gold instance before projection to the
    detector dataclass, preventing accidental additions to metadata.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _is_gold_key(key_text):
                raise GoldLeakageError(f"evaluation-only key at {path}.{key_text}")
            assert_no_gold(child, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_gold(child, path=f"{path}[{index}]")


def make_detection_input(record: Mapping[str, Any]) -> DetectionInput:
    """Project a materialized no-gold record to the detector's minimal input."""
    assert_no_gold(record)
    required = ("response_id", "source_id", "context_raw", "response_raw")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"no-gold instance lacks required fields: {missing}")

    raw_metadata = record.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("instance metadata must be a mapping")
    unknown = set(raw_metadata) - ALLOWED_METADATA_FIELDS
    if unknown:
        raise ValueError(f"metadata contains non-allowlisted fields: {sorted(unknown)}")

    return DetectionInput(
        response_id=str(record["response_id"]),
        source_id=str(record["source_id"]),
        context=str(record["context_raw"]),
        response=str(record["response_raw"]),
        query=(str(record["query_raw"]) if record.get("query_raw") else None),
        metadata=dict(raw_metadata),
    )


def result_record(result: DetectionResult, *, variant: str) -> dict[str, Any]:
    """Serialize the framework-neutral portion of one detector result."""
    return {
        "response_id": result.response_id,
        "source_id": result.source_id,
        "method": result.method,
        "variant": variant,
        "raw_score": result.raw_score,
        "score_direction": "higher_is_more_hallucinated",
        "status": result.status,
        "components": dict(result.components),
        "flagged_unit_ids": list(result.flagged_unit_ids),
        "failure": dict(result.failure) if result.failure else None,
        "usage": dict(result.usage),
        "artifact_refs": dict(result.artifact_refs),
    }


__all__ = [
    "ALLOWED_METADATA_FIELDS",
    "DetectionInput",
    "DetectionResult",
    "DetectorProtocol",
    "GoldLeakageError",
    "STATUS_EMPTY_GRAPH",
    "STATUS_FAILED",
    "STATUS_OK",
    "assert_no_gold",
    "make_detection_input",
    "result_record",
]
