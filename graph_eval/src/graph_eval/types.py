"""Public data contract shared with the experiment framework.

The detector never receives gold labels or gold spans.  Empty-graph and failure
states are explicit and are *never* masked by a numeric ``raw_score`` (see the
invariant enforced in :class:`DetectionResult`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]

# Detector-level status.  ``ok`` is the only status allowed to carry a score.
STATUS_OK = "ok"
STATUS_EMPTY_GRAPH = "empty_graph"
STATUS_FAILED = "failed"
VALID_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY_GRAPH, STATUS_FAILED})

# Per-triple parse status.  Duplicates are flagged, never silently dropped.
PARSE_OK = "ok"
PARSE_INVALID = "invalid"       # not exactly three non-empty strings
PARSE_DUPLICATE = "duplicate"   # canonical duplicate of an earlier valid triple


@dataclass(frozen=True)
class Triple:
    """A single extracted triple with provenance to the raw answer.

    ``raw_*`` are verbatim from the extractor and are never overwritten;
    ``normalized_*`` are explanation-only and do not affect the score.
    """

    triple_id: str
    raw_subject: str
    raw_relation: str
    raw_object: str
    parse_status: str = PARSE_OK
    duplicate_of: str | None = None
    origin_field: str = "response"
    normalized_subject: str | None = None
    normalized_relation: str | None = None
    normalized_object: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.parse_status == PARSE_OK

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.raw_subject, self.raw_relation, self.raw_object)


@dataclass(frozen=True)
class DetectionInput:
    """The only thing a detector may see for one RAGTruth instance.

    ``query`` is preserved for provenance/audit but faithful GraphEval does not
    use it as evidence.  No gold fields are present by construction.
    """

    response_id: str
    source_id: str
    context: str
    response: str
    query: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    """A method's raw prediction.  ``raw_score``: higher = more likely hallucination."""

    response_id: str
    source_id: str
    method: str
    raw_score: float | None
    components: Mapping[str, JSONValue]
    flagged_unit_ids: tuple[str, ...]
    status: str
    failure: Mapping[str, JSONValue] | None
    usage: Mapping[str, JSONValue]
    artifact_refs: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.status != STATUS_OK and self.raw_score is not None:
            # empty_graph / failed must remain a distinct state, not a 0/1 score.
            raise ValueError(
                f"status={self.status!r} must have raw_score=None, got {self.raw_score!r}"
            )
        if self.status == STATUS_FAILED and not self.failure:
            raise ValueError("status='failed' requires a non-empty failure mapping")
