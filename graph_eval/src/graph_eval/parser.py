"""Parse extractor output into ordered triples, keeping raw output and status.

Accepts a JSON array of ``[subject, relation, object]`` triples, or an object
carrying such a list under ``triples``/``triplets``/``result``.  Malformed output
is reported (never guessed), each item that is not exactly three non-empty
strings is kept with ``PARSE_INVALID``, and canonical duplicates are flagged
with ``duplicate_of`` rather than dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .types import PARSE_DUPLICATE, PARSE_INVALID, PARSE_OK, Triple

PARSER_VERSION = "v1"

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_MALFORMED = "malformed"


@dataclass(frozen=True)
class ParseOutcome:
    triples: tuple[Triple, ...]
    raw_output: str
    status: str  # ok | empty | malformed
    error: str | None = None
    invalid_count: int = 0

    @property
    def valid_triples(self) -> tuple[Triple, ...]:
        return tuple(t for t in self.triples if t.parse_status == PARSE_OK)


def _canon(text: str) -> str:
    return " ".join(text.split()).lower()


def _coerce_list(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("triples", "triplets", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise ValueError("object has no triple list under triples/triplets/result")
    raise ValueError("payload is neither a list nor an object with a triple list")


def parse_triples(raw_output: object) -> ParseOutcome:
    raw = raw_output if isinstance(raw_output, str) else str(raw_output)
    try:
        items = _coerce_list(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        return ParseOutcome((), raw, STATUS_MALFORMED, f"parse error: {exc}")

    if not items:
        return ParseOutcome((), raw, STATUS_EMPTY)

    triples: list[Triple] = []
    seen: dict[tuple[str, str, str], str] = {}
    invalid = 0
    for index, item in enumerate(items):
        tid = f"t_{index + 1}"
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and all(isinstance(x, str) and x.strip() for x in item)
        ):
            subject, relation, obj = item
            key = (_canon(subject), _canon(relation), _canon(obj))
            if key in seen:
                triples.append(
                    Triple(tid, subject, relation, obj,
                           parse_status=PARSE_DUPLICATE, duplicate_of=seen[key])
                )
            else:
                seen[key] = tid
                triples.append(Triple(tid, subject, relation, obj, parse_status=PARSE_OK))
        else:
            invalid += 1
            parts = list(item) if isinstance(item, (list, tuple)) else []
            subject = str(parts[0]) if len(parts) > 0 else ""
            relation = str(parts[1]) if len(parts) > 1 else ""
            obj = str(parts[2]) if len(parts) > 2 else ""
            triples.append(Triple(tid, subject, relation, obj, parse_status=PARSE_INVALID))

    return ParseOutcome(tuple(triples), raw, STATUS_OK, None, invalid)
