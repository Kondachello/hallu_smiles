"""Versioned GraphEval extraction prompt (Appendix A semantics).

Two output modes share one system instruction:
  * ``paper_prompt``     -> a bare JSON array of [s, r, o] string triples;
  * ``structured_json``  -> the gateway enforces a strict json_schema (an object
    with a ``triples`` array); the pipeline still re-validates locally.

Changing any string here must bump ``PROMPT_VERSION`` / ``SCHEMA_VERSION`` so the
extraction cache does not serve stale output. (See DEVIATIONS.md D3: the paper's
raw list-of-triples is canonicalized to JSON.)
"""
from __future__ import annotations

PROMPT_VERSION = "grapheval_appendix_a_v1"
SCHEMA_VERSION = "triples_json_v1"

_SYSTEM = (
    "You extract a knowledge graph of factual triples from a single passage of text.\n"
    "Steps:\n"
    "1. Identify entities (people, places, organizations, dates, values, concepts).\n"
    "2. Resolve coreferences so each entity is named consistently.\n"
    "3. Extract relations between entities as (subject, relation, object) triples.\n"
    "4. Keep linked information inside one relation when splitting would change meaning.\n"
    "5. Every triple is exactly three non-empty strings.\n"
    "6. Cover all factual information stated in the passage; do not add outside facts.\n"
    "\n"
    'Return ONLY a JSON array of triples, each a 3-element array of strings:\n'
    '[["subject","relation","object"], ...]\n'
    "No prose, no markdown, no code fences."
)

_REPAIR = (
    "Your previous message was not a valid JSON array of [subject, relation, object] "
    "string triples, or it was truncated. Return ONLY the corrected JSON array."
)

STRUCTURED_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "triples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                }
            },
            "required": ["triples"],
            "additionalProperties": False,
        },
    },
}


def build_messages(response_text: str, output_mode: str) -> list[dict]:
    """System instruction + the answer as the user turn (answer only)."""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": response_text},
    ]


def repair_messages(base_messages: list[dict], prior_raw: str) -> list[dict]:
    return [
        *base_messages,
        {"role": "assistant", "content": prior_raw},
        {"role": "user", "content": _REPAIR},
    ]
