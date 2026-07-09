"""RAGTruth loading, (Context, Query, Answer) construction, and the Data2txt serializer.

Data files (download with ``download_data.py``):
  - source_info.jsonl : one record per source instance.
      fields: source_id, task_type in {QA, Data2txt, Summary}, source, source_info, prompt
      * QA        -> source_info is {"question": str, "passages": str}
      * Data2txt  -> source_info is the business JSON dict
      * Summary   -> source_info is a str (the article)
  - response.jsonl : one record per LLM response.
      fields: id, source_id, model, temperature, labels[], split in {train,test}, quality, response
      * labels[] : {"start","end","text","label_type","meta", opt "due_to_null","implicit_true"}

Response-level ground truth: y = 1 (hallucinated) iff the response has >= 1 annotated span.
RAGTruth strict convention: implicit_true spans COUNT (ablate via exclude_implicit_true).
due_to_null spans always count (corpus standard).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TASK_TYPES = ("QA", "Data2txt", "Summary")


# --------------------------------------------------------------------------------------
# Data2txt JSON -> plain-text serializer (pure, unit-tested)
# --------------------------------------------------------------------------------------
def _fmt_value(v: Any) -> str:
    """Render a scalar the way RAGTruth's null-trap requires.

    booleans -> 'true'/'false'; None -> 'null (unknown)' (NOT 'no'/'false').
    """
    if v is None:
        return "null (unknown)"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _flatten_facts(subject: str, key_path: str, value: Any, lines: list[str]) -> None:
    """Emit one declarative fact per scalar, recursing into nested dicts with dotted keys."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_facts(subject, f"{key_path}.{k}", v, lines)
    elif isinstance(value, (list, tuple)):
        # Non-review lists (rare) are rendered element-wise with an index.
        for i, v in enumerate(value):
            _flatten_facts(subject, f"{key_path}[{i}]", v, lines)
    else:
        lines.append(f"{subject} has {key_path} = {_fmt_value(value)}.")


def serialize_business_json(info: dict[str, Any], include_reviews: bool = True) -> str:
    """Flatten a Yelp business JSON into deterministic declarative sentences.

    - Business name is the subject of every structured fact.
    - Top-level scalars: ``NAME has <field> = <value>.``
    - Nested attributes/hours: dotted key path, e.g.
      ``Subway has attributes.BusinessParking.garage = false.``
    - ``review_info`` is handled specially: its ``review_text`` values are appended
      verbatim after the structured lines (raw text preserved for KGGen).

    Determinism: dict insertion order is preserved (JSON preserves key order).
    """
    subject = str(info.get("name") or "The business").strip() or "The business"
    lines: list[str] = []
    reviews: list[str] = []

    for key, value in info.items():
        if key == "name":
            lines.append(f"The business is named {subject}.")
            continue
        if key == "review_info":
            if isinstance(value, list):
                for rev in value:
                    if isinstance(rev, dict) and rev.get("review_text") is not None:
                        reviews.append(str(rev["review_text"]))
            continue
        _flatten_facts(subject, key, value, lines)

    out = "\n".join(lines)
    if include_reviews and reviews:
        out += "\n\nCustomer reviews:\n" + "\n".join(r.strip() for r in reviews)
    return out


# --------------------------------------------------------------------------------------
# (Context, Query, Answer) construction
# --------------------------------------------------------------------------------------
@dataclass
class Instance:
    """A single (source, response) pair ready for the pipeline."""

    response_id: str
    source_id: str
    task: str
    gen_model: str
    split: str
    context: str          # C
    query: str | None     # Q (None/'' when the task has no query)
    response: str         # A
    y: int                # response-level ground-truth label
    gt_span_types: list[str] = field(default_factory=list)
    quality: str = ""
    prompt: str = ""      # original RAGTruth prompt (audit/debug only)


def build_context_query(source_rec: dict[str, Any]) -> tuple[str, str | None]:
    """Map a source_info record to (Context C, Query Q). Q is None when empty."""
    task = source_rec["task_type"]
    info = source_rec["source_info"]

    if task == "QA":
        if not isinstance(info, dict):
            raise ValueError(f"QA source_info must be a dict, got {type(info)}")
        context = str(info.get("passages", "")).strip()
        query = str(info.get("question", "")).strip() or None
        return context, query

    if task == "Data2txt":
        if not isinstance(info, dict):
            raise ValueError(f"Data2txt source_info must be a dict, got {type(info)}")
        return serialize_business_json(info), None  # Q empty -> G_q = empty

    if task == "Summary":
        # source_info is the article string.
        context = info if isinstance(info, str) else json.dumps(info, ensure_ascii=False)
        return str(context).strip(), None  # Q empty -> G_q = empty

    raise ValueError(f"Unknown task_type: {task!r}")


def response_label(resp_rec: dict[str, Any], exclude_implicit_true: bool = False) -> tuple[int, list[str]]:
    """Return (y, span_types). y=1 iff >=1 counted hallucination span.

    implicit_true spans count unless exclude_implicit_true is True.
    due_to_null spans always count.
    """
    labels = resp_rec.get("labels") or []
    counted_types: list[str] = []
    for span in labels:
        if exclude_implicit_true and _is_truthy_flag(span.get("implicit_true")):
            continue
        counted_types.append(str(span.get("label_type", "Unknown")))
    y = 1 if counted_types else 0
    return y, counted_types


def _is_truthy_flag(v: Any) -> bool:
    """RAGTruth flags appear as bool True or the strings 'true'/'yes'."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


# --------------------------------------------------------------------------------------
# Loading + joining the two JSONL files
# --------------------------------------------------------------------------------------
def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_sources(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """source_id -> source record."""
    path = Path(data_dir) / "source_info.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first."
        )
    sources: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(path):
        sources[str(rec["source_id"])] = rec
    return sources


def load_instances(
    data_dir: str | Path,
    exclude_implicit_true: bool = False,
    splits: Iterable[str] | None = None,
) -> list[Instance]:
    """Join responses with their source and construct Instances.

    ``splits`` optionally filters to e.g. {"train"} or {"test"}.
    """
    sources = load_sources(data_dir)
    resp_path = Path(data_dir) / "response.jsonl"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"{resp_path} not found. Run `python download_data.py` first."
        )

    want = set(splits) if splits is not None else None
    instances: list[Instance] = []
    for rec in _read_jsonl(resp_path):
        split = rec.get("split", "")
        if want is not None and split not in want:
            continue
        source_id = str(rec["source_id"])
        src = sources.get(source_id)
        if src is None:  # orphan response with no source -- skip defensively
            continue
        context, query = build_context_query(src)
        y, span_types = response_label(rec, exclude_implicit_true=exclude_implicit_true)
        instances.append(
            Instance(
                response_id=str(rec["id"]),
                source_id=source_id,
                task=src["task_type"],
                gen_model=str(rec.get("model", "unknown")),
                split=split,
                context=context,
                query=query,
                response=str(rec.get("response", "")),
                y=y,
                gt_span_types=span_types,
                quality=str(rec.get("quality", "")),
                prompt=str(src.get("prompt", "")),
            )
        )
    return instances


def unique_sources(instances: list[Instance]) -> dict[str, Instance]:
    """One representative Instance per source_id (its C and Q are shared across responses)."""
    out: dict[str, Instance] = {}
    for inst in instances:
        out.setdefault(inst.source_id, inst)
    return out
