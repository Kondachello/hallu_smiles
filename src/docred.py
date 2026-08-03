"""Deterministic, coverage-aware KG extraction evaluation for DocRED.

The primary metric intentionally measures agreement with DocRED annotations, not
truth in the open world.  Consequently predicted-triple precision is exposed as
``gold_supported_precision`` everywhere in this module.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .extract import Graph, UsageLogger
from .matching import Embedder, _token_boundary_substring, normalize


DOCRED_HF_REPO = "thunlp/docred"
DOCRED_HF_REVISION = "7985b4e0371e6c61a756feb41b7b27becf71c666"
DOCRED_FILES = {
    "train_annotated": "train_annotated.json.gz",
    "dev": "dev.json.gz",
    "rel_info": "rel_info.json.gz",
}
RELATION_THRESHOLD_GRID = (0.65, 0.75, 0.85)


class BudgetExceeded(RuntimeError):
    """Raised before another paid document is scheduled."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a diagnostic/checkpoint atomically without shared temporary names."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False,
        prefix=f".{destination.name}.", suffix=".tmp",
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        return json.load(handle)


@dataclass(frozen=True, order=True)
class Triple:
    head: int
    relation: str
    tail: int

    def to_dict(self) -> dict[str, Any]:
        return {"head": self.head, "relation": self.relation, "tail": self.tail}


@dataclass(frozen=True)
class DocREDDocument:
    split: str
    source_index: int
    document_id: str
    text: str
    entities: tuple[tuple[str, ...], ...]
    gold: frozenset[Triple]

    @property
    def gold_entity_pairs(self) -> frozenset[tuple[int, int]]:
        return frozenset((triple.head, triple.tail) for triple in self.gold)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_index": self.source_index,
            "document_id": self.document_id,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "gold_triples": len(self.gold),
        }


def _document_id(split: str, source_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"docred-v1\0{split}\0{source_index}\0{text}".encode("utf-8")
    ).hexdigest()
    return f"{split}-{source_index}-{digest[:16]}"


def load_docred_documents(data_dir: str | Path, split: str) -> list[DocREDDocument]:
    """Load an annotated DocRED split without exposing texts in manifests/logs."""
    if split not in {"train_annotated", "dev"}:
        raise ValueError("only annotated DocRED train_annotated and dev splits are supported")
    path = Path(data_dir) / DOCRED_FILES[split]
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"DocRED split is not a list: {path}")
    documents: list[DocREDDocument] = []
    for source_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"DocRED record {source_index} is not an object")
        sents = item.get("sents")
        vertices = item.get("vertexSet")
        labels = item.get("labels")
        if not isinstance(sents, list) or not isinstance(vertices, list) or not isinstance(labels, list):
            raise ValueError(f"DocRED record {source_index} lacks sents/vertexSet/labels")
        text = "\n".join(" ".join(str(token) for token in sentence) for sentence in sents).strip()
        if not text:
            raise ValueError(f"DocRED record {source_index} has empty text")
        entities: list[tuple[str, ...]] = []
        for entity_index, mentions in enumerate(vertices):
            if not isinstance(mentions, list):
                raise ValueError(f"DocRED entity {entity_index} is not a mention list")
            aliases = tuple(sorted({str(mention.get("name", "")).strip() for mention in mentions if isinstance(mention, dict) and str(mention.get("name", "")).strip()}))
            if not aliases:
                raise ValueError(f"DocRED entity {entity_index} has no usable aliases")
            entities.append(aliases)
        gold: set[Triple] = set()
        for label in labels:
            if not isinstance(label, dict):
                raise ValueError(f"DocRED label in record {source_index} is not an object")
            try:
                head, tail, relation = int(label["h"]), int(label["t"]), str(label["r"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid DocRED label in record {source_index}") from exc
            if not relation or not (0 <= head < len(entities) and 0 <= tail < len(entities)):
                raise ValueError(f"invalid DocRED triple indices in record {source_index}")
            gold.add(Triple(head, relation, tail))
        documents.append(DocREDDocument(
            split=split,
            source_index=source_index,
            document_id=_document_id(split, source_index, text),
            text=text,
            entities=tuple(entities),
            gold=frozenset(gold),
        ))
    return documents


def load_relation_info(data_dir: str | Path) -> dict[str, str]:
    payload = _read_json(Path(data_dir) / DOCRED_FILES["rel_info"])
    if not isinstance(payload, dict) or not payload:
        raise ValueError("DocRED rel_info is empty or invalid")
    result = {str(key): str(value) for key, value in payload.items() if str(value).strip()}
    if not result:
        raise ValueError("DocRED rel_info has no descriptions")
    return result


def select_documents(documents: Sequence[DocREDDocument], count: int, seed: int) -> list[DocREDDocument]:
    """Select independent of annotations using a stable hash rank."""
    if count < 1 or count > len(documents):
        raise ValueError(f"document count must be in [1, {len(documents)}]")
    ranked = sorted(
        documents,
        key=lambda document: hashlib.sha256(
            f"docred-manifest-v1\0{seed}\0{document.document_id}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:count]


def make_manifest(
    *, train_documents: Sequence[DocREDDocument], dev_documents: Sequence[DocREDDocument],
    train_count: int, dev_count: int, seed: int, data_dir: str | Path,
) -> dict[str, Any]:
    selected_train = select_documents(train_documents, train_count, seed)
    selected_dev = select_documents(dev_documents, dev_count, seed)
    data_root = Path(data_dir)
    file_hashes = {
        name: sha256_file(data_root / filename)
        for name, filename in DOCRED_FILES.items()
    }
    payload: dict[str, Any] = {
        "protocol": "docred-kg-eval-manifest-v1",
        "dataset": {
            "repository": DOCRED_HF_REPO,
            "revision": DOCRED_HF_REVISION,
            "files_sha256": file_hashes,
        },
        "seed": int(seed),
        "calibration": {"split": "train_annotated", "count": len(selected_train)},
        "heldout": {"split": "dev", "count": len(selected_dev)},
        "documents": [
            *[document.manifest_record() for document in selected_train],
            *[document.manifest_record() for document in selected_dev],
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def documents_from_manifest(
    manifest: Mapping[str, Any], train_documents: Sequence[DocREDDocument], dev_documents: Sequence[DocREDDocument],
) -> tuple[list[DocREDDocument], list[DocREDDocument]]:
    if manifest.get("protocol") != "docred-kg-eval-manifest-v1":
        raise ValueError("DocRED manifest has an incompatible protocol")
    if manifest.get("seed") != 42:
        raise ValueError("DocRED manifest does not use the fixed seed 42")
    calibration = manifest.get("calibration")
    heldout = manifest.get("heldout")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("split") != "train_annotated"
        or calibration.get("count") != 50
        or not isinstance(heldout, Mapping)
        or heldout.get("split") != "dev"
        or heldout.get("count") != 200
    ):
        raise ValueError("DocRED manifest is not the fixed 50/200 protocol")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("DocRED manifest has no dataset provenance")
    expected_hashes = dataset.get("files_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("DocRED manifest has no dataset file checksums")
    records = manifest.get("documents")
    if not isinstance(records, list):
        raise ValueError("DocRED manifest has no document records")
    index = {
        (document.split, document.source_index): document
        for document in [*train_documents, *dev_documents]
    }
    selected: dict[str, list[DocREDDocument]] = {"train_annotated": [], "dev": []}
    seen: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("DocRED manifest record is invalid")
        key = (str(record.get("split", "")), int(record.get("source_index", -1)))
        document = index.get(key)
        if document is None or key in seen:
            raise ValueError("DocRED manifest does not match local dataset")
        if document.document_id != record.get("document_id") or document.manifest_record()["text_sha256"] != record.get("text_sha256"):
            raise ValueError("DocRED manifest document fingerprint mismatch")
        seen.add(key)
        selected[key[0]].append(document)
    if len(selected["train_annotated"]) != 50 or len(selected["dev"]) != 200:
        raise ValueError("DocRED manifest must include exactly 50 calibration and 200 held-out documents")
    return selected["train_annotated"], selected["dev"]


@dataclass(frozen=True)
class EndpointResolution:
    status: str  # matched | unmatched | ambiguous
    entity_id: int | None = None
    method: str | None = None


class EntityResolver:
    """Conservative alias-only resolver from KGGen surfaces to DocRED entities."""

    def __init__(self, document: DocREDDocument):
        self.aliases: dict[int, set[str]] = {
            entity_id: {normalize(alias) for alias in names if normalize(alias)}
            for entity_id, names in enumerate(document.entities)
        }

    def resolve(self, value: str) -> EndpointResolution:
        query = normalize(value)
        if not query:
            return EndpointResolution("unmatched")
        exact = [entity_id for entity_id, aliases in self.aliases.items() if query in aliases]
        if len(exact) == 1:
            return EndpointResolution("matched", exact[0], "exact_alias")
        if len(exact) > 1:
            return EndpointResolution("ambiguous")
        if len(query) < 3:
            return EndpointResolution("unmatched")
        substring = [
            entity_id
            for entity_id, aliases in self.aliases.items()
            if any(
                _token_boundary_substring(query, alias) or _token_boundary_substring(alias, query)
                for alias in aliases
                if len(alias) >= 3
            )
        ]
        if len(substring) == 1:
            return EndpointResolution("matched", substring[0], "alias_substring")
        return EndpointResolution("ambiguous" if substring else "unmatched")


@dataclass(frozen=True)
class RelationResolution:
    status: str  # matched | unmatched | ambiguous
    relation_id: str | None = None
    score: float | None = None
    method: str | None = None


class RelationAligner:
    """Frozen local matcher from a natural-language predicate to DocRED labels."""

    def __init__(self, relation_info: Mapping[str, str], embedder: Embedder | None):
        self.relation_info = {str(key): str(value) for key, value in relation_info.items()}
        self.embedder = embedder
        self.relation_ids = sorted(self.relation_info)
        self.descriptions = [normalize(self.relation_info[key]) for key in self.relation_ids]
        self._embeddings = embedder.encode(self.descriptions) if embedder is not None else None

    def resolve(self, predicate: str, threshold: float) -> RelationResolution:
        value = normalize(predicate)
        if not value:
            return RelationResolution("unmatched")
        exact = [
            relation_id for relation_id, description in zip(self.relation_ids, self.descriptions)
            if value == relation_id.lower() or value == description
        ]
        if len(exact) == 1:
            return RelationResolution("matched", exact[0], 1.0, "exact")
        if len(exact) > 1:
            return RelationResolution("ambiguous")
        if self.embedder is None or self._embeddings is None:
            return RelationResolution("unmatched")
        vector = self.embedder.encode([value])[0]
        scores = self._embeddings @ vector
        order = np.argsort(-scores)
        best = int(order[0])
        best_score = float(scores[best])
        if len(order) > 1 and math.isclose(best_score, float(scores[int(order[1])]), abs_tol=1e-9):
            return RelationResolution("ambiguous", score=best_score, method="embedding_tie")
        if best_score < float(threshold):
            return RelationResolution("unmatched", score=best_score, method="embedding")
        return RelationResolution("matched", self.relation_ids[best], best_score, "embedding")


@dataclass
class AlignmentResult:
    triples: set[Triple] = field(default_factory=set)
    entity_pairs: set[tuple[int, int]] = field(default_factory=set)
    diagnostics: dict[str, int] = field(default_factory=dict)


def align_graph(
    document: DocREDDocument, graph: Graph, aligner: RelationAligner, threshold: float,
) -> AlignmentResult:
    resolver = EntityResolver(document)
    diagnostics = {
        "raw_predicted_triples": 0, "entity_aligned_predictions": 0,
        "relation_aligned_predictions": 0, "entity_unmatched": 0,
        "entity_ambiguous": 0, "relation_unmatched": 0,
        "relation_ambiguous": 0, "duplicate_predictions": 0,
    }
    result = AlignmentResult(diagnostics=diagnostics)
    for subject, predicate, object_ in sorted(graph.relations):
        diagnostics["raw_predicted_triples"] += 1
        head = resolver.resolve(subject)
        tail = resolver.resolve(object_)
        if head.status == "ambiguous" or tail.status == "ambiguous":
            diagnostics["entity_ambiguous"] += 1
            continue
        if head.status != "matched" or tail.status != "matched":
            diagnostics["entity_unmatched"] += 1
            continue
        assert head.entity_id is not None and tail.entity_id is not None
        diagnostics["entity_aligned_predictions"] += 1
        result.entity_pairs.add((head.entity_id, tail.entity_id))
        relation = aligner.resolve(predicate, threshold)
        if relation.status == "ambiguous":
            diagnostics["relation_ambiguous"] += 1
            continue
        if relation.status != "matched":
            diagnostics["relation_unmatched"] += 1
            continue
        assert relation.relation_id is not None
        triple = Triple(head.entity_id, relation.relation_id, tail.entity_id)
        if triple in result.triples:
            diagnostics["duplicate_predictions"] += 1
            continue
        result.triples.add(triple)
        diagnostics["relation_aligned_predictions"] += 1
    return result


def _ratio(numerator: int, denominator: int, *, zero_when_empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(zero_when_empty)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _ci(values: Sequence[float]) -> list[float]:
    return [round(float(np.quantile(values, 0.025)), 6), round(float(np.quantile(values, 0.975)), 6)]


def aggregate_document_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(scores)
    gold = sum(int(item["gold_triples"]) for item in values)
    predicted = sum(int(item["predicted_triples"]) for item in values)
    matched = sum(int(item["matched_triples"]) for item in values)
    gold_pairs = sum(int(item["gold_entity_pairs"]) for item in values)
    predicted_pairs = sum(int(item["predicted_entity_pairs"]) for item in values)
    matched_pairs = sum(int(item["matched_entity_pairs"]) for item in values)
    recall = _ratio(matched, gold)
    precision = _ratio(matched, predicted)
    pair_recall = _ratio(matched_pairs, gold_pairs)
    pair_precision = _ratio(matched_pairs, predicted_pairs)
    return {
        "documents": len(values),
        "gold_triples": gold,
        "predicted_triples": predicted,
        "matched_triples": matched,
        "triple_recall": recall,
        "gold_supported_precision": precision,
        "triple_f1": _f1(precision, recall),
        "gold_entity_pairs": gold_pairs,
        "predicted_entity_pairs": predicted_pairs,
        "matched_entity_pairs": matched_pairs,
        "entity_pair_recall": pair_recall,
        "entity_pair_gold_supported_precision": pair_precision,
        "entity_pair_f1": _f1(pair_precision, pair_recall),
    }


def score_document(document: DocREDDocument, graph: Graph | None, aligner: RelationAligner, threshold: float, *, failure: str | None = None) -> dict[str, Any]:
    aligned = align_graph(document, graph or Graph.empty(), aligner, threshold)
    matched = len(aligned.triples & document.gold)
    matched_pairs = len(aligned.entity_pairs & document.gold_entity_pairs)
    return {
        "document_id": document.document_id,
        "split": document.split,
        "gold_triples": len(document.gold),
        "predicted_triples": len(aligned.triples),
        "matched_triples": matched,
        "gold_entity_pairs": len(document.gold_entity_pairs),
        "predicted_entity_pairs": len(aligned.entity_pairs),
        "matched_entity_pairs": matched_pairs,
        "extraction_failed": failure is not None,
        "failure_kind": failure,
        "alignment": aligned.diagnostics,
    }


def select_relation_threshold(
    documents: Sequence[DocREDDocument], graphs: Mapping[str, Graph | None],
    failures: Mapping[str, str], aligner: RelationAligner,
    thresholds: Sequence[float] = RELATION_THRESHOLD_GRID,
) -> tuple[float, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        scored = [score_document(document, graphs.get(document.document_id), aligner, float(threshold), failure=failures.get(document.document_id)) for document in documents]
        metrics = aggregate_document_scores(scored)
        candidates.append({"threshold": float(threshold), **metrics})
    # Deterministic conservative tie-break: higher threshold then lower lexical order.
    selected = max(candidates, key=lambda item: (float(item["triple_f1"]), float(item["threshold"])))
    return float(selected["threshold"]), {"candidates": candidates, "selected": selected}


def evaluate_documents(
    documents: Sequence[DocREDDocument], graphs: Mapping[str, Graph | None],
    failures: Mapping[str, str], aligner: RelationAligner, threshold: float,
    *, bootstrap_seed: int = 42, n_bootstrap: int = 1000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = [score_document(document, graphs.get(document.document_id), aligner, threshold, failure=failures.get(document.document_id)) for document in documents]
    summary = aggregate_document_scores(scored)
    summary["extraction_coverage"] = _ratio(
        sum(not bool(item["extraction_failed"]) for item in scored), len(scored)
    )
    summary["extraction_failures"] = sum(bool(item["extraction_failed"]) for item in scored)
    diagnostics: dict[str, int] = {}
    for item in scored:
        for key, value in dict(item["alignment"]).items():
            diagnostics[key] = diagnostics.get(key, 0) + int(value)
    summary["alignment_diagnostics"] = dict(sorted(diagnostics.items()))
    if not scored or n_bootstrap < 1:
        raise ValueError("cannot bootstrap an empty DocRED evaluation")
    rng = Random(bootstrap_seed)
    samples = [
        aggregate_document_scores([scored[rng.randrange(len(scored))] for _ in range(len(scored))])
        for _ in range(n_bootstrap)
    ]
    summary["bootstrap"] = {
        "seed": int(bootstrap_seed), "replicates": int(n_bootstrap),
        "triple_recall_ci95": _ci([float(sample["triple_recall"]) for sample in samples]),
        "gold_supported_precision_ci95": _ci([float(sample["gold_supported_precision"]) for sample in samples]),
        "triple_f1_ci95": _ci([float(sample["triple_f1"]) for sample in samples]),
    }
    return summary, scored


@dataclass(frozen=True)
class PriceSnapshot:
    input_usd_per_million: float = 0.30
    output_usd_per_million: float = 2.50
    usd_per_eur: float = 1.10
    per_live_call_reserve_eur: float = 0.02
    per_document_reserve_eur: float = 0.04


class BudgetGuard:
    """Fail-closed spending estimate when the gateway does not report tokens.

    ``UsageLogger.api_calls`` is document-level: one KG extraction can issue
    raw, chunk, cluster, and retry operations before it records a completed
    document.  A strict live budget therefore reserves every backend operation.
    """

    def __init__(self, max_eur: float, snapshot: PriceSnapshot = PriceSnapshot()):
        if max_eur <= 0:
            raise ValueError("max_eur must be positive")
        self.max_eur = float(max_eur)
        self.snapshot = snapshot
        self._reserved_live_requests = 0
        self._lock = threading.RLock()

    @property
    def reserved_live_requests(self) -> int:
        with self._lock:
            return self._reserved_live_requests

    def estimate_eur(self, usage: Mapping[str, Any]) -> float:
        prompt = max(0, int(usage.get("prompt_tokens", 0)))
        completion = max(0, int(usage.get("completion_tokens", 0)))
        calls = max(0, int(usage.get("api_calls", usage.get("calls", 0))))
        token_usd = (
            prompt * self.snapshot.input_usd_per_million / 1_000_000
            + completion * self.snapshot.output_usd_per_million / 1_000_000
        )
        token_eur = token_usd / self.snapshot.usd_per_eur
        # The document-level fallback preserves compatibility for callers
        # without raw-operation instrumentation.
        reserve_count = max(calls, self.reserved_live_requests)
        reserve_eur = reserve_count * self.snapshot.per_live_call_reserve_eur
        return round(max(token_eur, reserve_eur), 6)

    def remaining_eur(self, usage: Mapping[str, Any]) -> float:
        return round(self.max_eur - self.estimate_eur(usage), 6)

    def assert_can_start_document(self, usage: Mapping[str, Any]) -> None:
        estimated = self.estimate_eur(usage)
        if estimated + self.snapshot.per_document_reserve_eur > self.max_eur:
            raise BudgetExceeded(
                f"budget exhausted before next document: estimated_eur={estimated:.4f} "
                f"reserve_eur={self.snapshot.per_document_reserve_eur:.4f} max_eur={self.max_eur:.4f}"
            )

    def reserve_live_request(self, usage: Mapping[str, Any]) -> None:
        """Reserve one real gateway operation immediately before it is sent.

        Reservations are never released: a failed transport may still have
        reached the provider. This makes the documented EUR limit a hard stop
        across chunks, clustering, protocol retries, and transient retries.
        """
        with self._lock:
            estimated = self.estimate_eur(usage)
            reserve = self.snapshot.per_live_call_reserve_eur
            if estimated + reserve > self.max_eur:
                raise BudgetExceeded(
                    f"budget exhausted before live request: estimated_eur={estimated:.4f} "
                    f"reserve_eur={reserve:.4f} max_eur={self.max_eur:.4f}"
                )
            self._reserved_live_requests += 1

    def assert_can_reserve_documents(
        self, usage: Mapping[str, Any], documents: int, *, requests_per_document: int | None = None,
    ) -> None:
        """Require a conservative reserve before leaving the live smoke stage."""
        if documents < 0:
            raise ValueError("documents must be non-negative")
        if requests_per_document is not None and requests_per_document <= 0:
            raise ValueError("requests_per_document must be positive when provided")
        estimated = self.estimate_eur(usage)
        reserve = (
            int(documents) * self.snapshot.per_document_reserve_eur
            if requests_per_document is None
            else int(documents) * int(requests_per_document) * self.snapshot.per_live_call_reserve_eur
        )
        if estimated + reserve > self.max_eur:
            raise BudgetExceeded(
                f"budget cannot reserve remaining live documents: estimated_eur={estimated:.4f} "
                f"reserve_eur={reserve:.4f} max_eur={self.max_eur:.4f}"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "max_eur": self.max_eur,
            "price_snapshot": asdict(self.snapshot),
            "reserved_live_requests": self.reserved_live_requests,
        }
