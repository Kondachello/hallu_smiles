#!/usr/bin/env python3
"""Create a scalar-only paired graph reference from the verified R12 archive.

The source archive contains historical graph outputs and is read only.  This
tool emits only response/source IDs, split, binary label, three scalar risks,
frozen thresholds, and archive checksum.  It deliberately never copies graph
payloads, contexts, answers, prompts, audit records, or logs into the new
candidate-agreement namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_agreement import GRAPH_REFERENCE_PROTOCOL
from src.metrics import ScoreResult


HISTORICAL_750_MANIFEST_SHA256 = "19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86"
METHOD_DIRS = {"strict": "strict", "support": "support", "support_critical": "support-critical"}


class ArchiveError(RuntimeError):
    """The selected artifact is not the verified R12 output layout."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArchiveReader:
    def __init__(self, archive: Path):
        self.archive = archive
        self.tar: tarfile.TarFile | None = None
        self.root: Path | None = None
        if archive.is_file():
            self.tar = tarfile.open(archive, "r:*")
            self.names = [member.name for member in self.tar.getmembers() if member.isfile()]
            self.checksum = _sha256_file(archive)
        elif archive.is_dir():
            self.root = archive
            self.names = [path.relative_to(archive).as_posix() for path in archive.rglob("*") if path.is_file()]
            digest = hashlib.sha256()
            for name in sorted(self.names):
                digest.update(name.encode("utf-8"))
                digest.update((archive / name).read_bytes())
            self.checksum = digest.hexdigest()
        else:
            raise ArchiveError("R12 archive path does not exist")

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()

    def read_bytes(self, name: str) -> bytes:
        if self.tar is not None:
            member = self.tar.getmember(name)
            handle = self.tar.extractfile(member)
            if handle is None:
                raise ArchiveError("archive member cannot be read")
            return handle.read()
        assert self.root is not None
        return (self.root / name).read_bytes()

    def matching(self, predicate) -> list[str]:
        return sorted(name for name in self.names if predicate(PurePosixPath(name)))


def _single(reader: _ArchiveReader, predicate, description: str) -> str:
    matches = reader.matching(predicate)
    if len(matches) != 1:
        raise ArchiveError(f"R12 archive needs exactly one {description}, found {len(matches)}")
    return matches[0]


def _json(reader: _ArchiveReader, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(reader.read_bytes(name))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"R12 archive contains invalid JSON: {name}") from exc
    if not isinstance(payload, dict):
        raise ArchiveError(f"R12 archive JSON is not an object: {name}")
    return payload


def _jsonl(reader: _ArchiveReader, name: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in reader.read_bytes(name).decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"R12 archive contains invalid scored JSONL: {name}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ArchiveError("R12 scored JSONL contains a non-object record")
    return rows


def _locate_manifest(reader: _ArchiveReader) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for name in reader.matching(lambda path: path.suffix == ".json" and "manifest" in path.name.lower()):
        try:
            payload = _json(reader, name)
        except ArchiveError:
            continue
        if payload.get("manifest_sha256") == HISTORICAL_750_MANIFEST_SHA256 and isinstance(payload.get("records"), list):
            candidates.append(payload)
    if len(candidates) != 1:
        raise ArchiveError("R12 archive does not contain one validated historical 750-row manifest")
    return candidates[0]


def _method_rows(reader: _ArchiveReader, method: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    directory = METHOD_DIRS[method]
    scored_name = _single(
        reader,
        lambda path: path.name == "scored.jsonl" and directory in path.parts and "cache-replay" not in path.parts,
        f"live {directory}/scored.jsonl",
    )
    tuning_name = _single(
        reader,
        lambda path: path.name == "tuning.json" and directory in path.parts and "cache-replay" not in path.parts,
        f"live {directory}/tuning.json",
    )
    rows = _jsonl(reader, scored_name)
    by_id = {str(row.get("response_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ArchiveError(f"R12 {method} scored rows have duplicate response IDs")
    return by_id, _json(reader, tuning_name)


def _method_risk(method: str, record: dict[str, Any], tuning: dict[str, Any]) -> float | None:
    score_payload = record.get("score")
    if not isinstance(score_payload, dict):
        raise ArchiveError(f"R12 {method} scored record lacks score payload")
    result = ScoreResult.from_dict(score_payload)
    alpha = float(tuning["alpha"])
    if method == "strict":
        return result.h_for_mode(alpha, "strict")
    if method == "support":
        return result.h_for_mode(alpha, "support")
    return result.critical_h(
        alpha, float(tuning["beta"]), int(tuning["top_k"]), float(tuning["unknown_risk"])
    )


def _is_graph_unscorable(record: dict[str, Any]) -> bool:
    score_payload = record.get("score")
    if not isinstance(score_payload, dict):
        raise ArchiveError("R12 scored record lacks score payload")
    return bool(ScoreResult.from_dict(score_payload).unscorable)


def extract(reader: _ArchiveReader) -> dict[str, Any]:
    manifest = _locate_manifest(reader)
    records = manifest["records"]
    if len(records) != 750:
        raise ArchiveError("historical manifest has wrong row count")
    selected = {
        str(row.get("response_id")): row
        for row in records
        if str(row.get("source_id")) != "12448"
    }
    if len(selected) != 749:
        raise ArchiveError("historical manifest violates the source-12448 quarantine")
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    tuning: dict[str, dict[str, Any]] = {}
    for method in METHOD_DIRS:
        by_method[method], tuning[method] = _method_rows(reader, method)
        if str(tuning[method].get("relation_mode")) != METHOD_DIRS[method]:
            raise ArchiveError(f"R12 {method} tuning relation mode mismatch")
        if "theta" not in tuning[method]:
            raise ArchiveError(f"R12 {method} tuning has no train-only threshold")
    output: list[dict[str, Any]] = []
    for response_id, source in selected.items():
        method_records = [by_method[method].get(response_id) for method in METHOD_DIRS]
        if any(record is None for record in method_records):
            raise ArchiveError("R12 graph methods have different scored response sets")
        canonical = method_records[0]
        assert canonical is not None
        if (
            str(canonical.get("source_id")) != str(source.get("source_id"))
            or str(canonical.get("split")) != str(source.get("split"))
            or int(canonical.get("y")) != int(source.get("y"))
        ):
            raise ArchiveError("R12 scored IDs disagree with the historical manifest")
        if any(_is_graph_unscorable(record) for record in method_records if record is not None):
            continue
        risks = {
            method: _method_risk(method, by_method[method][response_id], tuning[method])
            for method in METHOD_DIRS
        }
        # The graph contract explicitly excludes answer-graph-empty responses
        # from every graph-method headline metric; do not invent any scalar.
        if any(value is None for value in risks.values()):
            continue
        if not all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in risks.values()):
            raise ArchiveError("R12 graph risk is non-finite or outside [0,1]")
        output.append({
            "source_id": str(source["source_id"]), "response_id": response_id,
            "split": str(source["split"]), "y": int(source["y"]),
            "scores": {"strict": risks["strict"], "support": risks["support"], "support_critical": risks["support_critical"]},
        })
    payload = {
        "protocol": GRAPH_REFERENCE_PROTOCOL,
        "source_job_id": "bt1fud5f0v4sbr1ru4jo",
        "archive_sha256": reader.checksum,
        "manifest_sha256": HISTORICAL_750_MANIFEST_SHA256,
        "frozen_thresholds": {method: float(tuning[method]["theta"]) for method in METHOD_DIRS},
        "records": sorted(output, key=lambda row: (row["split"], row["source_id"], row["response_id"])),
    }
    # The hard 599/147 and 300/299/75/72 validation is applied immediately
    # after the scalar-only payload is atomically written by ``main``.
    if len(payload["records"]) != 746:
        raise ArchiveError("R12 graph reference must retain exactly 746 graph-scorable responses")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r12-archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reader = _ArchiveReader(Path(args.r12_archive).resolve())
    try:
        payload = extract(reader)
    finally:
        reader.close()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    # Validate the persisted output, then print only safe counts/checksums.
    from src.candidate_agreement import load_graph_reference

    reference = load_graph_reference(output, manifest_sha256=HISTORICAL_750_MANIFEST_SHA256)
    print(json.dumps({
        "archive_sha256": reader.checksum, "manifest_sha256": reference.manifest_sha256,
        "records": len(reference.rows), "train": sum(row.split == "train" for row in reference.rows),
        "test": sum(row.split == "test" for row in reference.rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
