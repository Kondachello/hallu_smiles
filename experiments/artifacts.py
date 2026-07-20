"""Small, atomic, checksum-verified run archive primitives for the first framework slice."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_id, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle_id, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write(Path(path), (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    _atomic_write(Path(path), content)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {target}:{number}") from exc
    return rows


@dataclass(frozen=True)
class RunArchive:
    root: Path
    run_id: str

    @property
    def path(self) -> Path:
        return self.root / self.run_id

    @classmethod
    def create(cls, root: str | Path, *, run_id: str, manifest: Mapping[str, Any]) -> "RunArchive":
        archive = cls(Path(root), run_id)
        if archive.path.exists():
            raise FileExistsError(f"run archive already exists: {archive.path}")
        for relative in (
            "gold", "stages", "graphs", "grapheval", "hallugraph", "predictions",
            "evaluation", "audit", "payloads/sha256", "reports/tables", "reports/plots",
        ):
            (archive.path / relative).mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "run_id": run_id,
            "run_status": "created",
            "created_at_utc": utc_now(),
            "artifact_schema_version": 1,
            "gold_access_state": "hidden",
            **dict(manifest),
        }
        atomic_write_json(archive.path / "run_manifest.json", run_manifest)
        atomic_write_json(archive.path / "schema_versions.json", {"framework_archive": 1})
        return archive

    def write_json(self, relative: str, value: Any) -> Path:
        target = self.path / relative
        atomic_write_json(target, value)
        return target

    def write_jsonl(self, relative: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        target = self.path / relative
        atomic_write_jsonl(target, rows)
        return target

    def read_jsonl(self, relative: str) -> list[dict[str, Any]]:
        return read_jsonl(self.path / relative)

    def update_status(self, status: str, **extra: Any) -> None:
        target = self.path / "run_manifest.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["run_status"] = status
        manifest.update(extra)
        atomic_write_json(target, manifest)

    def put_payload(self, payload: Any, *, media_type: str = "application/json") -> dict[str, Any]:
        content = payload if isinstance(payload, bytes) else canonical_json(payload).encode("utf-8")
        digest = sha256_bytes(content)
        suffix = ".json" if media_type == "application/json" else ".bin"
        target = self.path / "payloads" / "sha256" / f"{digest}{suffix}"
        if not target.exists():
            _atomic_write(target, content)
        return {"sha256": digest, "path": str(target.relative_to(self.path)), "media_type": media_type}

    def seal_predictions(self, *, expected_response_ids: Iterable[str]) -> dict[str, Any]:
        predictions = self.read_jsonl("predictions/raw_predictions.jsonl")
        expected = set(map(str, expected_response_ids))
        actual = {str(row["response_id"]) for row in predictions}
        methods = sorted({str(row["method"]) for row in predictions})
        duplicates = len(predictions) != len({(row["method"], row["response_id"]) for row in predictions})
        missing_by_method = {
            method: sorted(expected - {str(row["response_id"]) for row in predictions if row["method"] == method})
            for method in methods
        }
        if duplicates or any(missing_by_method.values()):
            raise ValueError(f"cannot seal incomplete predictions: duplicates={duplicates}, missing={missing_by_method}")
        tracked = [
            self.path / "predictions/raw_predictions.jsonl",
            self.path / "predictions/paired_predictions.jsonl",
            self.path / "stages/stage_calls.jsonl",
        ]
        checksums = {str(path.relative_to(self.path)): sha256_file(path) for path in tracked if path.exists()}
        seal = {
            "run_id": self.run_id,
            "sealed_at_utc": utc_now(),
            "expected_response_count": len(expected),
            "methods": methods,
            "prediction_count": len(predictions),
            "file_sha256": checksums,
            "gold_access_state": "hidden",
        }
        self.write_json("prediction_seal.json", seal)
        self.update_status("predictions_sealed", predictions_sealed_at_utc=seal["sealed_at_utc"])
        return seal

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        manifest_path = self.path / "run_manifest.json"
        if not manifest_path.exists():
            errors.append("missing run_manifest.json")
        seal_path = self.path / "prediction_seal.json"
        if seal_path.exists():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            for relative, expected in seal.get("file_sha256", {}).items():
                target = self.path / relative
                if not target.exists() or sha256_file(target) != expected:
                    errors.append(f"checksum mismatch: {relative}")
        return {"run_id": self.run_id, "valid": not errors, "errors": errors}
