"""Small immutable JSON cache and JSONL artifact writer for local execution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .errors import CacheIntegrityError
from .models import canonical_json


class JsonFileCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        if not namespace.replace("_", "").replace("-", "").isalnum() or len(key) < 16:
            raise CacheIntegrityError("unsafe cache namespace or key")
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> Mapping[str, Any] | None:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheIntegrityError(f"invalid cache entry: {path}") from exc
        if not isinstance(value, dict):
            raise CacheIntegrityError(f"cache entry is not an object: {path}")
        return value

    def put_immutable(self, namespace: str, key: str, value: Mapping[str, Any]) -> None:
        path = self._path(namespace, key)
        payload = canonical_json(dict(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current != payload:
                raise CacheIntegrityError(f"immutable cache collision: {path}")
            return
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, relative: str, record: Mapping[str, Any]) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(dict(record))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def append_jsonl(self, relative: str, record: Mapping[str, Any]) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(record)) + "\n")
        return path
