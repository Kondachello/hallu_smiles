"""Atomic, content-addressed JSON cache with a cache-only replay guard.

Mirrors the house style in ``hallu_smiles/src/cache.py``: the key records every
input that can change structured output, writes go to a per-writer unique temp
file and are swapped in with ``os.replace`` (crash-safe, race-free), and a
``cache_only`` miss raises instead of silently making a network call.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CacheOnlyMissError(RuntimeError):
    """Raised when a cache-only replay needs an artifact that is not cached."""

    def __init__(self, namespace: str, key: str, path: str | Path):
        self.namespace = namespace
        self.key = key
        self.path = Path(path)
        super().__init__(
            f"cache-only miss for {namespace}: key={key} expected={self.path}"
        )


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_key(identity: dict, canonical_input: str) -> str:
    """sha256 over the reproducibility identity plus the canonical input text."""
    payload = canonical_json({"identity": identity, "input": canonical_input})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, root: str | Path, namespace: str, *, cache_only: bool = False):
        self.root = Path(root) / namespace
        self.namespace = namespace
        self.cache_only = cache_only

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        if self.cache_only:
            raise CacheOnlyMissError(self.namespace, key, path)
        return None

    def put(self, key: str, value: Any) -> None:
        if self.cache_only:
            # A warm-cache replay must prove it wrote nothing new.
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)  # atomic swap
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
