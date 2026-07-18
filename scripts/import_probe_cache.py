#!/usr/bin/env python3
"""Safely import validated 3-QA caches and manifest into an API pilot run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from validate_api_probe_artifact import ProbeArtifact, validate_probe_artifact
except ImportError:  # pragma: no cover - package import in tests
    from scripts.validate_api_probe_artifact import ProbeArtifact, validate_probe_artifact


CACHE_FILE_RE = re.compile(r"^[0-9a-f]{64}\.json$")


def _ensure_safe_directory(path: Path, *, root: Path | None = None) -> None:
    """Create ``path`` without following symlinks in its controlled ancestry."""
    if root is not None:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes pilot destination: {path}") from exc
        current = root
        components = relative.parts
    else:
        current = Path(path.anchor)
        components = path.parts[1:]
    for component in components:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"symlink is forbidden in pilot destination: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"pilot destination component is not a directory: {current}")
        else:
            current.mkdir()


def _atomic_copy_bytes(destination: Path, data: bytes, *, root: Path) -> str:
    """Create a file atomically; an existing byte-identical cache is reusable."""
    digest = hashlib.sha256(data).hexdigest()
    _ensure_safe_directory(destination.parent, root=root)
    if destination.is_symlink():
        raise ValueError(f"symlink is forbidden as a cache destination: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise ValueError(f"cache destination is not a regular file: {destination}")
        existing = destination.read_bytes()
        if existing != data:
            raise ValueError(f"refusing to overwrite conflicting probe cache: {destination}")
        return digest
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return digest


def _cache_destination(member_name: str, destination: Path) -> Path | None:
    parts = PurePosixPath(member_name).parts
    for cache_kind in ("kg", "verdicts"):
        marker = (".cache", cache_kind)
        for index in range(len(parts) - 2):
            if tuple(parts[index : index + 2]) != marker:
                continue
            relative = parts[index + 2 :]
            if len(relative) != 1 or not CACHE_FILE_RE.fullmatch(relative[0]):
                raise ValueError(f"invalid {cache_kind} cache member: {member_name!r}")
            return destination / ".cache" / cache_kind / relative[0]
    return None


def import_probe_cache(
    artifact_path: str | Path,
    destination: str | Path,
    *,
    expected_commit: str | None = None,
    expected_model: str | None = None,
    expected_api_base: str | None = None,
    secret: str | None = None,
) -> dict[str, Any]:
    """Validate ``artifact_path`` and atomically import its immutable cache files."""
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError(f"pilot destination must not be a symlink: {destination}")
    _ensure_safe_directory(destination)
    if not destination.is_dir():
        raise ValueError(f"pilot destination is not a real directory: {destination}")
    gate = validate_probe_artifact(
        artifact_path,
        expected_commit=expected_commit,
        expected_model=expected_model,
        expected_api_base=expected_api_base,
        secret=secret,
    )
    imported: dict[str, int] = {"kg": 0, "verdicts": 0}
    digests: dict[str, str] = {}
    with ProbeArtifact(artifact_path) as artifact:
        manifest = artifact.raw("qa_pilot_manifest.json")
        digest = _atomic_copy_bytes(
            destination / "qa_pilot_manifest.json", manifest, root=destination
        )
        if digest != gate["manifest_sha256"]:
            raise ValueError("manifest changed between validation and import")
        for member in artifact.regular_members():
            target = _cache_destination(member.name, destination)
            if target is None:
                continue
            handle = artifact.tar.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read cache member {member.name!r}")
            data = handle.read()
            _atomic_copy_bytes(target, data, root=destination)
            kind = target.parent.name
            imported[kind] += 1
            digests[str(target.relative_to(destination))] = hashlib.sha256(data).hexdigest()
    if imported["kg"] == 0:
        raise ValueError("validated probe artifact contains no KG cache files")
    report = {
        "protocol": "hallu-api-probe-cache-import-v1",
        "status": "ready",
        "source_commit": gate["source_commit"],
        "model": gate["model"],
        "manifest_sha256": gate["manifest_sha256"],
        "imported": imported,
        "digests": dict(sorted(digests.items())),
    }
    (destination / "probe_cache_import.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-api-base")
    parser.add_argument("--secret-env")
    args = parser.parse_args()
    secret = os.environ.get(args.secret_env) if args.secret_env else None
    report = import_probe_cache(
        args.artifact,
        args.destination,
        expected_commit=args.expected_commit,
        expected_model=args.expected_model,
        expected_api_base=args.expected_api_base,
        secret=secret,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
