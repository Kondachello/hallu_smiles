#!/usr/bin/env python3
"""Secret-free GCS transfer helper for the GCP Compute Engine runner."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _client():
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - image-only dependency
        raise SystemExit("google-cloud-storage is required in the GCP runner image") from exc
    return storage.Client()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--bucket", required=True)
    download.add_argument("--object", required=True)
    download.add_argument("--output", required=True)
    download.add_argument("--sha256", required=True)
    upload = subparsers.add_parser("upload")
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--object", required=True)
    upload.add_argument("--input", required=True)
    upload.add_argument("--if-absent", action="store_true")
    args = parser.parse_args()
    bucket = _client().bucket(args.bucket)
    if args.operation == "download":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            bucket.blob(args.object).download_to_filename(str(temporary))
            actual = _sha256(temporary)
            if actual != args.sha256:
                raise SystemExit("downloaded GCS object checksum does not match immutable input provenance")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return
    source = Path(args.input)
    if not source.is_file():
        raise SystemExit("GCS upload input is missing")
    bucket.blob(args.object).upload_from_filename(
        str(source), if_generation_match=0 if args.if_absent else None
    )


if __name__ == "__main__":
    main()
