#!/usr/bin/env python3
"""Materialize the pinned public DocRED evaluation files on project storage.

The downloaded corpus is not committed into this repository.  Its pinned Hub
revision and per-file SHA-256 inventory are written next to the files, making a
later DataSphere resume offline and reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.docred import DOCRED_FILES, DOCRED_HF_REPO, DOCRED_HF_REVISION, sha256_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository", default=DOCRED_HF_REPO)
    parser.add_argument("--revision", default=DOCRED_HF_REVISION)
    args = parser.parse_args()

    root = Path(args.output_dir)
    metadata_path = root / "dataset-metadata.json"
    expected = {name: root / filename for name, filename in DOCRED_FILES.items()}
    if metadata_path.exists() and all(path.is_file() for path in expected.values()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("repository") != args.repository or metadata.get("revision") != args.revision:
            raise SystemExit("existing DocRED directory has a different pinned repository/revision")
        actual = {name: sha256_file(path) for name, path in expected.items()}
        if actual != metadata.get("files_sha256"):
            raise SystemExit("existing DocRED file checksum differs from recorded metadata")
        print(json.dumps({"status": "reused", **metadata}, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=True)
    # The Docker image keeps HF offline for models.  Dataset materialisation is
    # a bounded, public, one-time exception before the run returns offline.
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download

    for name, filename in DOCRED_FILES.items():
        downloaded = Path(hf_hub_download(
            repo_id=args.repository,
            repo_type="dataset",
            revision=args.revision,
            filename=f"data/{filename}",
        ))
        shutil.copy2(downloaded, expected[name])
    metadata = {
        "protocol": "docred-materialization-v1",
        "repository": args.repository,
        "revision": args.revision,
        "files_sha256": {name: sha256_file(path) for name, path in expected.items()},
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps({"status": "downloaded", **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
