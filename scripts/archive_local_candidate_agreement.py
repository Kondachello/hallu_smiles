#!/usr/bin/env python3
"""Archive only redacted candidate-agreement experiment artifacts."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


FORBIDDEN_NAMES = {
    "usage.jsonl", "stdout.log", "stderr.log", "gateway-manifest.raw.json", ".gateway-curl",
}
FORBIDDEN_PARTS = {"cache", "data", "score-checkpoints", "huggingface", "pip-cache"}


def _include(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in FORBIDDEN_PARTS for part in relative.parts) and relative.name not in FORBIDDEN_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    allowed_prefixes = (
        "ragtruth-candidate-agreement-artifacts-",
        "ragtruth-candidate-agreement-paired-",
    )
    if not root.is_dir() or not root.name.startswith(allowed_prefixes):
        raise SystemExit("run root is not a local candidate-agreement artifact directory")
    archive = Path(args.archive).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and _include(root, path):
                handle.add(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
    with tarfile.open(temporary, "r:gz") as handle:
        members = [Path(member.name) for member in handle.getmembers()]
        if not members or any(
            any(part in FORBIDDEN_PARTS for part in member.parts) or member.name in FORBIDDEN_NAMES
            for member in members
        ):
            raise SystemExit("refusing a non-redacted candidate-agreement archive")
    temporary.replace(archive)
    print(f"[ok] wrote redacted candidate-agreement archive: {archive}")


if __name__ == "__main__":
    main()
