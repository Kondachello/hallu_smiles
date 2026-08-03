#!/usr/bin/env python3
"""Create a redacted terminal artifact for a local DocRED evaluation."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


FORBIDDEN_PARTS = {"graphs", "cache", "checkpoints", "data"}
FORBIDDEN_NAMES = {"usage.jsonl", "docred.stdout.log", "docred.stderr.log"}


def _include(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return False
    return relative.name not in FORBIDDEN_NAMES and not relative.name.endswith(".raw.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if not root.is_dir() or not root.name.startswith("vertex-cpu-docred-kg-artifacts"):
        raise SystemExit("run root is not a local DocRED artifact directory")
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and _include(root, path):
                handle.add(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
    with tarfile.open(temporary, "r:gz") as handle:
        members = [member.name for member in handle.getmembers()]
        if not members or any(
            any(part in FORBIDDEN_PARTS for part in Path(name).parts)
            or Path(name).name in FORBIDDEN_NAMES
            for name in members
        ):
            raise SystemExit("refusing an incomplete or non-redacted DocRED archive")
    temporary.replace(archive)
    print(f"[ok] wrote redacted local DocRED artifact: {archive}")


if __name__ == "__main__":
    main()
