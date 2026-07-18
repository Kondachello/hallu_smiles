#!/usr/bin/env python3
"""Validate an authenticated gateway manifest without displaying its credential."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_datasphere_vertex_config import _validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--logical-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    _validate_manifest(manifest, args.logical_model)
    Path(args.output).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
