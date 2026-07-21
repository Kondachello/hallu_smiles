#!/usr/bin/env python3
"""Bind HHEM's custom-code foundation reference to a pinned local directory.

HHEM-2.1's remote model code calls ``AutoConfig.from_pretrained`` and
``AutoTokenizer.from_pretrained`` with the ``foundation`` field from its own
``config.json``.  A Hugging Face cache is not a stable interface between hub
client versions, so the CPU image deliberately rewrites that one field to the
immutable local FLAN-T5 asset directory during its network-enabled build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FOUNDATION_REPO = "google/flan-t5-base"
REQUIRED_FOUNDATION_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hhem-model-dir", required=True)
    parser.add_argument("--foundation-dir", required=True)
    args = parser.parse_args()

    hhem_dir = Path(args.hhem_model_dir)
    foundation_dir = Path(args.foundation_dir)
    hhem_config_path = hhem_dir / "config.json"
    if not hhem_config_path.is_file():
        raise RuntimeError(f"HHEM config is missing: {hhem_config_path}")
    missing = [name for name in REQUIRED_FOUNDATION_FILES if not (foundation_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete pinned FLAN-T5 foundation at {foundation_dir}: {missing}")

    payload = json.loads(hhem_config_path.read_text(encoding="utf-8"))
    actual = payload.get("foundation")
    if actual != FOUNDATION_REPO:
        raise RuntimeError(
            f"unexpected HHEM foundation {actual!r}; expected {FOUNDATION_REPO!r} before local binding"
        )
    payload["foundation"] = str(foundation_dir)
    hhem_config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "bound", "foundation_dir": str(foundation_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
