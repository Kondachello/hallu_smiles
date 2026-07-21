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
import re
from pathlib import Path


FOUNDATION_REPO = "google/flan-t5-base"
REQUIRED_FOUNDATION_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)
FOUNDATION_ASSIGNMENT = re.compile(
    r'^(?P<prefix>\s*foundation\s*=\s*)["\']google/flan-t5-base["\']\s*$', re.MULTILINE
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hhem-model-dir", required=True)
    parser.add_argument("--foundation-dir", required=True)
    args = parser.parse_args()

    hhem_dir = Path(args.hhem_model_dir)
    foundation_dir = Path(args.foundation_dir)
    hhem_config_path = hhem_dir / "config.json"
    hhem_configuration_path = hhem_dir / "configuration_hhem_v2.py"
    if not hhem_config_path.is_file():
        raise RuntimeError(f"HHEM config is missing: {hhem_config_path}")
    missing = [name for name in REQUIRED_FOUNDATION_FILES if not (foundation_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete pinned FLAN-T5 foundation at {foundation_dir}: {missing}")

    payload = json.loads(hhem_config_path.read_text(encoding="utf-8"))
    actual = payload.get("foundation", FOUNDATION_REPO)
    if actual not in (FOUNDATION_REPO, str(foundation_dir)):
        raise RuntimeError(
            f"unexpected HHEM foundation {actual!r}; expected {FOUNDATION_REPO!r} before local binding"
        )
    if not hhem_configuration_path.is_file():
        raise RuntimeError(f"HHEM custom configuration code is missing: {hhem_configuration_path}")

    # HHEM-2.1's ``HHEMv2Config`` keeps ``foundation`` as a class attribute;
    # its misspelled initializer does not populate it from config.json. Patch
    # the pinned local custom-code source before Transformers imports it.
    source = hhem_configuration_path.read_text(encoding="utf-8")
    local_assignment = f"foundation = {json.dumps(str(foundation_dir))}"
    patched, replacements = FOUNDATION_ASSIGNMENT.subn(
        lambda match: f"{match.group('prefix')}{json.dumps(str(foundation_dir))}", source, count=1
    )
    if replacements == 0 and local_assignment not in source:
        raise RuntimeError("could not find HHEM's google/flan-t5-base class attribute to bind")
    if replacements:
        hhem_configuration_path.write_text(patched, encoding="utf-8")
    payload["foundation"] = str(foundation_dir)
    hhem_config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "bound", "foundation_dir": str(foundation_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
