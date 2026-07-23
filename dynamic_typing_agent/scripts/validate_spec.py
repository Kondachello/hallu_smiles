#!/usr/bin/env python3
"""Dependency-free validation for the architecture/prompt/example specification."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts" / "v1"
VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    manifest = json.loads((PROMPTS / "manifest.json").read_text(encoding="utf-8"))
    prompt_ids: set[str] = set()
    for entry in manifest["entries"]:
        prompt_id = entry["prompt_id"]
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate prompt_id: {prompt_id}")
        prompt_ids.add(prompt_id)
        for key in ("system", "user", "schema"):
            path = PROMPTS / entry[key]
            if not path.is_file():
                raise FileNotFoundError(path)
        user = (PROMPTS / entry["user"]).read_text(encoding="utf-8")
        actual = set(VARIABLE.findall(user))
        expected = set(entry["required_variables"])
        if actual != expected:
            raise ValueError(f"{prompt_id}: template variables {actual} != {expected}")
        json.loads((PROMPTS / entry["schema"]).read_text(encoding="utf-8"))

    no_gold = read_jsonl(ROOT / "examples" / "dynamic_typing_20.no_gold.jsonl")
    expectations = read_jsonl(ROOT / "examples" / "dynamic_typing_20.expectations.jsonl")
    if len(no_gold) != 20 or len(expectations) != 20:
        raise ValueError("example corpus must contain exactly 20 linked cases")
    if {row["case_id"] for row in no_gold} != {row["case_id"] for row in expectations}:
        raise ValueError("no-gold and expectation case IDs differ")
    print(f"valid: {len(prompt_ids)} prompts, {len(no_gold)} no-gold examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

