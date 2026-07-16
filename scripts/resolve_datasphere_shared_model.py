#!/usr/bin/env python3
"""Print the active immutable model directory from DataSphere shared storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    family_root = Path(args.shared_root) / "models" / "llama-3.1-8b"
    active_path = family_root / "active-model.json"
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"no valid active model manifest: {active_path}") from exc
    if active.get("model_id") != args.model_id:
        raise SystemExit(f"active model ID mismatch: {active.get('model_id')!r}")
    model_dir = family_root / str(active.get("model_dir", ""))
    if not (model_dir / ".hallu_smiles_model_ready").is_file():
        raise SystemExit(f"active model is not ready: {model_dir}")
    print(model_dir)


if __name__ == "__main__":
    main()
