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

    models_root = Path(args.shared_root) / "models"
    candidates: list[tuple[Path, dict[str, object]]] = []
    for active_path in sorted(models_root.glob("*/active-model.json")):
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if active.get("model_id") == args.model_id:
            candidates.append((active_path.parent, active))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one active model manifest for {args.model_id!r} under {models_root}; "
            f"found {len(candidates)}"
        )
    family_root, active = candidates[0]
    model_dir = family_root / str(active.get("model_dir", ""))
    if not (model_dir / ".hallu_smiles_model_ready").is_file():
        raise SystemExit(f"active model is not ready: {model_dir}")
    print(model_dir)


if __name__ == "__main__":
    main()
