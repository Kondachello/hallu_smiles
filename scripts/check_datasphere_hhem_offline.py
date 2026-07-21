#!/usr/bin/env python3
"""Prove that the pinned local HHEM snapshot can score a pair with no network."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--foundation-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the HHEM offline smoke check")
    model_path = Path(args.model_path)
    foundation_path = Path(args.foundation_path)
    if not (model_path / "config.json").is_file() or not (model_path / "model.safetensors").is_file():
        raise RuntimeError(f"incomplete HHEM snapshot: {model_path}")
    hhem_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if hhem_config.get("foundation") != str(foundation_path):
        raise RuntimeError("HHEM config is not bound to the expected local FLAN-T5 foundation")
    hhem_configuration = model_path / "configuration_hhem_v2.py"
    if not hhem_configuration.is_file() or str(foundation_path) not in hhem_configuration.read_text(encoding="utf-8"):
        raise RuntimeError("HHEM custom configuration code is not bound to the local FLAN-T5 foundation")
    if not (foundation_path / "config.json").is_file() or not (foundation_path / "spiece.model").is_file():
        raise RuntimeError(f"incomplete local FLAN-T5 foundation: {foundation_path}")

    # This is intentionally the same Transformers load and ``predict`` protocol
    # used by graph_eval.nli.hhem, but remains standalone so Docker can verify the
    # image before the Job clones the repository source tree.
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), trust_remote_code=True, revision=args.revision, local_files_only=True
    )
    scores = model.predict([("The archive is in Northbridge.", "The archive is in Northbridge.")])
    if len(scores) != 1 or not math.isfinite(float(scores[0])) or not 0.0 <= float(scores[0]) <= 1.0:
        raise RuntimeError("offline HHEM predict() did not return one finite score in [0, 1]")
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "network_mode": "offline",
        "model_path": str(model_path),
        "foundation_path": str(foundation_path),
        "revision": args.revision,
        "n_scores": len(scores),
        "score_in_unit_interval": True,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
