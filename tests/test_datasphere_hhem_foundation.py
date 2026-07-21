"""Regression coverage for the local HHEM foundation binding in the CPU image."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_datasphere_hhem_foundation.py"


def test_prepare_hhem_foundation_replaces_remote_reference_with_local_pinned_dir(tmp_path):
    hhem_dir = tmp_path / "hhem"
    foundation_dir = tmp_path / "flan-t5-base"
    hhem_dir.mkdir()
    foundation_dir.mkdir()
    (hhem_dir / "config.json").write_text(
        json.dumps({"foundation": "google/flan-t5-base", "model_type": "hhem-v2"}),
        encoding="utf-8",
    )
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
    ):
        (foundation_dir / name).write_text("fixture", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--hhem-model-dir",
            str(hhem_dir),
            "--foundation-dir",
            str(foundation_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "bound"
    assert json.loads((hhem_dir / "config.json").read_text(encoding="utf-8"))["foundation"] == str(
        foundation_dir
    )
