from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_cache_replay.py"


def _write_outputs(directory: Path, *, cache_hit: bool, risk: float = 0.25) -> None:
    directory.mkdir()
    for filename, content in {
        "metrics.csv": "metric,value\nauc,0.5\n",
        "summary_metrics.csv": "metric,value\nf1,0.5\n",
        "tuning.json": '{"theta":0.5}\n',
    }.items():
        (directory / filename).write_text(content, encoding="utf-8")
    record = {
        "response_id": "fixture",
        "score": {
            "risk": risk,
            "relation_audits": [{"verifier_cache_hit": cache_hit, "verdict": "entailed"}],
            "critical": {
                "claim_audits": [{"verifier_cache_hit": cache_hit, "verdict": "unknown"}]
            },
        },
    }
    (directory / "scored.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _run(live: Path, replay: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--live-dir", str(live), "--replay-dir", str(replay)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_replay_allows_only_cache_observability_differences(tmp_path):
    live = tmp_path / "live"
    replay = tmp_path / "replay"
    _write_outputs(live, cache_hit=False)
    _write_outputs(replay, cache_hit=True)

    result = _run(live, replay)
    assert result.returncode == 0, result.stderr


def test_replay_rejects_scientific_score_difference(tmp_path):
    live = tmp_path / "live"
    replay = tmp_path / "replay"
    _write_outputs(live, cache_hit=False, risk=0.25)
    _write_outputs(replay, cache_hit=True, risk=0.5)

    result = _run(live, replay)
    assert result.returncode != 0
    assert "scientific content" in result.stderr
