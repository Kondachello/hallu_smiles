#!/usr/bin/env bash
# CPU-only, no-network two-pass shared-KGGen cache probe for DataSphere.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT in the rendered Job}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to read-only RAGTruth Project storage}"
CACHE_ROOT="${CACHE_ROOT:?Set CACHE_ROOT to a dedicated Project-storage namespace}"
HISTORICAL_CACHE_BASE="${HISTORICAL_CACHE_BASE:?Set historical cache base in the rendered Job}"
RESPONSE_ID="${RESPONSE_ID:?Set one explicit response id in the rendered Job}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set source commit in the rendered Job}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set immutable Docker identity in the rendered Job}"
PROBE_RUN_ID="shared-kggen-two-pass"

test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
test -f "$DATA_DIR/source_info.jsonl" || { echo "source_info.jsonl is missing from mounted RAGTruth." >&2; exit 2; }
test -f "$DATA_DIR/response.jsonl" || { echo "response.jsonl is missing from mounted RAGTruth." >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
export PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

actual_commit="$(git -C "$ROOT" rev-parse HEAD)"
test "$actual_commit" = "$EXPECTED_SOURCE_COMMIT"

# This discovery is read-only. It documents whether the old 100-QA checkpoint
# tree is mounted, but does not claim that its live KG cache is compatible with
# this fake controlled track and never writes below that tree.
"$CLIENT_PYTHON" - "$HISTORICAL_CACHE_BASE" "$RUN_ROOT/historical-kg-cache-candidates.json" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
roots = sorted(str(path.parent) for path in base.glob("**/kg") if path.is_dir()) if base.is_dir() else []
Path(sys.argv[2]).write_text(json.dumps({
    "historical_cache_base": str(base),
    "base_exists": base.is_dir(),
    "kg_roots": roots,
    "mode": "read_only_discovery",
    "compatibility_claimed": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[shared-kggen-mock] stage=two-pass response_id=$RESPONSE_ID"
"$CLIENT_PYTHON" "$ROOT/examples/mock_shared_kggen_one_instance.py" \
  --source-info "$DATA_DIR/source_info.jsonl" --responses "$DATA_DIR/response.jsonl" \
  --response-id "$RESPONSE_ID" --output-root "$RUN_ROOT/probe" \
  --cache-root "$CACHE_ROOT" --run-id "$PROBE_RUN_ID" --hallugraph-config "$ROOT/config.yaml" \
  | tee "$RUN_ROOT/two-pass-summary.log"

REPORT="$RUN_ROOT/probe/$PROBE_RUN_ID-materialize/reports/shared_kggen_two_pass_report.json"
test -f "$REPORT" || { echo "two-pass report is missing" >&2; exit 2; }
"$CLIENT_PYTHON" - "$REPORT" "$CACHE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["materialize_kggen_api_calls"] > 0
assert report["cache_replay_kggen_api_calls"] == 0
assert report["shared_graph_consistent_across_passes"] is True
assert Path(report["cache_root"]).resolve() == Path(sys.argv[2]).resolve()
print(json.dumps({
    "status": "passed",
    "materialize_kggen_api_calls": report["materialize_kggen_api_calls"],
    "cache_replay_kggen_api_calls": report["cache_replay_kggen_api_calls"],
    "shared_graph_sha256": report["shared_graph_sha256"],
}, sort_keys=True))
PY
echo "[shared-kggen-mock] completed"
