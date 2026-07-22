#!/usr/bin/env bash
# Replay one fully warm historical QA graph set. No secret or network call is needed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
: "${RUN_ROOT:?}" "${DATA_DIR:?}" "${HISTORICAL_CHECKPOINT_BASE:?}" "${RECORDED_GATEWAY_URL:?}" "${EXPECTED_SOURCE_COMMIT:?}"
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/current-cache"

test -x "$CLIENT_PYTHON"
test -f "$DATA_DIR/source_info.jsonl"
test -f "$DATA_DIR/response.jsonl"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" \
  --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-path /opt/hallu/models/all-MiniLM-L6-v2 --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --report "$RUN_ROOT/cpu-runtime.json"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_hhem_offline.py" \
  --model-path /opt/hallu/models/hhem-2.1-open --foundation-path /opt/hallu/models/flan-t5-base \
  --revision 0e7edb3689e710c52ba120086e8f91ea3ee87f23 --report "$RUN_ROOT/hhem-offline-smoke.json"
"$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_qa_cache.py" \
  --checkpoint-base "$HISTORICAL_CHECKPOINT_BASE" --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
  --runtime-manifest "$RUNTIME_MANIFEST" --output "$RUN_ROOT/historical-lineage.json"
HISTORICAL_CACHE_ROOT="$("$CLIENT_PYTHON" - "$RUN_ROOT/historical-lineage.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['historical_cache_root'])
PY
)"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_historical_cache_replay_config.py" \
  --base-config "$ROOT/config.yaml" --lineage "$RUN_ROOT/historical-lineage.json" \
  --gateway-url "$RECORDED_GATEWAY_URL" --data-dir "$DATA_DIR" \
  --cache-root "$RUN_ROOT/current-cache" --cache-read-root "$HISTORICAL_CACHE_ROOT" \
  --output "$RUN_ROOT/historical-cache-runtime.yaml" > "$RUN_ROOT/historical-cache-runtime-identity.json"
"$CLIENT_PYTHON" "$ROOT/scripts/historical_qa_cache_replay_probe.py" \
  --data-dir "$DATA_DIR" --output-root "$RUN_ROOT" \
  --hallugraph-config "$RUN_ROOT/historical-cache-runtime.yaml" \
  --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml" \
  --historical-cache-root "$HISTORICAL_CACHE_ROOT" --lineage "$RUN_ROOT/historical-lineage.json" \
  --run-id historical-cache-replay | tee "$RUN_ROOT/historical-cache-replay-summary.log"
