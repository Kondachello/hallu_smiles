#!/usr/bin/env bash
# Replay reproducibly selected fully warm historical QA graph sets. The only network request is the
# authenticated gateway manifest: its pinned identity reconstructs the old cache key.
# No request is ever made to a language model.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
: "${RUN_ROOT:?}" "${DATA_DIR:?}" "${HISTORICAL_CHECKPOINT_BASE:?}" "${RECORDED_GATEWAY_URL:?}" "${EXPECTED_SOURCE_COMMIT:?}"
: "${HALLU_GATEWAY_API_KEY:?Create DataSphere Project secret HALLU_GATEWAY_API_KEY}"
REPLAY_COUNT="${REPLAY_COUNT:-1}"
REPLAY_SELECTION_SEED="${REPLAY_SELECTION_SEED:-20260722}"
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/current-cache"
trap 'unset HALLU_GATEWAY_API_KEY' EXIT

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
MANIFEST_RAW="$RUN_ROOT/historical-gateway-manifest.raw.json"
MANIFEST="$RUN_ROOT/historical-gateway-manifest.json"
# Transient 429/5xx/network errors are retried with exponential backoff and jitter
# (same policy as the live gateway calls); 4xx fails fast since retrying won't help.
fetch_gateway_manifest() {
  local out="$1" attempt=1 max_attempts=5 delay=5 http_code
  while :; do
    http_code="$(curl --silent --show-error -o "$out" -w '%{http_code}' \
      -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" "$RECORDED_GATEWAY_URL/v1/hallu/manifest")" || http_code="000"
    if [[ "$http_code" == "200" ]]; then
      return 0
    fi
    if { [[ "$http_code" == "429" ]] || [[ "$http_code" =~ ^5[0-9][0-9]$ ]] || [[ "$http_code" == "000" ]]; } \
        && (( attempt < max_attempts )); then
      echo "gateway manifest fetch got HTTP $http_code (attempt $attempt/$max_attempts); retrying in ${delay}s" >&2
      sleep "$((delay + RANDOM % 6))"
      attempt=$((attempt + 1))
      delay=$((delay * 2 > 60 ? 60 : delay * 2))
      continue
    fi
    echo "gateway manifest fetch failed with HTTP $http_code (attempt $attempt/$max_attempts)" >&2
    return 1
  done
}
fetch_gateway_manifest "$MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$MANIFEST_RAW" --logical-model openai/gemini-2.5-flash --output "$MANIFEST"
rm -f "$MANIFEST_RAW"
ACTUAL_MANIFEST_SHA256="$("$CLIENT_PYTHON" - "$MANIFEST" <<'PY'
import json, sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding='utf-8'))))
PY
)"
EXPECTED_MANIFEST_SHA256="$("$CLIENT_PYTHON" - "$RUN_ROOT/historical-lineage.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['gateway_manifest_sha256'])
PY
)"
test "$ACTUAL_MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "gateway manifest no longer matches the immutable historical cache lineage" >&2
  exit 2
}
HISTORICAL_LLM_FINGERPRINT="$("$CLIENT_PYTHON" - "$RUN_ROOT/historical-lineage.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['llm_runtime_fingerprint'])
PY
)"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$MANIFEST" \
  --gateway-url "$RECORDED_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$RUN_ROOT/historical-cache-runtime.yaml" --data-dir "$DATA_DIR" \
  --work-dir "$RUN_ROOT" --cache-root "$RUN_ROOT/current-cache" \
  --llm-runtime-fingerprint-override "$HISTORICAL_LLM_FINGERPRINT" \
  --max-tokens 16384 --concurrency 1 --max-retries 0 --retry-backoff-base-s 5 \
  --retry-backoff-max-s 60 --retry-backoff-jitter-s 0 --cv-folds 5 \
  > "$RUN_ROOT/historical-cache-runtime-identity.json"
"$CLIENT_PYTHON" "$ROOT/scripts/historical_qa_cache_replay_probe.py" \
  --data-dir "$DATA_DIR" --output-root "$RUN_ROOT" \
  --hallugraph-config "$RUN_ROOT/historical-cache-runtime.yaml" \
  --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml" \
  --historical-cache-root "$HISTORICAL_CACHE_ROOT" --lineage "$RUN_ROOT/historical-lineage.json" \
  --run-id historical-cache-replay --replay-count "$REPLAY_COUNT" \
  --replay-selection-seed "$REPLAY_SELECTION_SEED" | tee "$RUN_ROOT/historical-cache-replay-summary.log"
