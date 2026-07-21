#!/usr/bin/env bash
# Exactly one real RAGTruth response through HalluGraph + GraphEval in DataSphere.
# The Project injects HALLU_GATEWAY_API_KEY. Do not enable shell tracing here.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/opt/hallu/models/all-MiniLM-L6-v2}"
HHEM_MODEL_PATH="${HHEM_MODEL_PATH:-/opt/hallu/models/hhem-2.1-open}"
HHEM_REVISION="${HHEM_REVISION:-0e7edb3689e710c52ba120086e8f91ea3ee87f23}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set the source commit in the Job template}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set immutable Docker identity in the Job template}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the read-only RAGTruth project storage}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the writable Job directory}"
RESPONSE_ID="${RESPONSE_ID:?Set one explicit RAGTruth response id in the rendered Job}"
HALLU_GATEWAY_URL="${HALLU_GATEWAY_URL:?Set the Cloud Run origin in the rendered Job}"
: "${HALLU_GATEWAY_API_KEY:?Create a DataSphere Project secret named HALLU_GATEWAY_API_KEY}"

RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
MANIFEST_RAW="$RUN_ROOT/gateway-manifest.raw.json"
MANIFEST="$RUN_ROOT/gateway-manifest.json"
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/hallu/models}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup() { unset HALLU_GATEWAY_API_KEY; }
trap cleanup EXIT
[[ "$HALLU_GATEWAY_URL" =~ ^https://[^/]+$ ]] || { echo "HALLU_GATEWAY_URL must be an HTTPS origin." >&2; exit 2; }
test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
test -f "$DATA_DIR/source_info.jsonl" || { echo "source_info.jsonl is missing from mounted RAGTruth." >&2; exit 2; }
test -f "$DATA_DIR/response.jsonl" || { echo "response.jsonl is missing from mounted RAGTruth." >&2; exit 2; }
mkdir -p "$RUN_ROOT"

echo "[live-one-instance] stage=runtime-preflight response_id=$RESPONSE_ID"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" \
  --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-path "$EMBEDDING_MODEL_PATH" --hhem-path "$HHEM_MODEL_PATH" \
  --expected-source-commit "$EXPECTED_SOURCE_COMMIT" --report "$RUN_ROOT/cpu-runtime.json"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_hhem_offline.py" \
  --model-path "$HHEM_MODEL_PATH" --revision "$HHEM_REVISION" --report "$RUN_ROOT/hhem-offline-smoke.json"

# This proves the Job received the Project secret. The bearer value is never
# echoed, and the raw response is validated then replaced by the safe manifest.
echo "[live-one-instance] stage=gateway-manifest"
curl --fail --silent --show-error -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" \
  "$HALLU_GATEWAY_URL/v1/hallu/manifest" > "$MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$MANIFEST_RAW" --logical-model "$("$CLIENT_PYTHON" - "$ROOT/config.yaml" <<'PY'
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['llm']['model'])
PY
)" --output "$MANIFEST"
rm -f "$MANIFEST_RAW"

echo "[live-one-instance] stage=runtime-config"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" \
  --max-tokens 8192 --length-retry-max-tokens 12288 --length-retry-attempts 1 \
  --concurrency 1 --max-retries 7 --retry-backoff-base-s 5 \
  > "$RUN_ROOT/runtime-config-identity.json"

echo "[live-one-instance] stage=paired-inference response_id=$RESPONSE_ID"
"$CLIENT_PYTHON" "$ROOT/scripts/ragtruth_one_instance_live_probe.py" \
  --data-dir "$DATA_DIR" --response-id "$RESPONSE_ID" --output-root "$RUN_ROOT" \
  --run-id "paired-live" --hallugraph-config "$RUNTIME_CONFIG" \
  --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.live.yaml" \
  --gateway-manifest "$MANIFEST" | tee "$RUN_ROOT/live-probe-summary.log"

echo "[live-one-instance] completed; audit=$RUN_ROOT/paired-live/audit/live_one_instance_events.jsonl"
