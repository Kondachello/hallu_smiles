#!/usr/bin/env bash
# Real CPU-only controlled shared-KGGen two-pass probe. Never enable shell tracing.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HALLU_GATEWAY_API_KEY:?Create DataSphere Project secret HALLU_GATEWAY_API_KEY}"
: "${HALLU_GATEWAY_URL:?Set Cloud Run origin}"; : "${RUN_ROOT:?}"; : "${CACHE_ROOT:?}"; : "${DATA_DIR:?}"; : "${RESPONSE_ID:?}"; : "${EXPECTED_SOURCE_COMMIT:?}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"; RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
trap 'unset HALLU_GATEWAY_API_KEY' EXIT
test -x "$CLIENT_PYTHON"; test -f "$DATA_DIR/source_info.jsonl"; test -f "$DATA_DIR/response.jsonl"; mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
MANIFEST_RAW="$RUN_ROOT/gateway-manifest.raw.json"; MANIFEST="$RUN_ROOT/gateway-manifest.json"; RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" --embedding-path /opt/hallu/models/all-MiniLM-L6-v2 --hhem-path /opt/hallu/models/hhem-2.1-open --expected-source-commit "$EXPECTED_SOURCE_COMMIT" --report "$RUN_ROOT/cpu-runtime.json"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_hhem_offline.py" --model-path /opt/hallu/models/hhem-2.1-open --foundation-path /opt/hallu/models/flan-t5-base --revision 0e7edb3689e710c52ba120086e8f91ea3ee87f23 --report "$RUN_ROOT/hhem-offline-smoke.json"
curl --fail --silent --show-error -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" "$HALLU_GATEWAY_URL/v1/hallu/manifest" > "$MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" --manifest "$MANIFEST_RAW" --logical-model openai/gemini-2.5-flash --output "$MANIFEST"; rm -f "$MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" --base-config "$ROOT/config.yaml" --gateway-manifest "$MANIFEST" --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" --output "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$CACHE_ROOT" --max-tokens 8192 --length-retry-max-tokens 12288 --length-retry-attempts 1 --concurrency 1 --max-retries 0 --retry-backoff-base-s 5 --retry-backoff-max-s 60 --retry-backoff-jitter-s 5 > "$RUN_ROOT/runtime-config-identity.json"
"$CLIENT_PYTHON" "$ROOT/scripts/shared_kggen_one_instance_controlled_live_probe.py" --data-dir "$DATA_DIR" --response-id "$RESPONSE_ID" --output-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" --run-id controlled-live --hallugraph-config "$RUNTIME_CONFIG" --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml" --gateway-manifest "$MANIFEST" | tee "$RUN_ROOT/controlled-live-summary.log"
