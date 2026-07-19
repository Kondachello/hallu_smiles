#!/usr/bin/env bash
# Bounded 3-QA compatibility probe: CPU orchestration -> Cloud Run -> Vertex AI.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/opt/hallu/models/all-MiniLM-L6-v2}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set the source commit in the Job template}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set the immutable Docker identity}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to RAGTruth project storage}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the writable Job directory}"
HALLU_GATEWAY_URL="${HALLU_GATEWAY_URL:?Set the Cloud Run origin in the rendered Job}"
: "${HALLU_GATEWAY_API_KEY:?Create a DataSphere Project secret named HALLU_GATEWAY_API_KEY}"

RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
GATEWAY_MANIFEST_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
HEALTH_REPORT="$RUN_ROOT/gateway-health.json"
STRICT_OUT="$RUN_ROOT/strict-extract"
REPLAY_OUT="$RUN_ROOT/cache-replay-extract"
PILOT_MANIFEST="$RUN_ROOT/qa_pilot_manifest.json"
METADATA="$RUN_ROOT/run_metadata.json"
USAGE_COUNTS="$RUN_ROOT/usage-counts.json"
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/hallu/models}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup() { unset HALLU_GATEWAY_API_KEY; }
trap cleanup EXIT

[[ "$HALLU_GATEWAY_URL" =~ ^https://[^/]+$ ]] || { echo "HALLU_GATEWAY_URL must be an HTTPS origin." >&2; exit 2; }
test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
test -f "$RUNTIME_MANIFEST" || { echo "runtime manifest is missing: $RUNTIME_MANIFEST" >&2; exit 2; }
test -f "$EMBEDDING_MODEL_PATH/config.json" || { echo "offline S-BERT snapshot is missing." >&2; exit 2; }
mkdir -p "$RUN_ROOT"
cp "$RUNTIME_MANIFEST" "$RUN_ROOT/runtime-manifest.json"
cp /opt/hallu/manifests/client.freeze.txt "$RUN_ROOT/client.freeze.txt"

"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" \
  --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-path "$EMBEDDING_MODEL_PATH" --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --report "$RUN_ROOT/cpu-runtime.json"

# Do not use curl -v or shell tracing: this request proves the Project secret
# reaches the gateway while keeping the bearer value out of logs and artifacts.
# Cloud Run's Google Front End reserves/intercepts the public ``/healthz`` path
# before requests reach the container.  The authenticated manifest is the
# gateway's authoritative readiness/identity probe and is validated below.
curl --fail --silent --show-error \
  -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" \
  "$HALLU_GATEWAY_URL/v1/hallu/manifest" > "$GATEWAY_MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$GATEWAY_MANIFEST_RAW" --logical-model "$("$CLIENT_PYTHON" - "$ROOT/config.yaml" <<'PY'
import yaml
import sys
print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['llm']['model'])
PY
)" --output "$GATEWAY_MANIFEST"
cp "$GATEWAY_MANIFEST" "$HEALTH_REPORT"
rm -f "$GATEWAY_MANIFEST_RAW"

"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" \
  --max-tokens 1024 --concurrency 2 > "$RUN_ROOT/runtime-config-identity.json"

"$CLIENT_PYTHON" "$ROOT/scripts/check_vertex_kggen_probe.py" \
  --config "$RUNTIME_CONFIG" --report "$RUN_ROOT/kggen-vertex-probe.json"
"$CLIENT_PYTHON" "$ROOT/scripts/check_vertex_verifier_probe.py" \
  --config "$RUNTIME_CONFIG" --report "$RUN_ROOT/verifier-vertex-probe.json"

"$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage extract \
  --relation-mode strict --qa-pilot --qa-pilot-manifest-out "$PILOT_MANIFEST" \
  --qa-pilot-limit 3 --output-dir "$STRICT_OUT"
"$CLIENT_PYTHON" - "$STRICT_OUT/extraction_summary.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
if payload.get('status') != 'ready' or payload.get('failures') != []:
    raise SystemExit('3-QA extraction is incomplete')
if payload.get('expected_sources') != 3 or payload.get('responses_completed') != 3:
    raise SystemExit('3-QA extraction did not produce three source/response graphs')
if payload.get('completed_records') != payload.get('expected_records'):
    raise SystemExit('3-QA extraction summary records are incomplete')
PY
test ! -s "$STRICT_OUT/failed_extractions.jsonl" || {
  echo "3-QA extraction wrote failed_extractions.jsonl" >&2
  exit 1
}

find "$RUN_ROOT/cache" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-before-replay.sha256"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage extract \
  --relation-mode strict --qa-pilot-manifest "$PILOT_MANIFEST" --qa-pilot-limit 3 \
  --output-dir "$REPLAY_OUT" --cache-only
cmp "$STRICT_OUT/extraction_summary.json" "$REPLAY_OUT/extraction_summary.json"
find "$RUN_ROOT/cache" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-after-replay.sha256"
cmp "$RUN_ROOT/cache-before-replay.sha256" "$RUN_ROOT/cache-after-replay.sha256"
test ! -s "$REPLAY_OUT/failed_extractions.jsonl" || {
  echo "cache-only replay wrote failed_extractions.jsonl" >&2
  exit 1
}
"$CLIENT_PYTHON" "$ROOT/scripts/summarize_vertex_probe_usage.py" \
  --kggen-probe "$RUN_ROOT/kggen-vertex-probe.json" \
  --verifier-probe "$RUN_ROOT/verifier-vertex-probe.json" \
  --live-usage "$STRICT_OUT/usage.jsonl" \
  --replay-usage "$REPLAY_OUT/usage.jsonl" --output "$USAGE_COUNTS"

"$CLIENT_PYTHON" - "$METADATA" "$EXPECTED_SOURCE_COMMIT" "$DATASPHERE_DOCKER_IMAGE_ID" "$GATEWAY_MANIFEST" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from gateway.core import canonical_manifest_sha256

manifest_path = Path(sys.argv[4])
raw = manifest_path.read_bytes()
manifest = json.loads(raw)
Path(sys.argv[1]).write_text(json.dumps({
    'state': 'completed',
    'mode': 'cpu-vertex-3qa-probe',
    'checked_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'source_commit': sys.argv[2],
    'datasphere_docker_image_id': sys.argv[3],
    'gateway_manifest_sha256': canonical_manifest_sha256(manifest),
    'gateway_manifest': manifest,
    'qa_pilot_limit': 3,
    'runs': ['kggen-schema-and-cluster', 'verifier-live-cache-only', 'strict-extract', 'strict-cache-only-replay'],
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
