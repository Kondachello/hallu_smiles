#!/usr/bin/env bash
# Fixed, resumable DocRED KGGen evaluation: CPU -> Cloud Run -> Vertex AI.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/opt/hallu/models/all-MiniLM-L6-v2}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set the source commit in the Job template}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set the immutable Docker identity}"
EXPECTED_GATEWAY_MANIFEST_SHA256="${EXPECTED_GATEWAY_MANIFEST_SHA256:?Set the verified gateway manifest hash}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the writable Job directory}"
HALLU_GATEWAY_URL="${HALLU_GATEWAY_URL:?Set the Cloud Run origin in the rendered Job}"
DOCRED_BUDGET_EUR="${DOCRED_BUDGET_EUR:-10.5}"
: "${HALLU_GATEWAY_API_KEY:?Create a DataSphere Project secret named HALLU_GATEWAY_API_KEY}"

DATASET_REVISION="7985b4e0371e6c61a756feb41b7b27becf71c666"
DOCRED_DATA_DIR="$DS_PROJECT_HOME/hallu_smiles/shared/docred/thunlp-docred-$DATASET_REVISION"
GATEWAY_MANIFEST_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
RUNTIME_IDENTITY="$RUN_ROOT/runtime-config-identity.json"
LIVE_OUT="$RUN_ROOT/docred-live"
REPLAY_OUT="$RUN_ROOT/docred-replay"
METADATA="$RUN_ROOT/run_metadata.json"
USAGE_COUNTS="$RUN_ROOT/usage-counts.json"
CHECKPOINT_ROOT=""

export PYTHONHASHSEED=42 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/hallu/models}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
  status=$?
  trap - EXIT
  unset HALLU_GATEWAY_API_KEY
  exit "$status"
}
trap cleanup EXIT

[[ "$HALLU_GATEWAY_URL" =~ ^https://[^/]+$ ]] || { echo "HALLU_GATEWAY_URL must be an HTTPS origin." >&2; exit 2; }
test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
test -f "$RUNTIME_MANIFEST" || { echo "runtime manifest is missing: $RUNTIME_MANIFEST" >&2; exit 2; }
test -f "$EMBEDDING_MODEL_PATH/config.json" || { echo "offline S-BERT snapshot is missing." >&2; exit 2; }
"$CLIENT_PYTHON" - "$DOCRED_BUDGET_EUR" <<'PY'
import sys
budget = float(sys.argv[1])
if not 0.0 < budget <= 10.5:
    raise SystemExit('DOCRED_BUDGET_EUR must be in (0, 10.5]')
PY
mkdir -p "$RUN_ROOT"
cp "$RUNTIME_MANIFEST" "$RUN_ROOT/runtime-manifest.json"
cp /opt/hallu/manifests/client.freeze.txt "$RUN_ROOT/client.freeze.txt"

"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" \
  --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-path "$EMBEDDING_MODEL_PATH" --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --report "$RUN_ROOT/cpu-runtime.json"

# Dataset materialisation is a one-time public operation on project disk.  Once
# checksummed, all dataset/model reads are offline for this and later resumes.
env -u HF_HUB_OFFLINE "$CLIENT_PYTHON" "$ROOT/scripts/fetch_docred_data.py" \
  --output-dir "$DOCRED_DATA_DIR" > "$RUN_ROOT/dataset-materialization.json"
export HF_HUB_OFFLINE=1

curl --fail --silent --show-error \
  -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" \
  "$HALLU_GATEWAY_URL/v1/hallu/manifest" > "$GATEWAY_MANIFEST_RAW"
"$CLIENT_PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$GATEWAY_MANIFEST_RAW" --logical-model "$("$CLIENT_PYTHON" - "$ROOT/config.yaml" <<'PY'
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['llm']['model'])
PY
)" --output "$GATEWAY_MANIFEST"
rm -f "$GATEWAY_MANIFEST_RAW"
GATEWAY_MANIFEST_SHA256="$("$CLIENT_PYTHON" - "$GATEWAY_MANIFEST" <<'PY'
import json
import sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding='utf-8'))))
PY
)"
test "$GATEWAY_MANIFEST_SHA256" = "$EXPECTED_GATEWAY_MANIFEST_SHA256" || {
  echo "gateway manifest changed since the validated gate" >&2; exit 2;
}

CHECKPOINT_ROOT="$DS_PROJECT_HOME/hallu_smiles/checkpoints/docred-kg/docred-kg-v1-$GATEWAY_MANIFEST_SHA256"
PERSISTENT_MANIFEST="$CHECKPOINT_ROOT/docred_manifest.json"
CHECKPOINT_IDENTITY="$CHECKPOINT_ROOT/checkpoint-identity.json"
mkdir -p "$CHECKPOINT_ROOT/kg"
CACHE_IDENTITY_ARGS=()
if test -f "$CHECKPOINT_IDENTITY"; then
  previous_fingerprint="$("$CLIENT_PYTHON" - "$CHECKPOINT_IDENTITY" "$GATEWAY_MANIFEST_SHA256" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
expected = {
    'protocol': 'docred-kg-evaluation-checkpoint-v1',
    'dataset_repository': 'thunlp/docred',
    'dataset_revision': '7985b4e0371e6c61a756feb41b7b27becf71c666',
    'gateway_manifest_sha256': sys.argv[2],
    'llm_max_tokens': 4096,
    'extraction_max_tokens_ceiling': 8192,
    'concurrency': 1,
    'request_min_interval_s': 4.0,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f'DocRED checkpoint identity mismatch for {key}')
fingerprint = payload.get('llm_runtime_fingerprint')
if not isinstance(fingerprint, str) or not fingerprint.startswith('vertex-gateway:'):
    raise SystemExit('DocRED checkpoint has no valid LLM cache identity')
print(fingerprint)
PY
)"
  CACHE_IDENTITY_ARGS=(--llm-runtime-fingerprint-override "$previous_fingerprint")
fi

"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$RUNTIME_CONFIG" --data-dir "$DOCRED_DATA_DIR" --work-dir "$RUN_ROOT" \
  --cache-root "$CHECKPOINT_ROOT" --max-tokens 4096 --extraction-max-tokens-ceiling 8192 \
  --concurrency 1 --max-retries 0 --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  "${CACHE_IDENTITY_ARGS[@]}" > "$RUNTIME_IDENTITY"
# KGGen's native multi-chunk executor is deliberately serial here. This keeps
# one gateway request at a time, makes inner progress observable, and preserves
# the standard KGGen aggregate-then-cluster protocol.
"$CLIENT_PYTHON" - "$RUNTIME_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding='utf-8'))
config['extraction']['serial_chunking'] = True
config['extraction']['explicit_clustering'] = True
config['extraction']['cluster_context_mode'] = 'source_text'
config['extraction']['max_tokens_ceiling'] = 8192
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
PY

"$CLIENT_PYTHON" - "$CHECKPOINT_IDENTITY" "$RUNTIME_IDENTITY" "$GATEWAY_MANIFEST_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
runtime = json.load(open(sys.argv[2], encoding='utf-8'))
payload = {
    'protocol': 'docred-kg-evaluation-checkpoint-v1',
    'dataset_repository': 'thunlp/docred',
    'dataset_revision': '7985b4e0371e6c61a756feb41b7b27becf71c666',
    'gateway_manifest_sha256': sys.argv[3],
    'llm_max_tokens': 4096,
    'extraction_max_tokens_ceiling': 8192,
    'concurrency': 1,
    'request_min_interval_s': 4.0,
    'llm_runtime_fingerprint': runtime['runtime_fingerprint'],
}
if path.exists():
    existing = json.loads(path.read_text(encoding='utf-8'))
    if existing != payload:
        raise SystemExit('DocRED checkpoint identity changed; refusing incompatible cache reuse')
else:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
PY

cp "$CHECKPOINT_IDENTITY" "$RUN_ROOT/checkpoint-identity.json"
cp "$DOCRED_DATA_DIR/dataset-metadata.json" "$RUN_ROOT/dataset-metadata.json"

set +e
"$CLIENT_PYTHON" "$ROOT/scripts/run_docred_kg_eval.py" \
  --config "$RUNTIME_CONFIG" --data-dir "$DOCRED_DATA_DIR" --output-dir "$LIVE_OUT" \
  --manifest "$PERSISTENT_MANIFEST" --stage all --budget-eur "$DOCRED_BUDGET_EUR"
live_status=$?
set -e
cp "$PERSISTENT_MANIFEST" "$RUN_ROOT/docred_manifest.json" 2>/dev/null || true
if test "$live_status" -eq 75; then
  echo "[docred] budget guard stopped before another live document" >&2
  exit 75
fi
test "$live_status" -eq 0 || exit "$live_status"

"$CLIENT_PYTHON" - "$LIVE_OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = json.loads((root / 'run_metadata.json').read_text(encoding='utf-8'))
metrics = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
if metadata.get('state') != 'completed':
    raise SystemExit('live DocRED stage did not complete')
if metrics.get('documents') != 200 or metrics.get('extraction_coverage') != 1.0:
    raise SystemExit('live DocRED extraction coverage is incomplete')
PY

inventory() {
  "$CLIENT_PYTHON" - "$CHECKPOINT_ROOT" "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(path for path in root.rglob('*') if path.is_file())
aggregate = hashlib.sha256()
total_bytes = 0
for path in files:
    # Path names can be content-addressed cache keys. Feed them to the local
    # aggregate but never serialise them in the archive report.
    aggregate.update(path.relative_to(root).as_posix().encode('utf-8'))
    aggregate.update(path.read_bytes())
    total_bytes += path.stat().st_size
Path(sys.argv[2]).write_text(json.dumps({
    'protocol': 'docred-cache-inventory-v1',
    'files': len(files),
    'bytes': total_bytes,
    'aggregate_sha256': aggregate.hexdigest(),
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}
inventory "$RUN_ROOT/cache-before-replay.json"
"$CLIENT_PYTHON" "$ROOT/scripts/run_docred_kg_eval.py" \
  --config "$RUNTIME_CONFIG" --data-dir "$DOCRED_DATA_DIR" --output-dir "$REPLAY_OUT" \
  --manifest "$PERSISTENT_MANIFEST" --stage replay --cache-only --budget-eur "$DOCRED_BUDGET_EUR"
"$CLIENT_PYTHON" "$ROOT/scripts/verify_docred_cache_replay.py" \
  --live-dir "$LIVE_OUT" --replay-dir "$REPLAY_OUT"
inventory "$RUN_ROOT/cache-after-replay.json"
cmp "$RUN_ROOT/cache-before-replay.json" "$RUN_ROOT/cache-after-replay.json"

"$CLIENT_PYTHON" - "$LIVE_OUT/usage.jsonl" "$REPLAY_OUT/usage.jsonl" "$USAGE_COUNTS" <<'PY'
import json
import sys
from pathlib import Path

def summary(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    final = rows[-1] if rows else {}
    return {
        'api_calls': int(final.get('cum_calls', 0)),
        'requests_total': int(final.get('cum_requests', 0)),
        'cache_hits': int(final.get('cum_cache_hits', 0)),
        'retries': int(final.get('cum_retries', 0)),
        'prompt_tokens': int(final.get('cum_prompt_tokens', 0)),
        'completion_tokens': int(final.get('cum_completion_tokens', 0)),
    }

payload = {'protocol': 'docred-usage-v1', 'live': summary(sys.argv[1]), 'replay': summary(sys.argv[2])}
if payload['replay']['api_calls'] != 0:
    raise SystemExit('DocRED cache-only replay made live inference calls')
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

"$CLIENT_PYTHON" - "$METADATA" "$EXPECTED_SOURCE_COMMIT" "$DATASPHERE_DOCKER_IMAGE_ID" "$GATEWAY_MANIFEST" "$DOCRED_BUDGET_EUR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from gateway.core import canonical_manifest_sha256

root = Path(sys.argv[1]).parent
live = json.loads((root / 'docred-live' / 'metrics.json').read_text(encoding='utf-8'))
usage = json.loads((root / 'usage-counts.json').read_text(encoding='utf-8'))
manifest = json.loads(Path(sys.argv[4]).read_text(encoding='utf-8'))
Path(sys.argv[1]).write_text(json.dumps({
    'state': 'completed',
    'mode': 'cpu-vertex-docred-kg-evaluation',
    'checked_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'source_commit': sys.argv[2],
    'datasphere_docker_image_id': sys.argv[3],
    'gateway_manifest_sha256': canonical_manifest_sha256(manifest),
    'fixed_protocol': {
        'calibration_documents': 50,
        'smoke_documents_within_calibration': 10,
        'heldout_dev_documents': 200,
        'seed': 42,
        'concurrency': 1,
        'base_max_tokens': 4096,
        'adaptive_max_tokens_ceiling': 8192,
        'budget_max_eur': float(sys.argv[5]),
    },
    'metrics': live,
    'usage': usage,
    'cache_only_replay': {'api_calls': usage['replay']['api_calls']},
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
