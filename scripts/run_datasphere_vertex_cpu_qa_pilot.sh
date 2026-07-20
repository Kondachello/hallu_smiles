#!/usr/bin/env bash
# Parameterized strict/support/support-critical QA experiment: CPU -> Cloud Run -> Vertex AI.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/opt/hallu/models/all-MiniLM-L6-v2}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set the source commit in the Job template}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set the immutable Docker identity}"
EXPECTED_GATEWAY_MANIFEST_SHA256="${EXPECTED_GATEWAY_MANIFEST_SHA256:?Set the verified 3-QA gateway manifest hash}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to RAGTruth project storage}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the writable Job directory}"
HALLU_GATEWAY_URL="${HALLU_GATEWAY_URL:?Set the Cloud Run origin in the rendered Job}"
: "${HALLU_GATEWAY_API_KEY:?Create a DataSphere Project secret named HALLU_GATEWAY_API_KEY}"
QA_SAMPLE_SIZE="${QA_SAMPLE_SIZE:-20}"
QA_TEST_FRACTION="${QA_TEST_FRACTION:-0.2}"
QA_CV_FOLDS="${QA_CV_FOLDS:-5}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-1}"

BASELINE_CONFIG="$RUN_ROOT/baseline_runtime_config.yaml"
CRITICAL_CONFIG="$RUN_ROOT/critical_runtime_config.yaml"
GATEWAY_MANIFEST_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
QA_MANIFEST="$RUN_ROOT/qa_manifest.json"
STRICT_OUT="$RUN_ROOT/strict"
SUPPORT_OUT="$RUN_ROOT/support"
CRITICAL_OUT="$RUN_ROOT/support-critical"
REPLAY_STRICT="$RUN_ROOT/cache-replay/strict"
REPLAY_SUPPORT="$RUN_ROOT/cache-replay/support"
REPLAY_CRITICAL="$RUN_ROOT/cache-replay/support-critical"
METADATA="$RUN_ROOT/run_metadata.json"
USAGE_COUNTS="$RUN_ROOT/usage-counts.json"
HISTORICAL_LINEAGE="$RUN_ROOT/historical-baseline-lineage.json"
KG_CACHE_PREFLIGHT="$RUN_ROOT/historical-kg-cache-preflight.json"
CHECKPOINT_ROOT=""
BASELINE_CACHE_ROOT=""
CRITICAL_CACHE_ROOT=""
export PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/hallu/models}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
  status=$?
  trap - EXIT
  # The project disk survives a Job error. Snapshot the same namespace into
  # the archive too, so a failed attempt is inspectable and a rerun of this
  # exact commit/gateway pair resumes only the cache misses.
  if [[ -n "$CHECKPOINT_ROOT" && -d "$CHECKPOINT_ROOT" ]]; then
    mkdir -p "$RUN_ROOT/cache" || true
    cp -a "$CHECKPOINT_ROOT/." "$RUN_ROOT/cache/" || true
  fi
  unset HALLU_GATEWAY_API_KEY
  exit "$status"
}
trap cleanup EXIT

[[ "$HALLU_GATEWAY_URL" =~ ^https://[^/]+$ ]] || { echo "HALLU_GATEWAY_URL must be an HTTPS origin." >&2; exit 2; }
test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
test -f "$RUNTIME_MANIFEST" || { echo "runtime manifest is missing: $RUNTIME_MANIFEST" >&2; exit 2; }
test -f "$EMBEDDING_MODEL_PATH/config.json" || { echo "offline S-BERT snapshot is missing." >&2; exit 2; }
mkdir -p "$RUN_ROOT"
QA_QUOTAS="$(
"$CLIENT_PYTHON" - "$QA_SAMPLE_SIZE" "$QA_TEST_FRACTION" <<'PY'
import sys
from src.sampling import qa_sample_quotas

train, test = qa_sample_quotas(int(sys.argv[1]), sys.argv[2])
print(train, test)
PY
)"
read -r QA_TRAIN_SOURCES QA_TEST_SOURCES <<< "$QA_QUOTAS"
[[ "$QA_CV_FOLDS" =~ ^[0-9]+$ ]] && (( QA_CV_FOLDS >= 2 )) || {
  echo "QA_CV_FOLDS must be an integer of at least 2" >&2
  exit 2
}
(( QA_CV_FOLDS <= QA_TRAIN_SOURCES )) || {
  echo "QA_CV_FOLDS cannot exceed the selected train source count" >&2
  exit 2
}
[[ "$LLM_CONCURRENCY" =~ ^[0-9]+$ ]] && (( LLM_CONCURRENCY >= 1 )) || {
  echo "LLM_CONCURRENCY must be a positive integer" >&2
  exit 2
}
echo "[qa-sample] total=$QA_SAMPLE_SIZE train=$QA_TRAIN_SOURCES test=$QA_TEST_SOURCES cv_folds=$QA_CV_FOLDS"
cp "$RUNTIME_MANIFEST" "$RUN_ROOT/runtime-manifest.json"
cp /opt/hallu/manifests/client.freeze.txt "$RUN_ROOT/client.freeze.txt"

"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vertex_cpu_runtime.py" \
  --python "$CLIENT_PYTHON" --runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-path "$EMBEDDING_MODEL_PATH" --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --report "$RUN_ROOT/cpu-runtime.json"

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
  echo "gateway manifest changed since the successful 3-QA gate" >&2
  exit 2
}
CHECKPOINT_BASE="$DS_PROJECT_HOME/hallu_smiles/checkpoints/vertex-qa/qa-${QA_SAMPLE_SIZE}-test-${QA_TEST_SOURCES}-cv-${QA_CV_FOLDS}"
# Historical 100-QA caches predate this source commit. They are read-only input
# artifacts. A runtime-image rebuild must not accidentally turn an otherwise
# identical model/gateway/extraction cache into a miss, so resolve an explicit
# recorded lineage and prove every C/Q/A key before the main pipeline starts.
test -d "$CHECKPOINT_BASE" || {
  echo "Historical QA checkpoint base is absent: $CHECKPOINT_BASE" >&2
  exit 2
}
mapfile -t BASELINE_CANDIDATES < <(
  find "$CHECKPOINT_BASE" -mindepth 1 -maxdepth 1 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort
)
VALID_BASELINE_CANDIDATES=()
for candidate in "${BASELINE_CANDIDATES[@]}"; do
  test -d "$candidate/kg" && test -d "$candidate/verdicts" && test -f "$candidate/checkpoint-identity.json" || continue
  if "$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
    --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
    --checkpoint-identity "$candidate/checkpoint-identity.json" \
    --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
    --qa-total "$QA_SAMPLE_SIZE" --qa-train "$QA_TRAIN_SOURCES" --qa-test "$QA_TEST_SOURCES" --cv-folds "$QA_CV_FOLDS" \
    --output "$RUN_ROOT/.candidate-historical-lineage.json" >/dev/null 2>&1; then
    VALID_BASELINE_CANDIDATES+=("$candidate")
  fi
done
if (( ${#VALID_BASELINE_CANDIDATES[@]} != 1 )); then
  echo "Expected exactly one compatible historical QA checkpoint; found ${#VALID_BASELINE_CANDIDATES[@]}." >&2
  printf 'candidates checked: %s\\n' "${BASELINE_CANDIDATES[*]:-none}" >&2
  exit 2
fi
BASELINE_CACHE_ROOT="${VALID_BASELINE_CANDIDATES[0]}"
"$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
  --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
  --checkpoint-identity "$BASELINE_CACHE_ROOT/checkpoint-identity.json" \
  --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
  --qa-total "$QA_SAMPLE_SIZE" --qa-train "$QA_TRAIN_SOURCES" --qa-test "$QA_TEST_SOURCES" --cv-folds "$QA_CV_FOLDS" \
  --output "$HISTORICAL_LINEAGE"
HISTORICAL_LLM_RUNTIME_FINGERPRINT="$("$CLIENT_PYTHON" - "$HISTORICAL_LINEAGE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['llm_runtime_fingerprint'])
PY
)"
CRITICAL_CACHE_ROOT="$CHECKPOINT_BASE/support-critical/${EXPECTED_SOURCE_COMMIT}-${GATEWAY_MANIFEST_SHA256}"
CHECKPOINT_ROOT="$CRITICAL_CACHE_ROOT"
mkdir -p "$CRITICAL_CACHE_ROOT/kg" "$CRITICAL_CACHE_ROOT/critical_claims" \
  "$CRITICAL_CACHE_ROOT/critical_coverage" "$CRITICAL_CACHE_ROOT/critical_verdicts"
"$CLIENT_PYTHON" - "$CRITICAL_CACHE_ROOT/checkpoint-identity.json" "$EXPECTED_SOURCE_COMMIT" "$GATEWAY_MANIFEST_SHA256" "$QA_SAMPLE_SIZE" "$QA_TRAIN_SOURCES" "$QA_TEST_SOURCES" "$QA_CV_FOLDS" "$BASELINE_CACHE_ROOT" "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'source_commit': sys.argv[2],
    'gateway_manifest_sha256': sys.argv[3],
    'qa_sample': {
        'total': int(sys.argv[4]),
        'train': int(sys.argv[5]),
        'test': int(sys.argv[6]),
        'alpha_cv_folds': int(sys.argv[7]),
    },
    'historical_baseline_cache_root': sys.argv[8],
    'historical_llm_runtime_fingerprint': sys.argv[9],
    'protocol': 'hallu-vertex-qa-support-critical-checkpoint-v1',
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$BASELINE_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$BASELINE_CACHE_ROOT" \
  --llm-runtime-fingerprint-override "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" \
  --max-tokens 16384 --concurrency "$LLM_CONCURRENCY" --max-retries 1000 --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --cv-folds "$QA_CV_FOLDS" \
  > "$RUN_ROOT/baseline-runtime-config-identity.json"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$CRITICAL_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$CRITICAL_CACHE_ROOT" \
  --kg-cache-read-dir "$BASELINE_CACHE_ROOT/kg" \
  --llm-runtime-fingerprint-override "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" \
  --max-tokens 16384 --concurrency "$LLM_CONCURRENCY" --max-retries 1000 --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --cv-folds "$QA_CV_FOLDS" \
  > "$RUN_ROOT/critical-runtime-config-identity.json"

# Generate the same deterministic manifest as the original 100-QA run, then
# validate every graph lookup serially. A bad lineage now fails here—before
# threaded extraction, metric tuning, or any new Vertex claim-verifier call.
"$CLIENT_PYTHON" "$ROOT/scripts/preflight_datasphere_kg_cache.py" \
  --config "$BASELINE_CONFIG" --data-dir "$DATA_DIR" \
  --qa-sample-size "$QA_SAMPLE_SIZE" --qa-test-fraction "$QA_TEST_FRACTION" \
  --manifest-output "$QA_MANIFEST" --report "$KG_CACHE_PREFLIGHT"

require_complete_extraction() {
  "$CLIENT_PYTHON" - "$1/extraction_summary.json" "$2" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding='utf-8'))
expected = int(sys.argv[2])
if summary.get('status') != 'ready' or summary.get('failures') != []:
    raise SystemExit('extraction is incomplete')
if summary.get('expected_sources') != expected or summary.get('responses_completed') != expected:
    raise SystemExit('extraction did not produce every source/response graph')
if summary.get('expected_records') != summary.get('completed_records'):
    raise SystemExit('extraction summary records are incomplete')
PY
  test ! -s "$1/failed_extractions.jsonl" || {
    echo "failed_extractions.jsonl is non-empty: $1" >&2
    exit 1
  }
}

# The current manifest must resolve exclusively through the historical graph
# cache.  A cache miss fails immediately rather than silently re-extracting.
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage extract \
  --relation-mode strict --qa-manifest "$QA_MANIFEST" \
  --output-dir "$STRICT_OUT" --cache-only
require_complete_extraction "$STRICT_OUT" "$QA_SAMPLE_SIZE"
mv "$STRICT_OUT/usage.jsonl" "$RUN_ROOT/strict-extraction-cache-usage.jsonl"

"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode strict --qa-manifest "$QA_MANIFEST" \
  --output-dir "$STRICT_OUT" --cache-only
require_complete_extraction "$STRICT_OUT" "$QA_SAMPLE_SIZE"
mv "$STRICT_OUT/usage.jsonl" "$RUN_ROOT/strict-baseline-cache-usage.jsonl"

find "$BASELINE_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/historical-cache-before.sha256"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode support --qa-manifest "$QA_MANIFEST" \
  --output-dir "$SUPPORT_OUT" --cache-only
require_complete_extraction "$SUPPORT_OUT" "$QA_SAMPLE_SIZE"
mv "$SUPPORT_OUT/usage.jsonl" "$RUN_ROOT/support-baseline-cache-usage.jsonl"
find "$BASELINE_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/historical-cache-after-baseline.sha256"
cmp "$RUN_ROOT/historical-cache-before.sha256" "$RUN_ROOT/historical-cache-after-baseline.sha256"

# Only this stage is allowed to make new Vertex calls: claims, adversarial
# coverage, and four-way evidence verdicts.  KGGen remains cache-only.
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$CRITICAL_CONFIG" --stage all \
  --relation-mode support-critical --qa-manifest "$QA_MANIFEST" \
  --output-dir "$CRITICAL_OUT" --kg-cache-only
require_complete_extraction "$CRITICAL_OUT" "$QA_SAMPLE_SIZE"
mv "$CRITICAL_OUT/usage.jsonl" "$RUN_ROOT/support-critical-live-usage.jsonl"
"$CLIENT_PYTHON" "$ROOT/scripts/compare_qa_pilot_results.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --critical-dir "$CRITICAL_OUT" --output "$RUN_ROOT/comparison.json"
"$CLIENT_PYTHON" "$ROOT/scripts/summarize_support_critical_diagnostics.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --critical-dir "$CRITICAL_OUT" --output "$RUN_ROOT/support-critical-diagnostic.json"

find "$CRITICAL_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/critical-cache-before-replay.sha256"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode strict --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_STRICT" --cache-only
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode support --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_SUPPORT" --cache-only
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$CRITICAL_CONFIG" --stage all \
  --relation-mode support-critical --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_CRITICAL" --cache-only
require_complete_extraction "$REPLAY_STRICT" "$QA_SAMPLE_SIZE"
require_complete_extraction "$REPLAY_SUPPORT" "$QA_SAMPLE_SIZE"
require_complete_extraction "$REPLAY_CRITICAL" "$QA_SAMPLE_SIZE"
cmp "$STRICT_OUT/metrics.csv" "$REPLAY_STRICT/metrics.csv"
cmp "$SUPPORT_OUT/metrics.csv" "$REPLAY_SUPPORT/metrics.csv"
cmp "$CRITICAL_OUT/metrics.csv" "$REPLAY_CRITICAL/metrics.csv"
find "$CRITICAL_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/critical-cache-after-replay.sha256"
cmp "$RUN_ROOT/critical-cache-before-replay.sha256" "$RUN_ROOT/critical-cache-after-replay.sha256"

"$CLIENT_PYTHON" - "$USAGE_COUNTS" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2])
def usage(name):
    path = root / name
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    final = rows[-1] if rows else {}
    return {
        'usage_file': str(path),
        'api_calls': int(final.get('cum_calls', 0)),
        'requests_total': int(final.get('cum_requests', 0)),
        'cache_hits': int(final.get('cum_cache_hits', 0)),
        'prompt_tokens': int(final.get('cum_prompt_tokens', 0)),
        'completion_tokens': int(final.get('cum_completion_tokens', 0)),
    }
payload = {
    'status': 'ready',
    'strict_extraction_cache_only': usage('strict-extraction-cache-usage.jsonl'),
    'strict_baseline_cache_only': usage('strict-baseline-cache-usage.jsonl'),
    'support_baseline_cache_only': usage('support-baseline-cache-usage.jsonl'),
    'support_critical_live': usage('support-critical-live-usage.jsonl'),
    'strict_replay': usage('cache-replay/strict/usage.jsonl'),
    'support_replay': usage('cache-replay/support/usage.jsonl'),
    'support_critical_replay': usage('cache-replay/support-critical/usage.jsonl'),
}
if any(payload[name]['api_calls'] for name in (
    'strict_extraction_cache_only', 'strict_baseline_cache_only', 'support_baseline_cache_only',
    'strict_replay', 'support_replay', 'support_critical_replay',
)):
    raise SystemExit('cache-only metric replay made live inference calls')
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

"$CLIENT_PYTHON" - "$METADATA" "$EXPECTED_SOURCE_COMMIT" "$DATASPHERE_DOCKER_IMAGE_ID" "$GATEWAY_MANIFEST" "$QA_SAMPLE_SIZE" "$QA_TRAIN_SOURCES" "$QA_TEST_SOURCES" "$QA_CV_FOLDS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from gateway.core import canonical_manifest_sha256

manifest = json.loads(Path(sys.argv[4]).read_text(encoding='utf-8'))
Path(sys.argv[1]).write_text(json.dumps({
    'state': 'completed',
    'mode': 'cpu-vertex-qa-strict-support-critical',
    'checked_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'source_commit': sys.argv[2],
    'datasphere_docker_image_id': sys.argv[3],
    'gateway_manifest_sha256': canonical_manifest_sha256(manifest),
    'gateway_manifest': manifest,
    'qa_sample': {
        'total': int(sys.argv[5]),
        'train': int(sys.argv[6]),
        'test': int(sys.argv[7]),
        'alpha_cv_folds': int(sys.argv[8]),
    },
    'runs': ['strict-baseline-cache-only', 'support-baseline-cache-only', 'support-critical-live', 'cache-only-strict', 'cache-only-support', 'cache-only-support-critical'],
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
