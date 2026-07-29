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
QA_FORCE_CACHE_ONLY="${QA_FORCE_CACHE_ONLY:-0}"
# A DataSphere Job must not spend days repeating one 408/429.  This is a
# per-request circuit breaker (not a Job watchdog): completed entries remain
# atomic on project disk, and a retry can resume them.  Twelve 90-second
# request windows plus backoff is intentionally generous for transient quota
# recovery while still making a stalled request diagnosable within minutes.
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-12}"
QA_EXCLUDE_SOURCE_IDS="${QA_EXCLUDE_SOURCE_IDS:-}"

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
CRITICAL_GATEWAY_PROBE="$RUN_ROOT/support-critical-gateway-probe.json"
CRITICAL_RESILIENCE_PREFLIGHT="$RUN_ROOT/support-critical-resilience-preflight.json"
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
[[ "$LLM_MAX_RETRIES" =~ ^[0-9]+$ ]] && (( LLM_MAX_RETRIES >= 1 )) || {
  echo "LLM_MAX_RETRIES must be a positive integer" >&2
  exit 2
}
[[ "$QA_FORCE_CACHE_ONLY" == "0" || "$QA_FORCE_CACHE_ONLY" == "1" ]] || {
  echo "QA_FORCE_CACHE_ONLY must be 0 or 1" >&2
  exit 2
}
EXCLUDE_SOURCE_ARGS=()
if [[ -n "$QA_EXCLUDE_SOURCE_IDS" ]]; then
  IFS=',' read -r -a EXCLUDED_SOURCE_IDS <<< "$QA_EXCLUDE_SOURCE_IDS"
  declare -A SEEN_EXCLUDED_SOURCE_IDS=()
  for source_id in "${EXCLUDED_SOURCE_IDS[@]}"; do
    [[ "$source_id" =~ ^[A-Za-z0-9._:-]+$ ]] || {
      echo "QA_EXCLUDE_SOURCE_IDS contains an invalid source ID" >&2
      exit 2
    }
    [[ -z "${SEEN_EXCLUDED_SOURCE_IDS[$source_id]+x}" ]] || {
      echo "QA_EXCLUDE_SOURCE_IDS contains duplicate source ID: $source_id" >&2
      exit 2
    }
    SEEN_EXCLUDED_SOURCE_IDS[$source_id]=1
    EXCLUDE_SOURCE_ARGS+=(--exclude-source-id "$source_id")
  done
fi
echo "[qa-sample] total=$QA_SAMPLE_SIZE train=$QA_TRAIN_SOURCES test=$QA_TEST_SOURCES cv_folds=$QA_CV_FOLDS max_retries=$LLM_MAX_RETRIES exclusions=${QA_EXCLUDE_SOURCE_IDS:-none}"
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
# Every sample gets a writable primary cache namespace.  Compatible historical
# namespaces are read-only inputs, so an initial larger manifest fills only its
# missing content-addressed entries and a later replay can run fully offline.
CHECKPOINT_PARENT="$DS_PROJECT_HOME/hallu_smiles/checkpoints/vertex-qa"
CHECKPOINT_BASE="$CHECKPOINT_PARENT/qa-${QA_SAMPLE_SIZE}-test-${QA_TEST_SOURCES}-cv-${QA_CV_FOLDS}"
BASELINE_PROTOCOL_NAMESPACE="baseline-v1-${GATEWAY_MANIFEST_SHA256}"
BASELINE_CACHE_ROOT="$CHECKPOINT_BASE/$BASELINE_PROTOCOL_NAMESPACE"
mkdir -p "$BASELINE_CACHE_ROOT/kg" "$BASELINE_CACHE_ROOT/verdicts"

# Resolve a historical graph/verdict lineage independently of its manifest
# size. Cache keys are content-addressed and include the validated gateway and
# recorded runtime fingerprint, so the 100-QA lineage is safe read-through
# input for a 1000-QA primary namespace without being modified.
mapfile -t BASELINE_CANDIDATES < <(
  find "$CHECKPOINT_PARENT" -mindepth 2 -maxdepth 2 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort
)
VALID_BASELINE_CANDIDATES=()
for candidate in "${BASELINE_CANDIDATES[@]}"; do
  test -d "$candidate/kg" && test -d "$candidate/verdicts" && test -f "$candidate/checkpoint-identity.json" || continue
  read -r candidate_total candidate_train candidate_test candidate_folds < <("$CLIENT_PYTHON" - "$candidate/checkpoint-identity.json" <<'PY'
import json
import sys

sample = json.load(open(sys.argv[1], encoding='utf-8')).get('qa_sample', {})
try:
    print(int(sample['total']), int(sample['train']), int(sample['test']), int(sample['alpha_cv_folds']))
except (KeyError, TypeError, ValueError):
    raise SystemExit(2)
PY
)
  if "$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
    --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
    --checkpoint-identity "$candidate/checkpoint-identity.json" \
    --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
    --qa-total "$candidate_total" --qa-train "$candidate_train" --qa-test "$candidate_test" --cv-folds "$candidate_folds" \
    --output "$RUN_ROOT/.candidate-historical-lineage.json" >/dev/null 2>&1; then
    VALID_BASELINE_CANDIDATES+=("$candidate")
  fi
done
if (( ${#VALID_BASELINE_CANDIDATES[@]} != 1 )); then
  echo "Expected exactly one compatible historical QA checkpoint; found ${#VALID_BASELINE_CANDIDATES[@]}." >&2
  printf 'candidates checked: %s\\n' "${BASELINE_CANDIDATES[*]:-none}" >&2
  exit 2
fi
HISTORICAL_BASELINE_CACHE_ROOT="${VALID_BASELINE_CANDIDATES[0]}"
read -r HISTORICAL_QA_TOTAL HISTORICAL_QA_TRAIN HISTORICAL_QA_TEST HISTORICAL_QA_CV_FOLDS < <("$CLIENT_PYTHON" - "$HISTORICAL_BASELINE_CACHE_ROOT/checkpoint-identity.json" <<'PY'
import json
import sys

sample = json.load(open(sys.argv[1], encoding='utf-8'))['qa_sample']
print(int(sample['total']), int(sample['train']), int(sample['test']), int(sample['alpha_cv_folds']))
PY
)
"$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
  --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
  --checkpoint-identity "$HISTORICAL_BASELINE_CACHE_ROOT/checkpoint-identity.json" \
  --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
  --qa-total "$HISTORICAL_QA_TOTAL" --qa-train "$HISTORICAL_QA_TRAIN" --qa-test "$HISTORICAL_QA_TEST" --cv-folds "$HISTORICAL_QA_CV_FOLDS" \
  --output "$HISTORICAL_LINEAGE"
HISTORICAL_LLM_RUNTIME_FINGERPRINT="$("$CLIENT_PYTHON" - "$HISTORICAL_LINEAGE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['llm_runtime_fingerprint'])
PY
)"
# Critical artifacts are independent of source commit when their component
# protocol, prompt version, LLM identity, and evidence key are unchanged. A
# stable namespace therefore resumes a partial Job instead of discarding every
# completed verdict after a retry-only code change. Older commit namespaces are
# read-through inputs and remain untouched.
CRITICAL_PROTOCOL_NAMESPACE="support-critical-v1-${GATEWAY_MANIFEST_SHA256}"
CRITICAL_CACHE_ROOT="$CHECKPOINT_BASE/support-critical/$CRITICAL_PROTOCOL_NAMESPACE"
CRITICAL_CACHE_READ_ARGS=()
if test -d "$CHECKPOINT_BASE/support-critical"; then
  while IFS= read -r previous_critical_root; do
    test "$previous_critical_root" = "$CRITICAL_CACHE_ROOT" && continue
    CRITICAL_CACHE_READ_ARGS+=(--critical-cache-read-root "$previous_critical_root")
  done < <(find "$CHECKPOINT_BASE/support-critical" -mindepth 1 -maxdepth 1 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort)
fi
HISTORICAL_SAMPLE_ROOT="$(dirname "$HISTORICAL_BASELINE_CACHE_ROOT")"
if test -d "$HISTORICAL_SAMPLE_ROOT/support-critical"; then
  while IFS= read -r previous_critical_root; do
    CRITICAL_CACHE_READ_ARGS+=(--critical-cache-read-root "$previous_critical_root")
  done < <(find "$HISTORICAL_SAMPLE_ROOT/support-critical" -mindepth 1 -maxdepth 1 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort)
fi
CHECKPOINT_ROOT="$CRITICAL_CACHE_ROOT"
mkdir -p "$CRITICAL_CACHE_ROOT/kg" "$CRITICAL_CACHE_ROOT/critical_claims" \
  "$CRITICAL_CACHE_ROOT/critical_coverage" "$CRITICAL_CACHE_ROOT/critical_verdicts"
"$CLIENT_PYTHON" - "$CRITICAL_CACHE_ROOT/checkpoint-identity.json" "$EXPECTED_SOURCE_COMMIT" "$GATEWAY_MANIFEST_SHA256" "$QA_SAMPLE_SIZE" "$QA_TRAIN_SOURCES" "$QA_TEST_SOURCES" "$QA_CV_FOLDS" "$HISTORICAL_BASELINE_CACHE_ROOT" "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
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
    'critical_protocol_namespace': 'support-critical-v1',
    'protocol': 'hallu-vertex-qa-support-critical-checkpoint-v1',
}
if path.exists():
    existing = json.loads(path.read_text(encoding='utf-8'))
    for key in ('gateway_manifest_sha256', 'qa_sample', 'historical_baseline_cache_root', 'historical_llm_runtime_fingerprint', 'critical_protocol_namespace', 'protocol'):
        if existing.get(key) != payload[key]:
            raise SystemExit(f'critical checkpoint identity mismatch for {key}')
else:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)
PY

"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$BASELINE_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$BASELINE_CACHE_ROOT" \
  --kg-cache-read-dir "$HISTORICAL_BASELINE_CACHE_ROOT/kg" \
  --relation-cache-read-dir "$HISTORICAL_BASELINE_CACHE_ROOT/verdicts" \
  --llm-runtime-fingerprint-override "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" \
  --max-tokens 16384 --concurrency "$LLM_CONCURRENCY" --max-retries "$LLM_MAX_RETRIES" --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --cv-folds "$QA_CV_FOLDS" \
  > "$RUN_ROOT/baseline-runtime-config-identity.json"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$HALLU_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$CRITICAL_CONFIG" --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$CRITICAL_CACHE_ROOT" \
  --kg-cache-read-dir "$BASELINE_CACHE_ROOT/kg" \
  --kg-cache-read-dir "$HISTORICAL_BASELINE_CACHE_ROOT/kg" \
  --llm-runtime-fingerprint-override "$HISTORICAL_LLM_RUNTIME_FINGERPRINT" \
  "${CRITICAL_CACHE_READ_ARGS[@]}" \
  --max-tokens 16384 --concurrency "$LLM_CONCURRENCY" --max-retries "$LLM_MAX_RETRIES" --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --cv-folds "$QA_CV_FOLDS" \
  > "$RUN_ROOT/critical-runtime-config-identity.json"

# Generate the deterministic manifest before any live inference. The first
# larger run deliberately allows misses here: historical roots serve the
# existing content hits and the strict baseline below atomically fills only the
# remaining KG keys in this run's primary namespace.
"$CLIENT_PYTHON" "$ROOT/scripts/preflight_datasphere_kg_cache.py" \
  --config "$BASELINE_CONFIG" --data-dir "$DATA_DIR" \
  --qa-sample-size "$QA_SAMPLE_SIZE" --qa-test-fraction "$QA_TEST_FRACTION" \
  --manifest-output "$QA_MANIFEST" --report "$KG_CACHE_PREFLIGHT" --allow-missing \
  "${EXCLUDE_SOURCE_ARGS[@]}"

# Check every selected answer's deterministic segmentation and no-network
# fallback offsets before asking Vertex anything. This catches code/config
# regressions that otherwise surface only halfway through the scoring loop.
"$CLIENT_PYTHON" "$ROOT/scripts/preflight_support_critical_resilience.py" \
  --config "$CRITICAL_CONFIG" --data-dir "$DATA_DIR" --qa-manifest "$QA_MANIFEST" \
  --report "$CRITICAL_RESILIENCE_PREFLIGHT"

# Probe the exact three new schemas before baseline reports/tuning. It does
# not require a particular semantic answer, and its cache-only replay proves
# the critical namespace is writable and resumable before the 100-QA loop.
GATEWAY_PROBE_ARGS=()
BASELINE_RUN_ARGS=()
CRITICAL_RUN_ARGS=(--kg-cache-only)
if [[ "$QA_FORCE_CACHE_ONLY" == "1" ]]; then
  GATEWAY_PROBE_ARGS+=(--cache-only)
  BASELINE_RUN_ARGS+=(--cache-only)
  CRITICAL_RUN_ARGS=(--cache-only)
  echo "[cache-only] force mode enabled: any missing inference entry is a hard failure"
fi
"$CLIENT_PYTHON" "$ROOT/scripts/check_support_critical_gateway_probe.py" \
  --config "$CRITICAL_CONFIG" --report "$CRITICAL_GATEWAY_PROBE" "${GATEWAY_PROBE_ARGS[@]}"

require_complete_extraction() {
  "$CLIENT_PYTHON" - "$1/extraction_summary.json" "$2" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding='utf-8'))
expected = int(sys.argv[2])
excluded = list(summary.get('excluded_records', []))
excluded_sources = list(summary.get('excluded_source_ids', []))
if summary.get('status') not in {'ready', 'ready_with_explicit_exclusions'} or summary.get('failures') != []:
    raise SystemExit('extraction is incomplete')
if len(set(excluded_sources)) != len(excluded_sources):
    raise SystemExit('extraction exclusion source IDs are not unique')
if len(excluded) != len(excluded_sources):
    raise SystemExit('extraction exclusion records do not match source exclusions')
analysis_expected = expected - len(excluded_sources)
if summary.get('expected_sources') != expected or summary.get('analysis_expected_sources') != analysis_expected:
    raise SystemExit('extraction source denominator is inconsistent')
if summary.get('references_completed') != analysis_expected or summary.get('responses_completed') != analysis_expected:
    raise SystemExit('extraction did not produce every non-quarantined source/response graph')
if summary.get('analysis_expected_records') != summary.get('completed_records'):
    raise SystemExit('extraction summary records are incomplete')
PY
  test ! -s "$1/failed_extractions.jsonl" || {
    echo "failed_extractions.jsonl is non-empty: $1" >&2
    exit 1
  }
}

# Fill the primary KG cache read-through from the validated historical root.
# Strict has no text-verifier calls, so this stage can make only the required
# KGGen requests for cold content and preserves the historical root unchanged.
find "$HISTORICAL_BASELINE_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/historical-cache-before.sha256"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode strict --qa-manifest "$QA_MANIFEST" \
  --output-dir "$STRICT_OUT" "${BASELINE_RUN_ARGS[@]}" "${EXCLUDE_SOURCE_ARGS[@]}"
require_complete_extraction "$STRICT_OUT" "$QA_SAMPLE_SIZE"
mv "$STRICT_OUT/usage.jsonl" "$RUN_ROOT/strict-cache-fill-usage.jsonl"

# Historical support verdicts are also read-through. New text-verdict entries
# are written only to the primary 1000-QA namespace.
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode support --qa-manifest "$QA_MANIFEST" \
  --output-dir "$SUPPORT_OUT" "${BASELINE_RUN_ARGS[@]}" "${EXCLUDE_SOURCE_ARGS[@]}"
require_complete_extraction "$SUPPORT_OUT" "$QA_SAMPLE_SIZE"
mv "$SUPPORT_OUT/usage.jsonl" "$RUN_ROOT/support-cache-fill-usage.jsonl"
find "$HISTORICAL_BASELINE_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/historical-cache-after-baseline.sha256"
cmp "$RUN_ROOT/historical-cache-before.sha256" "$RUN_ROOT/historical-cache-after-baseline.sha256"

# KGGen is now fully warm in the primary/read-through baseline cache. This
# stage permits only the support-critical claim, coverage, and four-way
# evidence components to make live gateway calls.
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$CRITICAL_CONFIG" --stage all \
  --relation-mode support-critical --qa-manifest "$QA_MANIFEST" \
  --output-dir "$CRITICAL_OUT" "${CRITICAL_RUN_ARGS[@]}" "${EXCLUDE_SOURCE_ARGS[@]}"
require_complete_extraction "$CRITICAL_OUT" "$QA_SAMPLE_SIZE"
mv "$CRITICAL_OUT/usage.jsonl" "$RUN_ROOT/support-critical-live-usage.jsonl"
"$CLIENT_PYTHON" "$ROOT/scripts/compare_qa_pilot_results.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --critical-dir "$CRITICAL_OUT" --output "$RUN_ROOT/comparison.json"
"$CLIENT_PYTHON" "$ROOT/scripts/summarize_support_critical_diagnostics.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --critical-dir "$CRITICAL_OUT" --output "$RUN_ROOT/support-critical-diagnostic.json"

# Every mutable namespace must be byte-stable during the mandatory cache-only
# replay: KG, support-verdicts, and all three critical components.
find "$BASELINE_CACHE_ROOT" "$CRITICAL_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-before-replay.sha256"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode strict --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_STRICT" --cache-only "${EXCLUDE_SOURCE_ARGS[@]}"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$BASELINE_CONFIG" --stage all \
  --relation-mode support --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_SUPPORT" --cache-only "${EXCLUDE_SOURCE_ARGS[@]}"
"$CLIENT_PYTHON" "$ROOT/run.py" --config "$CRITICAL_CONFIG" --stage all \
  --relation-mode support-critical --qa-manifest "$QA_MANIFEST" \
  --output-dir "$REPLAY_CRITICAL" --cache-only "${EXCLUDE_SOURCE_ARGS[@]}"
require_complete_extraction "$REPLAY_STRICT" "$QA_SAMPLE_SIZE"
require_complete_extraction "$REPLAY_SUPPORT" "$QA_SAMPLE_SIZE"
require_complete_extraction "$REPLAY_CRITICAL" "$QA_SAMPLE_SIZE"
for name in strict support support-critical; do
  live_dir="$RUN_ROOT/$name"
  replay_dir="$RUN_ROOT/cache-replay/$name"
  # The verifier keeps metrics.csv, summary_metrics.csv, and tuning.json byte
  # identical; it compares scored.jsonl semantically while excluding only its
  # cache-observability flag.
  "$CLIENT_PYTHON" "$ROOT/scripts/verify_cache_replay.py" \
    --live-dir "$live_dir" --replay-dir "$replay_dir"
done
find "$BASELINE_CACHE_ROOT" "$CRITICAL_CACHE_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-after-replay.sha256"
cmp "$RUN_ROOT/cache-before-replay.sha256" "$RUN_ROOT/cache-after-replay.sha256"

"$CLIENT_PYTHON" - "$USAGE_COUNTS" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2])
def usage(name):
    path = root / name
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    final = rows[-1] if rows else {}
    components = {}
    retry_reasons = {}
    for row in rows:
        kind = str(row.get('kind', 'unknown'))
        if row.get('event') == 'retry':
            reason = str(row.get('retry_reason', 'unknown'))
            retry_reasons[reason] = retry_reasons.get(reason, 0) + 1
            continue
        item = components.setdefault(kind, {
            'requests': 0, 'live_calls': 0, 'cache_hits': 0,
            'seconds': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0,
        })
        item['requests'] += 1
        item['live_calls'] += 0 if bool(row.get('cached')) else 1
        item['cache_hits'] += int(bool(row.get('cached')))
        item['seconds'] += float(row.get('seconds', 0.0))
        item['prompt_tokens'] += int(row.get('prompt_tokens', 0))
        item['completion_tokens'] += int(row.get('completion_tokens', 0))
    return {
        'usage_file': str(path),
        'api_calls': int(final.get('cum_calls', 0)),
        'requests_total': int(final.get('cum_requests', 0)),
        'cache_hits': int(final.get('cum_cache_hits', 0)),
        'retries': int(final.get('cum_retries', 0)),
        'api_attempts': int(final.get('cum_calls', 0)) + int(final.get('cum_retries', 0)),
        'prompt_tokens': int(final.get('cum_prompt_tokens', 0)),
        'completion_tokens': int(final.get('cum_completion_tokens', 0)),
        'components': components,
        'retry_reasons': retry_reasons,
    }
payload = {
    'status': 'ready',
    'strict_cache_fill': usage('strict-cache-fill-usage.jsonl'),
    'support_cache_fill': usage('support-cache-fill-usage.jsonl'),
    'support_critical_live': usage('support-critical-live-usage.jsonl'),
    'strict_replay': usage('cache-replay/strict/usage.jsonl'),
    'support_replay': usage('cache-replay/support/usage.jsonl'),
    'support_critical_replay': usage('cache-replay/support-critical/usage.jsonl'),
}
if any(payload[name]['api_attempts'] for name in (
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
    'runs': ['strict-cache-fill', 'support-cache-fill', 'support-critical-live', 'cache-only-strict', 'cache-only-support', 'cache-only-support-critical'],
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
