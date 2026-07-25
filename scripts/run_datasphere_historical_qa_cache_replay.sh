#!/usr/bin/env bash
# Replay reproducibly selected fully warm historical QA graph sets. The only network request is the
# authenticated gateway manifest: its pinned identity reconstructs the old cache key.
# No request is ever made to a language model.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
: "${RUN_ROOT:?}" "${DATA_DIR:?}" "${CHECKPOINT_PARENT:?}" "${RECORDED_GATEWAY_URL:?}" "${EXPECTED_SOURCE_COMMIT:?}"
: "${HALLU_GATEWAY_API_KEY:?Create DataSphere Project secret HALLU_GATEWAY_API_KEY}"
QA_SAMPLE_SIZE="${QA_SAMPLE_SIZE:-100}"
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

# --- Authenticated gateway manifest (retried on transient errors) ---------
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
GATEWAY_MANIFEST_SHA256="$("$CLIENT_PYTHON" - "$MANIFEST" <<'PY'
import json, sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding='utf-8'))))
PY
)"

# --- Discover the historical checkpoint matching this gateway manifest and
# the requested QA_SAMPLE_SIZE. Checkpoints of different sizes (100, 750, ...)
# can share one CHECKPOINT_PARENT and even one gateway manifest, so QA_SAMPLE_SIZE
# disambiguates rather than a hardcoded single checkpoint path. Each candidate's
# own checkpoint-identity.json supplies its train/test/cv-fold shape -- nothing
# beyond QA_SAMPLE_SIZE needs to be known in advance. --------------------------
HISTORICAL_LINEAGE="$RUN_ROOT/historical-lineage.json"
DISCOVERY_REPORT="$RUN_ROOT/reports/historical_cache_discovery.json"
mkdir -p "$(dirname "$DISCOVERY_REPORT")"
test -d "$CHECKPOINT_PARENT" || {
  echo "Historical checkpoint parent is absent: $CHECKPOINT_PARENT" >&2
  exit 2
}
mapfile -t CANDIDATES < <(
  find "$CHECKPOINT_PARENT" -mindepth 2 -maxdepth 2 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort
)
VALID_CANDIDATES=()
CANDIDATES_REPORT="[]"
for candidate in "${CANDIDATES[@]}"; do
  test -d "$candidate/kg" && test -f "$candidate/checkpoint-identity.json" || continue
  read -r candidate_total candidate_train candidate_test candidate_folds < <("$CLIENT_PYTHON" - "$candidate/checkpoint-identity.json" <<'PY'
import json, sys
sample = json.load(open(sys.argv[1], encoding='utf-8')).get('qa_sample', {})
try:
    print(int(sample['total']), int(sample['train']), int(sample['test']), int(sample['alpha_cv_folds']))
except (KeyError, TypeError, ValueError):
    raise SystemExit(2)
PY
) || continue
  status="skipped_wrong_total"
  if [[ "$candidate_total" == "$QA_SAMPLE_SIZE" ]]; then
    if "$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
      --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
      --checkpoint-identity "$candidate/checkpoint-identity.json" \
      --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
      --qa-total "$candidate_total" --qa-train "$candidate_train" --qa-test "$candidate_test" --cv-folds "$candidate_folds" \
      --output "$RUN_ROOT/.candidate-historical-lineage.json" >/dev/null 2>&1; then
      VALID_CANDIDATES+=("$candidate")
      status="valid"
    else
      status="failed_lineage_validation"
    fi
  fi
  CANDIDATES_REPORT="$("$CLIENT_PYTHON" - "$CANDIDATES_REPORT" "$candidate" "$candidate_total" "$candidate_train" "$candidate_test" "$candidate_folds" "$status" <<'PY'
import json, sys
rows = json.loads(sys.argv[1])
rows.append({
    "path": sys.argv[2], "total": int(sys.argv[3]), "train": int(sys.argv[4]),
    "test": int(sys.argv[5]), "cv_folds": int(sys.argv[6]), "status": sys.argv[7],
})
print(json.dumps(rows))
PY
)"
done
"$CLIENT_PYTHON" - "$DISCOVERY_REPORT" "$GATEWAY_MANIFEST_SHA256" "$QA_SAMPLE_SIZE" "$CHECKPOINT_PARENT" "$CANDIDATES_REPORT" <<'PY'
import json, sys
path, sha, requested_total, parent, candidates_json = sys.argv[1:6]
candidates = json.loads(candidates_json)
report = {
    "protocol": "hallu-historical-cache-discovery-v1",
    "checkpoint_parent": parent,
    "gateway_manifest_sha256": sha,
    "requested_qa_sample_size": int(requested_total),
    "candidates": candidates,
    "valid_count": sum(1 for c in candidates if c["status"] == "valid"),
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(json.dumps(report, sort_keys=True))
PY
if (( ${#VALID_CANDIDATES[@]} != 1 )); then
  echo "Expected exactly one compatible historical QA checkpoint for QA_SAMPLE_SIZE=$QA_SAMPLE_SIZE; found ${#VALID_CANDIDATES[@]}." >&2
  echo "See $DISCOVERY_REPORT for every candidate that was checked and why it was rejected." >&2
  exit 2
fi
HISTORICAL_CACHE_DIR="${VALID_CANDIDATES[0]}"
"$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
  --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
  --checkpoint-identity "$HISTORICAL_CACHE_DIR/checkpoint-identity.json" \
  --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
  --qa-total "$QA_SAMPLE_SIZE" \
  --qa-train "$("$CLIENT_PYTHON" - "$HISTORICAL_CACHE_DIR/checkpoint-identity.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['qa_sample']['train'])
PY
)" \
  --qa-test "$("$CLIENT_PYTHON" - "$HISTORICAL_CACHE_DIR/checkpoint-identity.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['qa_sample']['test'])
PY
)" \
  --cv-folds "$("$CLIENT_PYTHON" - "$HISTORICAL_CACHE_DIR/checkpoint-identity.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['qa_sample']['alpha_cv_folds'])
PY
)" \
  --output "$HISTORICAL_LINEAGE"
HISTORICAL_CACHE_ROOT="$HISTORICAL_CACHE_DIR/kg"
HISTORICAL_LLM_FINGERPRINT="$("$CLIENT_PYTHON" - "$HISTORICAL_LINEAGE" <<'PY'
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
  --historical-cache-root "$HISTORICAL_CACHE_ROOT" --lineage "$HISTORICAL_LINEAGE" \
  --run-id historical-cache-replay --replay-count "$REPLAY_COUNT" \
  --replay-selection-seed "$REPLAY_SELECTION_SEED" | tee "$RUN_ROOT/historical-cache-replay-summary.log"
