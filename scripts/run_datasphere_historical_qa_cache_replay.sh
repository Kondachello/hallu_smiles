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
QA_TEST_FRACTION="${QA_TEST_FRACTION:-0.2}"
QA_CV_FOLDS="${QA_CV_FOLDS:-5}"
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

HISTORICAL_LINEAGE="$RUN_ROOT/historical-lineage.json"
DISCOVERY_REPORT="$RUN_ROOT/reports/historical_cache_discovery.json"
mkdir -p "$(dirname "$DISCOVERY_REPORT")"
test -d "$CHECKPOINT_PARENT" || {
  echo "Historical checkpoint parent is absent: $CHECKPOINT_PARENT" >&2
  exit 2
}

# --- Step A: find WHICH directory to read from (the "target" checkpoint for
# QA_SAMPLE_SIZE). run_datasphere_vertex_cpu_qa_pilot.sh names baseline
# checkpoint dirs by convention (qa-<total>-test-<test>-cv-<folds>/baseline-v1-<sha>)
# and never writes checkpoint-identity.json for that namespace (only for its
# support-critical one) -- so a fresh baseline checkpoint (e.g. 750-QA) has no
# identity file and must be found by path alone. -------------------------
read -r QA_TRAIN_SOURCES QA_TEST_SOURCES < <("$CLIENT_PYTHON" - "$QA_SAMPLE_SIZE" "$QA_TEST_FRACTION" <<'PY'
import sys
from src.sampling import qa_sample_quotas
train, test = qa_sample_quotas(int(sys.argv[1]), sys.argv[2])
print(train, test)
PY
)
DIRECT_CHECKPOINT_BASE="$CHECKPOINT_PARENT/qa-${QA_SAMPLE_SIZE}-test-${QA_TEST_SOURCES}-cv-${QA_CV_FOLDS}"
DIRECT_BASELINE_DIR="$DIRECT_CHECKPOINT_BASE/baseline-v1-${GATEWAY_MANIFEST_SHA256}"
TARGET_CACHE_ROOT=""
TARGET_MODE="none"
if [[ -d "$DIRECT_BASELINE_DIR/kg" ]] && [[ -n "$(ls -A "$DIRECT_BASELINE_DIR/kg" 2>/dev/null)" ]]; then
  TARGET_MODE="direct"
  TARGET_CACHE_ROOT="$DIRECT_BASELINE_DIR/kg"
fi

# --- Step B: find WHICH fingerprint every baseline checkpoint under this
# gateway manifest was actually written with. Every generation of the QA
# checkpoint (100-QA, 750-QA, ...) is written with
# --llm-runtime-fingerprint-override pinned to the SAME original historical
# lineage fingerprint (see run_datasphere_vertex_cpu_qa_pilot.sh), not a
# freshly computed one -- that is what keeps their cache keys mutually
# compatible. So this search is independent of QA_SAMPLE_SIZE: it only needs
# to find the one identity.json-sealed checkpoint (of any size) that is
# registered in historical_kg_cache_lineages.json for the current gateway
# manifest, and use ITS fingerprint. --------------------------------------
mapfile -t IDENTITY_CANDIDATES < <(
  find "$CHECKPOINT_PARENT" -mindepth 2 -maxdepth 2 -type d -name "*-${GATEWAY_MANIFEST_SHA256}" -print 2>/dev/null | sort
)
LINEAGE_FINGERPRINT=""
LINEAGE_SOURCE_DIR=""
LINEAGE_CANDIDATES_REPORT="[]"
for candidate in "${IDENTITY_CANDIDATES[@]}"; do
  test -f "$candidate/checkpoint-identity.json" || continue
  read -r candidate_total candidate_train candidate_test candidate_folds < <("$CLIENT_PYTHON" - "$candidate/checkpoint-identity.json" <<'PY'
import json, sys
sample = json.load(open(sys.argv[1], encoding='utf-8')).get('qa_sample', {})
try:
    print(int(sample['total']), int(sample['train']), int(sample['test']), int(sample['alpha_cv_folds']))
except (KeyError, TypeError, ValueError):
    raise SystemExit(2)
PY
) || continue
  status="failed_lineage_validation"
  if [[ -z "$LINEAGE_FINGERPRINT" ]] && "$CLIENT_PYTHON" "$ROOT/scripts/resolve_datasphere_historical_cache_lineage.py" \
    --lineages "$ROOT/datasphere/historical_kg_cache_lineages.json" \
    --checkpoint-identity "$candidate/checkpoint-identity.json" \
    --runtime-manifest "$RUNTIME_MANIFEST" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
    --qa-total "$candidate_total" --qa-train "$candidate_train" --qa-test "$candidate_test" --cv-folds "$candidate_folds" \
    --output "$RUN_ROOT/.candidate-historical-lineage.json" >/dev/null 2>&1; then
    LINEAGE_FINGERPRINT="$("$CLIENT_PYTHON" - "$RUN_ROOT/.candidate-historical-lineage.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['llm_runtime_fingerprint'])
PY
)"
    LINEAGE_SOURCE_DIR="$candidate"
    cp "$RUN_ROOT/.candidate-historical-lineage.json" "$HISTORICAL_LINEAGE"
    status="valid_fingerprint_source"
  fi
  LINEAGE_CANDIDATES_REPORT="$("$CLIENT_PYTHON" - "$LINEAGE_CANDIDATES_REPORT" "$candidate" "$candidate_total" "$candidate_train" "$candidate_test" "$candidate_folds" "$status" <<'PY'
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

# --- Step C: combine. The directory to read is TARGET_CACHE_ROOT (direct, by
# naming convention, for the requested QA_SAMPLE_SIZE); if no fresh checkpoint
# exists for that size, fall back to reading directly from the fingerprint
# source checkpoint itself (this is the original 100-QA case). Either way,
# every cache-key computation always uses LINEAGE_FINGERPRINT as an explicit
# override -- never a freshly computed fingerprint -- because that is the
# pinned identity every checkpoint generation was actually written with. ---
if [[ -z "$LINEAGE_FINGERPRINT" ]]; then
  "$CLIENT_PYTHON" - "$DISCOVERY_REPORT" "$GATEWAY_MANIFEST_SHA256" "$QA_SAMPLE_SIZE" "$CHECKPOINT_PARENT" "$LINEAGE_CANDIDATES_REPORT" "$DIRECT_BASELINE_DIR" "$TARGET_MODE" <<'PY'
import json, sys
path, sha, requested_total, parent, candidates_json, direct_dir, target_mode = sys.argv[1:8]
candidates = json.loads(candidates_json)
report = {
    "protocol": "hallu-historical-cache-discovery-v1",
    "checkpoint_parent": parent,
    "gateway_manifest_sha256": sha,
    "requested_qa_sample_size": int(requested_total),
    "discovery_mode": "failed_no_fingerprint_source",
    "target_mode": target_mode,
    "target_baseline_dir": direct_dir,
    "candidates": candidates,
    "valid_count": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(json.dumps(report, sort_keys=True))
PY
  echo "No checkpoint under $CHECKPOINT_PARENT with a checkpoint-identity.json resolves against historical_kg_cache_lineages.json for gateway manifest $GATEWAY_MANIFEST_SHA256; cannot determine the pinned LLM runtime fingerprint." >&2
  echo "See $DISCOVERY_REPORT for every candidate that was checked and why it was rejected." >&2
  exit 2
fi
if [[ "$TARGET_MODE" != "direct" ]]; then
  # No fresh baseline-v1-<sha> checkpoint for this exact QA_SAMPLE_SIZE: read
  # straight from the fingerprint-source checkpoint itself (this is the
  # original 100-QA case, where the lineage checkpoint IS the target).
  TARGET_MODE="lineage"
  TARGET_CACHE_ROOT="$LINEAGE_SOURCE_DIR/kg"
fi
"$CLIENT_PYTHON" - "$DISCOVERY_REPORT" "$GATEWAY_MANIFEST_SHA256" "$QA_SAMPLE_SIZE" "$CHECKPOINT_PARENT" "$LINEAGE_CANDIDATES_REPORT" "$DIRECT_BASELINE_DIR" "$TARGET_MODE" "$TARGET_CACHE_ROOT" <<'PY'
import json, sys
path, sha, requested_total, parent, candidates_json, direct_dir, target_mode, target_cache_root = sys.argv[1:9]
candidates = json.loads(candidates_json)
report = {
    "protocol": "hallu-historical-cache-discovery-v1",
    "checkpoint_parent": parent,
    "gateway_manifest_sha256": sha,
    "requested_qa_sample_size": int(requested_total),
    "discovery_mode": target_mode,
    "target_cache_root": target_cache_root,
    "candidates": candidates,
    "valid_count": sum(1 for c in candidates if c["status"] == "valid_fingerprint_source"),
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(json.dumps(report, sort_keys=True))
PY
test -d "$TARGET_CACHE_ROOT" && test -n "$(ls -A "$TARGET_CACHE_ROOT" 2>/dev/null)" || {
  echo "Resolved target cache root is missing or empty: $TARGET_CACHE_ROOT" >&2
  exit 2
}
"$CLIENT_PYTHON" - "$HISTORICAL_LINEAGE" "$TARGET_CACHE_ROOT" "$TARGET_MODE" <<'PY'
import json, sys
lineage_path, cache_root, target_mode = sys.argv[1:4]
lineage = json.load(open(lineage_path, encoding='utf-8'))
lineage["historical_cache_root"] = cache_root
lineage["target_mode"] = target_mode
with open(lineage_path, "w", encoding="utf-8") as fh:
    json.dump(lineage, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
HISTORICAL_CACHE_ROOT="$TARGET_CACHE_ROOT"
"$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$MANIFEST" \
  --gateway-url "$RECORDED_GATEWAY_URL" --datasphere-runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$RUN_ROOT/historical-cache-runtime.yaml" --data-dir "$DATA_DIR" \
  --work-dir "$RUN_ROOT" --cache-root "$RUN_ROOT/current-cache" \
  --llm-runtime-fingerprint-override "$LINEAGE_FINGERPRINT" \
  --max-tokens 16384 --concurrency 1 --max-retries 0 --retry-backoff-base-s 5 \
  --retry-backoff-max-s 60 --retry-backoff-jitter-s 0 --cv-folds "$QA_CV_FOLDS" \
  > "$RUN_ROOT/historical-cache-runtime-identity.json"

# In direct (750-QA) mode the baseline-v1 kg only holds graphs newly extracted
# for that run; the QA records it shares with the smaller lineage checkpoint
# (e.g. the 100-QA cache) are read through from that checkpoint's kg. Chain it
# as a second, lower-priority read source so those shared records resolve too.
LINEAGE_KG_DIR=""
if [[ "$TARGET_MODE" == "direct" && -n "$LINEAGE_SOURCE_DIR" && -d "$LINEAGE_SOURCE_DIR/kg" \
      && "$LINEAGE_SOURCE_DIR/kg" != "$HISTORICAL_CACHE_ROOT" ]]; then
  LINEAGE_KG_DIR="$LINEAGE_SOURCE_DIR/kg"
fi

if [[ "${TYPED_METRIC_PASS:-0}" == "1" ]]; then
  # Typed-vertex metric pass: HalluGraph/GraphEval graphs stay cache-only (read
  # through the resolved roots), but the dynamic typing agent assigns a type to
  # every vertex (permitted gateway LLM calls + local HHEM NLI) and we score the
  # type-aware EG + edge RP -> CFI_type. Reuses the full cache resolution above.
  export HALLU_TYPING_MODEL="${HALLU_TYPING_MODEL:-openai/gemini-2.5-flash}"
  export HALLU_GATEWAY_URL="${HALLU_GATEWAY_URL:-$RECORDED_GATEWAY_URL}"
  export HALLU_HHEM_MODEL_PATH="${HALLU_HHEM_MODEL_PATH:-/opt/hallu/models/hhem-2.1-open}"
  # The cache-only runtime image ships litellm/torch/transformers but not the typing
  # agent's graph runtime (langgraph/langchain-core). Install the missing ones into a
  # job-local target appended to PYTHONPATH (image deps such as pydantic keep priority).
  if ! "$CLIENT_PYTHON" -c "import langgraph, langchain_core" >/dev/null 2>&1; then
    # Pin the cache-key package snapshot to the pinned IMAGE versions BEFORE the
    # bootstrap: langchain-core>=1 drags in pydantic>=2.11 (and a newer tenacity)
    # which would otherwise shadow the image's pydantic 2.10.6 / tenacity 9.0.0 on
    # sys.path and, since those versions feed the cache key, make every historical
    # graph miss. The typing agent still runs against the newer libs; only the key
    # snapshot is frozen to the image. See src.cache._installed_versions.
    export HALLU_CACHE_KEY_PACKAGE_VERSIONS="$("$CLIENT_PYTHON" - <<'PY'
import json
from importlib import metadata
out = {}
for p in ("kg-gen","dspy","litellm","pydantic","jsonschema","tenacity",
          "sentence-transformers","transformers","torch","numpy"):
    try:
        out[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        out[p] = "not-installed"
print(json.dumps(out))
PY
)"
    echo "pinned cache-key packages: $HALLU_CACHE_KEY_PACKAGE_VERSIONS"
    TYPED_PYDEPS="$RUN_ROOT/pydeps"
    "$CLIENT_PYTHON" -m pip install --quiet --disable-pip-version-check \
      --target "$TYPED_PYDEPS" "langgraph>=1,<2" "langchain-core>=1,<2"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$TYPED_PYDEPS"
    "$CLIENT_PYTHON" -c "import langgraph, langchain_core; print('typing deps ready')"
  fi
  TYPED_KG_ARGS=(--historical-cache-root "$HISTORICAL_CACHE_ROOT")
  [[ -n "$LINEAGE_KG_DIR" ]] && TYPED_KG_ARGS+=(--additional-cache-root "$LINEAGE_KG_DIR")
  "$CLIENT_PYTHON" "$ROOT/scripts/typed_metric_pass.py" \
    --data-dir "$DATA_DIR" --output-root "$RUN_ROOT" \
    --hallugraph-config "$RUN_ROOT/historical-cache-runtime.yaml" \
    --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml" \
    --typing-config "${HALLU_TYPING_CONFIG:-$ROOT/dynamic_typing_agent/config/live-gateway-hhem.yaml}" \
    "${TYPED_KG_ARGS[@]}" --gateway-manifest-sha256 "$GATEWAY_MANIFEST_SHA256" \
    --qa-sample-size "$QA_SAMPLE_SIZE" --qa-test-fraction "$QA_TEST_FRACTION" \
    --replay-count "$REPLAY_COUNT" --replay-selection-seed "$REPLAY_SELECTION_SEED" \
    --alpha "${TYPED_METRIC_ALPHA:-0.5}" --max-workers "${TYPED_METRIC_MAX_WORKERS:-1}" | tee "$RUN_ROOT/typed-metric-pass-summary.log"
elif [[ "${DIAGNOSTIC_ONLY:-0}" == "1" ]]; then
  # Read-only: check how many computed cache_keys for the selected QA texts
  # already exist as files in the chained read sources, using the exact same
  # config this replay would otherwise run with. No gateway call beyond the
  # manifest fetch already done above; no LLM inference; nothing written to
  # the checkpoint.
  DIAG_KG_ARGS=(--kg-dir "$HISTORICAL_CACHE_ROOT")
  [[ -n "$LINEAGE_KG_DIR" ]] && DIAG_KG_ARGS+=(--kg-dir "$LINEAGE_KG_DIR")
  "$CLIENT_PYTHON" "$ROOT/scripts/diagnose_historical_cache_key_mismatch.py" \
    --config "$RUN_ROOT/historical-cache-runtime.yaml" \
    "${DIAG_KG_ARGS[@]}" --data-dir "$DATA_DIR" \
    --qa-sample-size "$QA_SAMPLE_SIZE" --qa-test-fraction "$QA_TEST_FRACTION" \
    | tee "$RUN_ROOT/diagnostic-cache-key-report.json"
else
  REPLAY_SOURCE_ARGS=(--historical-cache-root "$HISTORICAL_CACHE_ROOT")
  [[ -n "$LINEAGE_KG_DIR" ]] && REPLAY_SOURCE_ARGS+=(--additional-cache-root "$LINEAGE_KG_DIR")
  "$CLIENT_PYTHON" "$ROOT/scripts/historical_qa_cache_replay_probe.py" \
    --data-dir "$DATA_DIR" --output-root "$RUN_ROOT" \
    --hallugraph-config "$RUN_ROOT/historical-cache-runtime.yaml" \
    --grapheval-config "$ROOT/graph_eval/config.datasphere.one-instance.shared-kggen.live.yaml" \
    "${REPLAY_SOURCE_ARGS[@]}" --lineage "$HISTORICAL_LINEAGE" \
    --qa-sample-size "$QA_SAMPLE_SIZE" --qa-test-fraction "$QA_TEST_FRACTION" \
    --run-id historical-cache-replay --replay-count "$REPLAY_COUNT" \
    --replay-selection-seed "$REPLAY_SELECTION_SEED" | tee "$RUN_ROOT/historical-cache-replay-summary.log"
fi
