#!/usr/bin/env bash
# Serial, resumable paired candidate-agreement evaluation on external disk.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${HALLU_CA_ROOT:-/Volumes/mySSD/hallu_smiles/candidate-agreement}"
SAMPLE_ROOT="${HALLU_CA_SAMPLE_ROOT:-/Volumes/mySSD/hallu_smiles/semantic-entropy}"
PYTHON="${HALLU_CA_PYTHON:-$SAMPLE_ROOT/.venv/bin/python}"
GATEWAY_URL="${HALLU_GATEWAY_URL:-https://hallu-docred-vertex-gateway-qbfs4yp45q-ez.a.run.app}"
EXPECTED_MANIFEST_SHA256="${HALLU_CA_GATEWAY_MANIFEST_SHA256:-cb4eb33be732ca708097a2c9b6267d6c6f7c61c1516fa4f64903226039247058}"
R12_ARCHIVE="${HALLU_CA_R12_ARCHIVE:-}"
RUN_ID="${HALLU_CA_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_WALL_SECONDS="${HALLU_CA_MAX_WALL_SECONDS:-259200}"
MAX_AUTO_RESUMES="${HALLU_CA_MAX_AUTO_RESUMES:-2}"
MIN_FREE_GIB="${HALLU_CA_MIN_FREE_GIB:-16}"
KEYCHAIN_SERVICE="${HALLU_GATEWAY_KEYCHAIN_SERVICE:-hallu-docred-gateway-api-key}"
KEYCHAIN_ACCOUNT="${HALLU_GATEWAY_KEYCHAIN_ACCOUNT:-maslovartemij}"
RUN_ROOT=""
AUTH_CONFIG=""
MONITOR_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_local_ragtruth_candidate_agreement.sh [--preflight]

The main run is fixed: the historical 750-row RAGTruth manifest (seed 42),
source 12448 quarantined, 15 Gemini samples at 65,535 tokens, and the exact
R12 graph-scored 147-response held-out comparison.  It never submits a
DataSphere job or runs graph inference.

Before a live fill, provide the downloaded, read-only R12 terminal archive via
HALLU_CA_R12_ARCHIVE.  The launcher refuses to generate if the R12 reference,
gateway manifest, or exact 8,385-hit/2,850-cold sample-cache contract fails.
EOF
}

safe_unlink() {
  [[ -n "${1:-}" ]] || return 0
  "$PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
}

load_gateway_key() {
  local key
  if ! key="$(security find-generic-password -w -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" 2>/dev/null)" || [[ -z "$key" ]]; then
    echo "Gateway key is unavailable in Keychain service $KEYCHAIN_SERVICE for the configured account." >&2
    return 1
  fi
  HALLU_GATEWAY_API_KEY="$key"
  export HALLU_GATEWAY_API_KEY
}

local_preflight() {
  test -x "$PYTHON" || { echo "External semantic runtime is missing: $PYTHON" >&2; return 2; }
  command -v security >/dev/null || { echo "macOS Keychain command is required" >&2; return 2; }
  command -v curl >/dev/null || { echo "curl is required" >&2; return 2; }
  command -v caffeinate >/dev/null || { echo "caffeinate is required" >&2; return 2; }
  command -v xelatex >/dev/null || { echo "XeLaTeX is required for the final report" >&2; return 2; }
  mkdir -p "$LOCAL_ROOT"
  local free_kib min_kib
  free_kib="$(df -Pk "$LOCAL_ROOT" | awk 'NR == 2 {print $4}')"
  min_kib=$((MIN_FREE_GIB * 1024 * 1024))
  (( free_kib >= min_kib )) || { echo "Insufficient free external-disk space" >&2; return 2; }
  test -f "$SAMPLE_ROOT/data/ragtruth-main/source_info.jsonl" || { echo "RAGTruth source data is missing under $SAMPLE_ROOT" >&2; return 2; }
  test -f "$SAMPLE_ROOT/data/ragtruth-main/response.jsonl" || { echo "RAGTruth response data is missing under $SAMPLE_ROOT" >&2; return 2; }
  test -f "$SAMPLE_ROOT/models/deberta-v2-xlarge-mnli/model-manifest.json" || { echo "Local DeBERTa NLI snapshot is missing" >&2; return 2; }
  [[ -n "$R12_ARCHIVE" && -f "$R12_ARCHIVE" ]] || {
    echo "Set HALLU_CA_R12_ARCHIVE to the read-only downloaded R12 terminal archive before any live fill." >&2
    return 2
  }
  echo "[ok] local candidate-agreement preflight passed; no Gemini inference was sent"
}

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  safe_unlink "$AUTH_CONFIG"
  unset HALLU_GATEWAY_API_KEY
  if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]]; then
    "$PYTHON" "$ROOT/scripts/archive_local_candidate_agreement.py" \
      --run-root "$RUN_ROOT" --archive "$LOCAL_ROOT/archives/$(basename "$RUN_ROOT").tar.gz" || status=1
  fi
  exit "$status"
}

mode="run"
if [[ $# -gt 0 ]]; then
  case "$1" in
    --preflight) mode="preflight" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
fi
local_preflight
if [[ "$mode" == "preflight" ]]; then
  exit 0
fi

if [[ "${HALLU_CA_CAFFEINATED:-}" != "1" ]]; then
  export HALLU_CA_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0"
fi

RUN_ROOT="$LOCAL_ROOT/runs/ragtruth-candidate-agreement-artifacts-$RUN_ID"
test ! -e "$RUN_ROOT" || { echo "run root already exists: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT/archives" "$LOCAL_ROOT/checkpoints"
printf '%s\n' "$$" > "$RUN_ROOT/runner.pid"
trap cleanup EXIT
"$PYTHON" "$ROOT/scripts/monitor_local_candidate_agreement.py" \
  --run-root "$RUN_ROOT" --pid "$$" --interval-seconds 900 > "$RUN_ROOT/local-monitor.log" 2>&1 &
MONITOR_PID=$!

# R12 is consumed read-only and reduced immediately to scalar, redacted pairing.
GRAPH_REFERENCE="$RUN_ROOT/r12-paired-graph-reference.json"
"$PYTHON" "$ROOT/scripts/extract_r12_paired_graph_reference.py" \
  --r12-archive "$R12_ARCHIVE" --output "$GRAPH_REFERENCE" > "$RUN_ROOT/r12-reference-status.json"

load_gateway_key
AUTH_CONFIG="$(mktemp "$RUN_ROOT/.gateway-curl.XXXXXX")"
chmod 600 "$AUTH_CONFIG"
printf 'header = "Authorization: Bearer %s"\n' "$HALLU_GATEWAY_API_KEY" > "$AUTH_CONFIG"
GATEWAY_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
set +e
curl --fail --silent --show-error --config "$AUTH_CONFIG" --output "$GATEWAY_RAW" "$GATEWAY_URL/v1/hallu/manifest"
manifest_status=$?
set -e
if [[ "$manifest_status" -ne 0 ]]; then
  echo "gateway manifest check failed; no Gemini generation started" >&2
  exit "$manifest_status"
fi
"$PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$GATEWAY_RAW" --logical-model "openai/gemini-2.5-flash" --output "$GATEWAY_MANIFEST"
safe_unlink "$AUTH_CONFIG"
AUTH_CONFIG=""
safe_unlink "$GATEWAY_RAW"
GATEWAY_HASH="$("$PYTHON" - "$GATEWAY_MANIFEST" <<'PY'
import json
import sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding='utf-8'))))
PY
)"
if [[ "$GATEWAY_HASH" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "gateway manifest is not cache-compatible with the verified semantic sample cache; no Gemini generation started" >&2
  exit 2
fi

SAMPLE_CACHE_ROOT="$SAMPLE_ROOT/checkpoints/semantic-entropy-v1-$EXPECTED_MANIFEST_SHA256"
test -d "$SAMPLE_CACHE_ROOT/semantic_entropy" || { echo "Compatible semantic sample cache is missing" >&2; exit 2; }
SEMANTIC_RUNTIME="$RUN_ROOT/semantic-runtime-config.yaml"
"$PYTHON" "$ROOT/scripts/make_local_semantic_entropy_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$GATEWAY_URL" --nli-model-path "$SAMPLE_ROOT/models/deberta-v2-xlarge-mnli" \
  --data-dir "$SAMPLE_ROOT/data/ragtruth-main" --cache-root "$SAMPLE_CACHE_ROOT" \
  --output "$SEMANTIC_RUNTIME" > "$RUN_ROOT/semantic-runtime-identity.json"
RUNTIME_CONFIG="$RUN_ROOT/runtime-config.yaml"
"$PYTHON" "$ROOT/scripts/make_local_candidate_agreement_config.py" \
  --semantic-runtime-config "$SEMANTIC_RUNTIME" \
  --candidate-cache-root "$LOCAL_ROOT/checkpoints/candidate-agreement-v1-$EXPECTED_MANIFEST_SHA256" \
  --expected-gateway-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --output "$RUNTIME_CONFIG" \
  > "$RUN_ROOT/runtime-identity.json"

MANIFEST="$LOCAL_ROOT/checkpoints/manifests/ragtruth-qa-candidate-agreement-750.json"
PREFLIGHT_OUT="$RUN_ROOT/sample-cache-preflight"
# `python_hash_seed` is part of the semantic cache identity.  Run the
# cache-only inventory under the same pinned process environment as the
# historical writer and the live/replay stages below.
TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
  "$PYTHON" "$ROOT/scripts/run_ragtruth_candidate_agreement.py" \
  --config "$RUNTIME_CONFIG" --data-dir "$SAMPLE_ROOT/data/ragtruth-main" --output-dir "$PREFLIGHT_OUT" \
  --manifest "$MANIFEST" --graph-reference "$GRAPH_REFERENCE" \
  --required-gateway-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --preflight-sample-inventory

# This offline stage has existing Gemini samples by construction.  It creates
# ten redacted local NLI comparison cache entries but sends zero Gemini calls.
SMOKE_OUT="$RUN_ROOT/offline-sample-smoke-10"
TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
  "$PYTHON" "$ROOT/scripts/run_ragtruth_candidate_agreement.py" \
    --config "$RUNTIME_CONFIG" --data-dir "$SAMPLE_ROOT/data/ragtruth-main" --output-dir "$SMOKE_OUT" \
    --manifest "$MANIFEST" --graph-reference "$GRAPH_REFERENCE" \
    --required-gateway-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --offline-sample-smoke 10
"$PYTHON" - "$SMOKE_OUT/run_metadata.json" <<'PY'
import json
import sys
m=json.load(open(sys.argv[1], encoding='utf-8'))
if m.get('state') != 'completed_offline_sample_smoke' or int(m.get('usage', {}).get('api_calls', -1)) != 0:
    raise SystemExit('offline smoke was not a zero-Gemini sample-cache check')
PY

run_resilient() {
  local output_dir="$1" deadline_epoch="$2" attempts=0 status=0
  while true; do
    local now_epoch remaining
    now_epoch="$(date +%s)"
    remaining=$((deadline_epoch - now_epoch))
    (( remaining > 0 )) || return 76
    set +e
    TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
      "$PYTHON" "$ROOT/scripts/run_ragtruth_candidate_agreement.py" \
        --config "$RUNTIME_CONFIG" --data-dir "$SAMPLE_ROOT/data/ragtruth-main" --output-dir "$output_dir" \
        --manifest "$MANIFEST" --graph-reference "$GRAPH_REFERENCE" \
        --required-gateway-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
        --n-bootstrap 1000 --wall-clock-budget-s "$remaining" \
        > >(tee -a "$RUN_ROOT/stdout.log") 2> >(tee -a "$RUN_ROOT/stderr.log" >&2)
    status=$?
    set -e
    if [[ "$status" -ne 75 ]]; then
      return "$status"
    fi
    attempts=$((attempts + 1))
    (( attempts <= MAX_AUTO_RESUMES )) || return 75
    sleep "$((120 * attempts))"
  done
}

MAIN_OUT="$RUN_ROOT/main-750"
DEADLINE_EPOCH=$(( $(date +%s) + MAX_WALL_SECONDS ))
run_resilient "$MAIN_OUT" "$DEADLINE_EPOCH"
"$PYTHON" - "$SMOKE_OUT/run_metadata.json" "$MAIN_OUT/run_metadata.json" <<'PY'
import json
import sys
m_smoke=json.load(open(sys.argv[1], encoding='utf-8'))
m=json.load(open(sys.argv[2], encoding='utf-8'))
if m.get('state') != 'completed' or int(m.get('sources_completed', -1)) != 749:
    raise SystemExit('candidate-agreement run did not cover all 749 eligible sources')
if int(m_smoke.get('nli_pair_evaluations', -1)) + int(m.get('nli_pair_evaluations', -1)) != 22470:
    raise SystemExit('candidate-agreement protocol did not perform the required 22,470 directional NLI comparisons')
PY

CANDIDATE_CACHE="$LOCAL_ROOT/checkpoints/candidate-agreement-v1-$EXPECTED_MANIFEST_SHA256/candidate_agreement"
INVENTORY_BEFORE="$RUN_ROOT/cache-before-replay.sha256"
INVENTORY_AFTER="$RUN_ROOT/cache-after-replay.sha256"
find "$SAMPLE_CACHE_ROOT/semantic_entropy" "$CANDIDATE_CACHE" -type f -print0 | sort -z | xargs -0 shasum -a 256 > "$INVENTORY_BEFORE"
REPLAY_OUT="$MAIN_OUT/replay"
TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
  "$PYTHON" "$ROOT/scripts/run_ragtruth_candidate_agreement.py" \
    --config "$RUNTIME_CONFIG" --data-dir "$SAMPLE_ROOT/data/ragtruth-main" --output-dir "$REPLAY_OUT" \
    --manifest "$MANIFEST" --graph-reference "$GRAPH_REFERENCE" \
    --required-gateway-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
    --n-bootstrap 1000 --cache-only --recompute-from-cache --marker-checkpoint-dir "$MAIN_OUT"
find "$SAMPLE_CACHE_ROOT/semantic_entropy" "$CANDIDATE_CACHE" -type f -print0 | sort -z | xargs -0 shasum -a 256 > "$INVENTORY_AFTER"
"$PYTHON" "$ROOT/scripts/verify_candidate_agreement_cache_replay.py" \
  --live-dir "$MAIN_OUT" --replay-dir "$REPLAY_OUT" \
  --cache-before "$INVENTORY_BEFORE" --cache-after "$INVENTORY_AFTER"

# The archive is written by the EXIT trap.  Produce and validate the TeX
# report after a fresh redacted archive is available to keep report provenance
# restricted to terminal artifacts.
"$PYTHON" "$ROOT/scripts/archive_local_candidate_agreement.py" \
  --run-root "$RUN_ROOT" --archive "$LOCAL_ROOT/archives/$(basename "$RUN_ROOT").tar.gz"
REPORT="$ROOT/docs/ragtruth-candidate-agreement-results.tex"
"$PYTHON" "$ROOT/scripts/write_candidate_agreement_results_tex.py" \
  --archive "$LOCAL_ROOT/archives/$(basename "$RUN_ROOT").tar.gz" --output "$REPORT"
mkdir -p "$RUN_ROOT/report-build"
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$RUN_ROOT/report-build" "$REPORT" > "$RUN_ROOT/xelatex.log"
echo "[ok] candidate-agreement live run, replay, archive, and XeLaTeX report completed"
