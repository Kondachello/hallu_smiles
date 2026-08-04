#!/usr/bin/env bash
# Resumable external-disk RAGTruth QA SemanticEntropy baseline.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${HALLU_SE_ROOT:-/Volumes/mySSD/hallu_smiles/semantic-entropy}"
PYTHON="${HALLU_SE_PYTHON:-$LOCAL_ROOT/.venv/bin/python}"
BOOTSTRAP_PYTHON="${HALLU_SE_BOOTSTRAP_PYTHON:-/opt/homebrew/bin/python3.12}"
GATEWAY_URL="${HALLU_GATEWAY_URL:-https://hallu-vertex-gateway-453887629111.europe-west4.run.app}"
RUN_ID="${HALLU_SE_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_WALL_SECONDS="${HALLU_SE_MAX_WALL_SECONDS:-86400}"
MIN_FREE_GIB="${HALLU_SE_MIN_FREE_GIB:-12}"
MAX_AUTO_RESUMES="${HALLU_SE_MAX_AUTO_RESUMES:-2}"
RUN_ROOT=""
AUTH_CONFIG=""
MONITOR_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_local_ragtruth_semantic_entropy.sh [--preflight]

The runner materializes all data, the DeBERTa NLI model, Python venv, Gemini
cache/checkpoints, logs and redacted archives under HALLU_SE_ROOT (default:
/Volumes/mySSD/hallu_smiles/semantic-entropy). It runs a 10-QA measured pilot,
then automatically chooses the largest balanced 80/20 source sample up to 750
that fits a conservative 24-hour estimate (or HALLU_SE_MAX_WALL_SECONDS).
EOF
}

keychain_services() {
  if [[ -n "${HALLU_GATEWAY_KEYCHAIN_SERVICE:-}" ]]; then
    printf '%s\n' "$HALLU_GATEWAY_KEYCHAIN_SERVICE"
    return
  fi
  printf '%s\n' "HALLU_GATEWAY_API_KEY" "hallu-smiles-gateway-api-key" "hallu-smiles/gateway-api-key" "gemini"
}

load_gateway_key() {
  local service key
  while IFS= read -r service; do
    if key="$(security find-generic-password -w -s "$service" 2>/dev/null)" && [[ -n "$key" ]]; then
      HALLU_GATEWAY_API_KEY="$key"
      export HALLU_GATEWAY_API_KEY
      return 0
    fi
  done < <(keychain_services)
  echo "No gateway key found in Keychain. Set HALLU_GATEWAY_KEYCHAIN_SERVICE to its service name." >&2
  return 1
}

disk_available_kib() {
  df -Pk "$LOCAL_ROOT" | awk 'NR == 2 {print $4}'
}

bootstrap_runtime() {
  mkdir -p "$LOCAL_ROOT" "$LOCAL_ROOT/huggingface" "$LOCAL_ROOT/pip-cache"
  if [[ ! -x "$PYTHON" ]]; then
    test -x "$BOOTSTRAP_PYTHON" || {
      echo "Python 3.10-3.12 bootstrap interpreter missing: $BOOTSTRAP_PYTHON" >&2
      return 2
    }
    "$BOOTSTRAP_PYTHON" -m venv "$(dirname "$(dirname "$PYTHON")")"
  fi
  PIP_CACHE_DIR="$LOCAL_ROOT/pip-cache" "$PYTHON" -m pip install --upgrade pip >/dev/null
  PIP_CACHE_DIR="$LOCAL_ROOT/pip-cache" "$PYTHON" -m pip install --quiet -r "$ROOT/requirements-semantic-entropy.txt"
  local model_dir="$LOCAL_ROOT/models/deberta-v2-xlarge-mnli"
  if [[ ! -f "$model_dir/model-manifest.json" ]]; then
    HF_HOME="$LOCAL_ROOT/huggingface" TRANSFORMERS_CACHE="$LOCAL_ROOT/huggingface/transformers" \
      "$PYTHON" "$ROOT/scripts/setup_semantic_entropy_runtime.py" --output-dir "$model_dir"
  fi
}

preflight() {
  test -x "$PYTHON" || { echo "External semantic-entropy venv missing: $PYTHON" >&2; return 2; }
  command -v curl >/dev/null || { echo "curl is required" >&2; return 2; }
  command -v security >/dev/null || { echo "macOS Keychain command is required" >&2; return 2; }
  command -v caffeinate >/dev/null || { echo "caffeinate is required on macOS" >&2; return 2; }
  mkdir -p "$LOCAL_ROOT"
  local free_kib minimum_kib
  free_kib="$(disk_available_kib)"
  minimum_kib=$((MIN_FREE_GIB * 1024 * 1024))
  (( free_kib >= minimum_kib )) || {
    echo "Insufficient external-disk space: need ${MIN_FREE_GIB} GiB free under $LOCAL_ROOT" >&2
    return 2
  }
  "$PYTHON" - <<'PY'
import importlib.util
required = ("numpy", "sklearn", "tenacity", "yaml", "torch", "transformers", "huggingface_hub")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("semantic runtime missing: " + ", ".join(missing))
PY
  test -f "$LOCAL_ROOT/models/deberta-v2-xlarge-mnli/model-manifest.json" || {
    echo "Local DeBERTa snapshot missing: $LOCAL_ROOT/models/deberta-v2-xlarge-mnli" >&2
    return 2
  }
  load_gateway_key
  unset HALLU_GATEWAY_API_KEY
  echo "[ok] semantic-entropy preflight passed; no Gemini inference was sent"
}

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  rm -f "${AUTH_CONFIG:-}"
  unset HALLU_GATEWAY_API_KEY
  if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]]; then
    "$PYTHON" "$ROOT/scripts/archive_local_semantic_entropy.py" \
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
if [[ "$mode" == "preflight" ]]; then
  bootstrap_runtime
  preflight
  exit $?
fi

if [[ "${HALLU_SE_CAFFEINATED:-}" != "1" ]]; then
  export HALLU_SE_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0"
fi

bootstrap_runtime
preflight
RUN_ROOT="$LOCAL_ROOT/runs/ragtruth-semantic-entropy-artifacts-$RUN_ID"
test ! -e "$RUN_ROOT" || { echo "run root already exists: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT/archives" "$LOCAL_ROOT/data" "$LOCAL_ROOT/checkpoints"
printf '%s\n' "$$" > "$RUN_ROOT/runner.pid"
trap cleanup EXIT

load_gateway_key
AUTH_CONFIG="$(mktemp "$RUN_ROOT/.gateway-curl.XXXXXX")"
chmod 600 "$AUTH_CONFIG"
printf 'header = "Authorization: Bearer %s"\n' "$HALLU_GATEWAY_API_KEY" > "$AUTH_CONFIG"
GATEWAY_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
set +e
gateway_manifest_status="$(curl --fail --silent --show-error --config "$AUTH_CONFIG" \
  --output "$GATEWAY_RAW" --write-out '%{http_code}' "$GATEWAY_URL/v1/hallu/manifest")"
gateway_manifest_curl_status=$?
set -e
if [[ "$gateway_manifest_curl_status" -ne 0 ]]; then
  echo "gateway manifest request failed (HTTP ${gateway_manifest_status:-unavailable}); no Gemini inference started" >&2
  exit "$gateway_manifest_curl_status"
fi
"$PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$GATEWAY_RAW" --logical-model "openai/gemini-2.5-flash" --output "$GATEWAY_MANIFEST"
rm -f "$GATEWAY_RAW" "$AUTH_CONFIG"
AUTH_CONFIG=""

DATA_DIR="$LOCAL_ROOT/data/ragtruth-main"
"$PYTHON" "$ROOT/download_data.py" --data-dir "$DATA_DIR" > "$RUN_ROOT/dataset-materialization.json"
MODEL_DIR="$LOCAL_ROOT/models/deberta-v2-xlarge-mnli"
HF_HOME="$LOCAL_ROOT/huggingface" TRANSFORMERS_CACHE="$LOCAL_ROOT/huggingface/transformers" \
  "$PYTHON" "$ROOT/scripts/setup_semantic_entropy_runtime.py" --output-dir "$MODEL_DIR" \
  > "$RUN_ROOT/nli-materialization.json"

GATEWAY_HASH="$("$PYTHON" - "$GATEWAY_MANIFEST" <<'PY'
import json
import sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
CACHE_ROOT="$LOCAL_ROOT/checkpoints/semantic-entropy-v1-$GATEWAY_HASH"
mkdir -p "$CACHE_ROOT"
RUNTIME_CONFIG="$RUN_ROOT/runtime-config.yaml"
"$PYTHON" "$ROOT/scripts/make_local_semantic_entropy_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$GATEWAY_URL" --nli-model-path "$MODEL_DIR" --data-dir "$DATA_DIR" \
  --cache-root "$CACHE_ROOT" --output "$RUNTIME_CONFIG" > "$RUN_ROOT/runtime-identity.json"

"$PYTHON" "$ROOT/scripts/monitor_local_semantic_entropy.py" \
  --run-root "$RUN_ROOT" --pid "$$" --interval-seconds 900 > "$RUN_ROOT/local-monitor.log" 2>&1 &
MONITOR_PID=$!

run_resilient() {
  local output_dir="$1" manifest="$2" total="$3" deadline_epoch="$4" attempts=0 status=0
  while true; do
    local now_epoch remaining
    now_epoch="$(date +%s)"
    remaining=$((deadline_epoch - now_epoch))
    (( remaining > 0 )) || { echo "live semantic-entropy wall-clock budget exhausted" >&2; return 76; }
    set +e
    HF_HOME="$LOCAL_ROOT/huggingface" TRANSFORMERS_CACHE="$LOCAL_ROOT/huggingface/transformers" \
      TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
      "$PYTHON" "$ROOT/scripts/run_ragtruth_semantic_entropy.py" \
        --config "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --output-dir "$output_dir" \
        --manifest "$manifest" --total-sources "$total" --seed 42 --n-bootstrap 1000 \
        --wall-clock-budget-s "$remaining" \
        > >(tee -a "$RUN_ROOT/stdout.log") 2> >(tee -a "$RUN_ROOT/stderr.log" >&2)
    status=$?
    set -e
    if [[ "$status" -ne 75 ]]; then
      return "$status"
    fi
    attempts=$((attempts + 1))
    if (( attempts > MAX_AUTO_RESUMES )); then
      echo "gateway remained unavailable after ${MAX_AUTO_RESUMES} autonomous resumes" >&2
      return 75
    fi
    local cooldown=$((120 * attempts))
    echo "retryable gateway pause; resuming durable cache/checkpoints in ${cooldown}s (attempt ${attempts}/${MAX_AUTO_RESUMES})" >&2
    sleep "$cooldown"
  done
}

PILOT_OUT="$RUN_ROOT/pilot-10"
PILOT_MANIFEST="$CACHE_ROOT/manifests/ragtruth-qa-entropy-10.json"
GLOBAL_DEADLINE_EPOCH=$(( $(date +%s) + MAX_WALL_SECONDS ))
# A failed 10-QA measurement should not consume the full unattended-run bound.
PILOT_DEADLINE_EPOCH=$(( $(date +%s) + 21600 ))
if (( PILOT_DEADLINE_EPOCH > GLOBAL_DEADLINE_EPOCH )); then
  PILOT_DEADLINE_EPOCH="$GLOBAL_DEADLINE_EPOCH"
fi
run_resilient "$PILOT_OUT" "$PILOT_MANIFEST" 10 "$PILOT_DEADLINE_EPOCH"

REMAINING_WALL_SECONDS=$(( GLOBAL_DEADLINE_EPOCH - $(date +%s) ))
(( REMAINING_WALL_SECONDS > 0 )) || { echo "No live wall-clock budget remains after the 10-QA pilot" >&2; exit 76; }
TARGET_SOURCES="$("$PYTHON" - "$PILOT_OUT/run_metadata.json" "$REMAINING_WALL_SECONDS" <<'PY'
import json
import math
import sys
metadata = json.load(open(sys.argv[1], encoding="utf-8"))
elapsed = float(metadata["elapsed_seconds"])
done = int(metadata["sources_completed"])
if metadata.get("state") != "completed" or done != 10 or elapsed <= 0:
    raise SystemExit("10-QA pilot did not finish completely; refusing to estimate a larger run")
# Reserve 10% for run-start, checkpoint, report and filesystem overhead.
capacity = int((float(sys.argv[2]) * 0.90) / (elapsed / done))
target = min(750, max(10, (capacity // 10) * 10))
print(target)
PY
)"
printf '%s\n' "$TARGET_SOURCES" > "$RUN_ROOT/selected-target-sources.txt"
echo "[semantic-entropy] measured pilot selected target=$TARGET_SOURCES QA sources" >&2

MAIN_OUT="$RUN_ROOT/main-$TARGET_SOURCES"
MAIN_MANIFEST="$CACHE_ROOT/manifests/ragtruth-qa-entropy-$TARGET_SOURCES.json"
run_resilient "$MAIN_OUT" "$MAIN_MANIFEST" "$TARGET_SOURCES" "$GLOBAL_DEADLINE_EPOCH"

set +e
HF_HOME="$LOCAL_ROOT/huggingface" TRANSFORMERS_CACHE="$LOCAL_ROOT/huggingface/transformers" \
  TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 \
  "$PYTHON" "$ROOT/scripts/run_ragtruth_semantic_entropy.py" \
    --config "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --output-dir "$MAIN_OUT/replay" \
    --manifest "$MAIN_MANIFEST" --total-sources "$TARGET_SOURCES" --seed 42 --n-bootstrap 1000 \
    --cache-only --recompute-from-cache > >(tee -a "$RUN_ROOT/stdout.log") 2> >(tee -a "$RUN_ROOT/stderr.log" >&2)
replay_status=$?
set -e
test "$replay_status" -eq 0 || exit "$replay_status"
cmp "$MAIN_OUT/scores.jsonl" "$MAIN_OUT/replay/scores.jsonl"

"$PYTHON" - "$RUN_ROOT/run_metadata.json" "$PILOT_OUT/run_metadata.json" "$MAIN_OUT/run_metadata.json" "$TARGET_SOURCES" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
pilot = json.load(open(sys.argv[2], encoding="utf-8"))
main = json.load(open(sys.argv[3], encoding="utf-8"))
payload = {
    "protocol": "local-ragtruth-semantic-entropy-launch-v1",
    "state": "completed",
    "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "pilot_sources": 10,
    "pilot_elapsed_seconds": pilot.get("elapsed_seconds"),
    "target_sources": int(sys.argv[4]),
    "main_elapsed_seconds": main.get("elapsed_seconds"),
    "main_manifest_sha256": main.get("manifest_sha256"),
    "main_usage": main.get("usage"),
    "cache_replay_verified": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "[ok] RAGTruth SemanticEntropy pilot, bounded evaluation, and cache-only replay completed"
