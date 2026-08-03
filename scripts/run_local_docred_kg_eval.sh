#!/usr/bin/env bash
# Fixed, resumable local DocRED KGGen evaluation: CPU -> Cloud Run -> Vertex AI.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LOCAL_DOCRED_PYTHON:-$ROOT/.venv/bin/python}"
LOCAL_ROOT="${HALLU_DOCRED_ROOT:-/Volumes/mySSD/hallu_smiles/docred-kggen}"
GATEWAY_URL="${HALLU_GATEWAY_URL:-https://hallu-vertex-gateway-453887629111.europe-west4.run.app}"
DOCRED_BUDGET_EUR="${DOCRED_BUDGET_EUR:-10.5}"
MIN_FREE_GIB="${HALLU_DOCRED_MIN_FREE_GIB:-20}"
EMBEDDING_PATH="${HALLU_DOCRED_EMBEDDING_PATH:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41}"
RUN_ID="${HALLU_DOCRED_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT=""
AUTH_CONFIG=""
MONITOR_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_local_docred_kg_eval.sh [--preflight]

The paid run reads HALLU_GATEWAY_API_KEY from macOS Keychain, never from a file
or command-line argument. It must be started inside tmux or screen; the script uses
caffeinate and writes its data, checkpoints, snapshots, archives and builds
under HALLU_DOCRED_ROOT (default: /Volumes/mySSD/hallu_smiles/docred-kggen).
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

preflight() {
  test -x "$PYTHON" || { echo "Python venv missing: $PYTHON" >&2; return 2; }
  command -v caffeinate >/dev/null || { echo "caffeinate is required on macOS" >&2; return 2; }
  if ! command -v tmux >/dev/null && ! command -v screen >/dev/null; then
    echo "[warn] tmux and screen are unavailable; use scripts/start_local_docred_kg_eval.sh for the nohup fallback" >&2
  fi
  command -v curl >/dev/null || { echo "curl is required" >&2; return 2; }
  command -v xelatex >/dev/null || { echo "xelatex is required to build the final report" >&2; return 2; }
  mkdir -p "$LOCAL_ROOT"
  local free_kib minimum_kib
  free_kib="$(disk_available_kib)"
  minimum_kib=$((MIN_FREE_GIB * 1024 * 1024))
  (( free_kib >= minimum_kib )) || {
    echo "Insufficient external-disk space: need ${MIN_FREE_GIB} GiB free under $LOCAL_ROOT" >&2
    return 2
  }
  test -f "$EMBEDDING_PATH/config.json" || {
    echo "Offline S-BERT snapshot is missing: $EMBEDDING_PATH" >&2
    return 2
  }
  HF_HUB_OFFLINE=1 "$PYTHON" - "$EMBEDDING_PATH" <<'PY'
import sys
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(sys.argv[1], local_files_only=True, device="cpu")
if model.get_sentence_embedding_dimension() <= 0:
    raise SystemExit("invalid S-BERT embedding dimension")
PY
  load_gateway_key
  unset HALLU_GATEWAY_API_KEY
  echo "[ok] local DocRED preflight passed; no gateway request was sent"
}

archive_artifacts() {
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]] || return 0
  "$PYTHON" "$ROOT/scripts/archive_local_docred_kg_eval.py" \
    --run-root "$RUN_ROOT" --archive "$LOCAL_ROOT/archives/$(basename "$RUN_ROOT").tar.gz"
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
  if ! archive_artifacts; then
    status=1
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
  preflight
  exit $?
fi

if [[ "${HALLU_DOCRED_CAFFEINATED:-}" != "1" ]]; then
  export HALLU_DOCRED_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0"
fi

[[ -n "${TMUX:-}" || -n "${STY:-}" || "${HALLU_DOCRED_DETACHED:-}" == "1" ]] || {
  echo "Refusing paid run outside tmux or screen. Use scripts/start_local_docred_kg_eval.sh." >&2
  exit 2
}
preflight

RUN_ROOT="$LOCAL_ROOT/runs/vertex-cpu-docred-kg-artifacts-$RUN_ID"
test ! -e "$RUN_ROOT" || { echo "run root already exists: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT/archives"
printf '%s\n' "$$" > "$RUN_ROOT/runner.pid"
trap cleanup EXIT

load_gateway_key
AUTH_CONFIG="$(mktemp "$RUN_ROOT/.gateway-curl.XXXXXX")"
chmod 600 "$AUTH_CONFIG"
printf 'header = "Authorization: Bearer %s"\n' "$HALLU_GATEWAY_API_KEY" > "$AUTH_CONFIG"

DATASET_REVISION="7985b4e0371e6c61a756feb41b7b27becf71c666"
DATA_DIR="$LOCAL_ROOT/data/thunlp-docred-$DATASET_REVISION"
GATEWAY_RAW="$RUN_ROOT/gateway-manifest.raw.json"
GATEWAY_MANIFEST="$RUN_ROOT/gateway-manifest.json"
RUNTIME_MANIFEST="$RUN_ROOT/local-runtime-manifest.json"
RUNTIME_CONFIG="$RUN_ROOT/runtime-config.yaml"
RUNTIME_IDENTITY="$RUN_ROOT/runtime-config-identity.json"

# DocRED files are a one-time public download. All subsequent data and S-BERT
# reads are offline, and every KG cache is outside the repository.
env -u HF_HUB_OFFLINE "$PYTHON" "$ROOT/scripts/fetch_docred_data.py" \
  --output-dir "$DATA_DIR" > "$RUN_ROOT/dataset-materialization.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42

curl --fail --silent --show-error --config "$AUTH_CONFIG" \
  --output "$GATEWAY_RAW" "$GATEWAY_URL/v1/hallu/manifest"
"$PYTHON" "$ROOT/scripts/validate_vertex_gateway_manifest.py" \
  --manifest "$GATEWAY_RAW" --logical-model "$("$PYTHON" - "$ROOT/config.yaml" <<'PY'
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["llm"]["model"])
PY
)" --output "$GATEWAY_MANIFEST"
rm -f "$GATEWAY_RAW"
AUTH_CONFIG=""
rm -f "$RUN_ROOT/.gateway-curl."* || true

GATEWAY_MANIFEST_SHA256="$("$PYTHON" - "$GATEWAY_MANIFEST" <<'PY'
import json
import sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
"$PYTHON" "$ROOT/scripts/write_local_vertex_runtime_manifest.py" \
  --embedding-model-path "$EMBEDDING_PATH" --output "$RUNTIME_MANIFEST" > "$RUN_ROOT/local-runtime-identity.json"
"$PYTHON" "$ROOT/scripts/make_local_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$GATEWAY_URL" --local-runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-model-path "$EMBEDDING_PATH" --output "$RUNTIME_CONFIG" \
  --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$RUN_ROOT/cache-placeholder" \
  --max-tokens 4096 --extraction-max-tokens-ceiling 8192 --concurrency 1 \
  --max-retries 0 --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --rate-limit-cooldown-max-s 900 > "$RUNTIME_IDENTITY"

RUNTIME_FINGERPRINT="$("$PYTHON" - "$RUNTIME_IDENTITY" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime_fingerprint"])
PY
)"
CACHE_FINGERPRINT="${RUNTIME_FINGERPRINT#vertex-gateway:}"
CHECKPOINT_ROOT="$LOCAL_ROOT/checkpoints/docred-kg-v1-${GATEWAY_MANIFEST_SHA256}-${CACHE_FINGERPRINT}"
mkdir -p "$CHECKPOINT_ROOT"
"$PYTHON" "$ROOT/scripts/make_local_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$GATEWAY_MANIFEST" \
  --gateway-url "$GATEWAY_URL" --local-runtime-manifest "$RUNTIME_MANIFEST" \
  --embedding-model-path "$EMBEDDING_PATH" --output "$RUNTIME_CONFIG" \
  --data-dir "$DATA_DIR" --work-dir "$RUN_ROOT" --cache-root "$CHECKPOINT_ROOT" \
  --max-tokens 4096 --extraction-max-tokens-ceiling 8192 --concurrency 1 \
  --max-retries 0 --retry-backoff-base-s 5 --retry-backoff-max-s 60 \
  --rate-limit-cooldown-max-s 900 > "$RUNTIME_IDENTITY"

CHECKPOINT_IDENTITY="$CHECKPOINT_ROOT/checkpoint-identity.json"
"$PYTHON" - "$CHECKPOINT_IDENTITY" "$GATEWAY_MANIFEST_SHA256" "$RUNTIME_IDENTITY" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
runtime = json.load(open(sys.argv[3], encoding="utf-8"))
payload = {
    "protocol": "local-docred-kg-evaluation-checkpoint-v1",
    "dataset_repository": "thunlp/docred",
    "dataset_revision": "7985b4e0371e6c61a756feb41b7b27becf71c666",
    "gateway_manifest_sha256": sys.argv[2],
    "llm_runtime_fingerprint": runtime["runtime_fingerprint"],
    "llm_max_tokens": 4096,
    "extraction_max_tokens_ceiling": 8192,
    "concurrency": 1,
    "request_min_interval_s": 4.0,
}
if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
    raise SystemExit("local DocRED checkpoint identity mismatch")
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
cp "$CHECKPOINT_IDENTITY" "$RUN_ROOT/checkpoint-identity.json"
cp "$DATA_DIR/dataset-metadata.json" "$RUN_ROOT/dataset-metadata.json"

"$PYTHON" "$ROOT/scripts/monitor_local_docred_kg_eval.py" \
  --run-root "$RUN_ROOT" --pid "$$" --interval-seconds 900 > "$RUN_ROOT/local-monitor.log" 2>&1 &
MONITOR_PID=$!

PERSISTENT_MANIFEST="$CHECKPOINT_ROOT/docred_manifest.json"
LIVE_OUT="$RUN_ROOT/docred-live"
REPLAY_OUT="$RUN_ROOT/docred-replay"
set +e
"$PYTHON" "$ROOT/scripts/run_docred_kg_eval.py" \
  --config "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --output-dir "$LIVE_OUT" \
  --manifest "$PERSISTENT_MANIFEST" --stage all --budget-eur "$DOCRED_BUDGET_EUR" \
  > >(tee "$RUN_ROOT/docred.stdout.log") 2> >(tee "$RUN_ROOT/docred.stderr.log" >&2)
live_status=$?
set -e
cp "$PERSISTENT_MANIFEST" "$RUN_ROOT/docred_manifest.json" 2>/dev/null || true
if [[ "$live_status" -eq 75 ]]; then
  echo "[docred] budget guard stopped before another live request" >&2
  exit 75
fi
test "$live_status" -eq 0 || exit "$live_status"

"$PYTHON" - "$LIVE_OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
if metadata.get("state") != "completed":
    raise SystemExit("live DocRED stage did not complete")
if metrics.get("documents") != 200 or metrics.get("extraction_coverage") != 1.0:
    raise SystemExit("live DocRED extraction coverage is incomplete")
PY
"$PYTHON" "$ROOT/scripts/write_docred_cache_inventory.py" \
  --cache-root "$CHECKPOINT_ROOT" --output "$RUN_ROOT/cache-before-replay.json"
"$PYTHON" "$ROOT/scripts/run_docred_kg_eval.py" \
  --config "$RUNTIME_CONFIG" --data-dir "$DATA_DIR" --output-dir "$REPLAY_OUT" \
  --manifest "$PERSISTENT_MANIFEST" --stage replay --cache-only --budget-eur "$DOCRED_BUDGET_EUR" \
  >> "$RUN_ROOT/docred.stdout.log" 2>> "$RUN_ROOT/docred.stderr.log"
"$PYTHON" "$ROOT/scripts/verify_docred_cache_replay.py" \
  --live-dir "$LIVE_OUT" --replay-dir "$REPLAY_OUT"
"$PYTHON" "$ROOT/scripts/write_docred_cache_inventory.py" \
  --cache-root "$CHECKPOINT_ROOT" --output "$RUN_ROOT/cache-after-replay.json"
cmp "$RUN_ROOT/cache-before-replay.json" "$RUN_ROOT/cache-after-replay.json"
"$PYTHON" "$ROOT/scripts/write_docred_usage_counts.py" \
  --live-usage "$LIVE_OUT/usage.jsonl" --replay-usage "$REPLAY_OUT/usage.jsonl" \
  --output "$RUN_ROOT/usage-counts.json"

"$PYTHON" - "$RUN_ROOT/run_metadata.json" "$RUNTIME_IDENTITY" "$GATEWAY_MANIFEST_SHA256" "$DOCRED_BUDGET_EUR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

runtime = json.load(open(sys.argv[2], encoding="utf-8"))
Path(sys.argv[1]).write_text(json.dumps({
    "state": "completed",
    "mode": "local-cpu-vertex-docred-kg-evaluation",
    "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "gateway_manifest_sha256": sys.argv[3],
    "local_runtime_fingerprint": runtime["local_runtime_fingerprint"],
    "fixed_protocol": {
        "calibration_documents": 50,
        "smoke_documents_within_calibration": 10,
        "heldout_dev_documents": 200,
        "seed": 42,
        "concurrency": 1,
        "base_max_tokens": 4096,
        "adaptive_max_tokens_ceiling": 8192,
        "budget_max_eur": float(sys.argv[4]),
    },
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

ARCHIVE="$LOCAL_ROOT/archives/$(basename "$RUN_ROOT").tar.gz"
"$PYTHON" "$ROOT/scripts/archive_local_docred_kg_eval.py" --run-root "$RUN_ROOT" --archive "$ARCHIVE"
"$PYTHON" "$ROOT/scripts/write_docred_kg_results_tex.py" \
  --artifact "$ARCHIVE" --output "$ROOT/docs/docred-kg-extraction-results.tex"
mkdir -p "$RUN_ROOT/tex-build"
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$RUN_ROOT/tex-build" \
  "$ROOT/docs/docred-kg-extraction-results.tex" >/dev/null
echo "[ok] local DocRED evaluation, redacted archive, and XeLaTeX report completed"
