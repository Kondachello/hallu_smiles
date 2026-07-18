#!/usr/bin/env bash
# DataSphere GPU Job: shared read-only model -> strict QA pilot -> support QA pilot.
# Model/data staging happens separately from a c1.4 Jupyter session.  Do not add
# model downloads or runtime package installation here: both waste GPU units.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/hallu/server/bin/python}"
VLLM_BIN="${VLLM_BIN:-/opt/hallu/server/bin/vllm}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/opt/hallu/runtime-manifest.json}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/opt/hallu/models/all-MiniLM-L6-v2}"
EMBEDDING_MODEL_REVISION="${EMBEDDING_MODEL_REVISION:-1110a243fdf4706b3f48f1d95db1a4f5529b4d41}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Set the exact pushed source commit selected by the Job template}"
DATASPHERE_DOCKER_IMAGE_ID="${DATASPHERE_DOCKER_IMAGE_ID:?Set the immutable DataSphere Docker resource ID}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID to the Hugging Face model identifier}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the shared RAGTruth directory}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to this Job writable output directory}"
PORT="${VLLM_PORT:-8000}"
MIN_WORK_FREE_GB="${MIN_WORK_FREE_GB:-5}"
# FP16 works on both V100 and A100.  In particular, a V100 cannot serve the
# model's native BF16 dtype.  8192 tokens comfortably covers the 6000-char KG
# chunks while avoiding a needless 128K KV-cache reservation on a 32-GiB V100.
MODEL_DTYPE="${MODEL_DTYPE:-half}"
MODEL_MAX_MODEL_LEN="${MODEL_MAX_MODEL_LEN:-8192}"
# KGGen 0.4 otherwise requests up to 16k completion tokens.  The exact real
# relation probe proved that 256 truncates source 15138 before JSON validation;
# 1024 gives bounded 4x headroom inside 8192, and the exact gate verifies it.
KGGEN_MAX_TOKENS="${KGGEN_MAX_TOKENS:-1024}"
# The official QA method keeps KGGen LLM clustering enabled and unbounded.
# Supplying a positive cap is reserved for an explicitly labelled diagnostic,
# never for the final strict/support comparison.
KGGEN_CLUSTER_MAX_ITEMS="${KGGEN_CLUSTER_MAX_ITEMS:-}"
# KGGen 0.4 and DSPy share mutable state inside a backend instance.  Nested
# chunk/response thread pools produced a local-vLLM deadlock and hours of idle
# V100 time.  The Job is intentionally serial; vLLM remains the only GPU work.
KGGEN_CONCURRENCY="${KGGEN_CONCURRENCY:-1}"
CLUSTER_CONTEXT_MODE="source_text"
CLUSTER_CONTEXT_PROTOCOL="kggen-native-source-text-v1"
# vLLM 0.8.5 accepts only the bare backend enum at the server CLI. Request-level
# options disable XGrammar's unbounded syntactic-whitespace loops and forbid
# fallback; neither option changes the JSON Schema or its accepted documents.
STRUCTURED_OUTPUT_BACKEND="xgrammar"
GUIDED_DECODING_BACKEND="xgrammar"
STRUCTURED_OUTPUT_REQUEST_BACKEND="xgrammar:disable-any-whitespace,no-fallback"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# LiteLLM otherwise fetches an optional model-cost map from GitHub on its first
# import. Jobs need no cost pricing to call localhost, and an unreachable
# external endpoint must never block KGGen before the first real request.
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"
RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
MANIFEST="$RUN_ROOT/qa_pilot_manifest.json"
STRICT_OUT="$RUN_ROOT/strict"
SUPPORT_OUT="$RUN_ROOT/support"
VLLM_LOG="$RUN_ROOT/vllm.log"
GPU_LOG="$RUN_ROOT/gpu.csv"
METADATA="$RUN_ROOT/run_metadata.json"
UNITS_PER_SECOND="${DATASPHERE_UNITS_PER_SECOND:-}"
GPU_TIME_LIMIT_SECONDS="${GPU_TIME_LIMIT_SECONDS:-}"
QA_PILOT_LIMIT="${QA_PILOT_LIMIT:-}"
VLLM_PID=""
GPU_PID=""
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

# Keep dynamic shared-model resolution inside this runner rather than in the
# outer Job shell.  The runner's stdout/stderr are archived on every exit, so a
# Project-disk or interpreter startup failure has a concrete phase and a
# bounded timeout instead of leaving an allocated GPU with no diagnostic.
if [[ -z "$MODEL_PATH" ]]; then
  : "${DS_PROJECT_HOME:?DS_PROJECT_HOME is required with attach-project-disk}"
  echo "[startup] resolving active shared model path."
  MODEL_PATH="$(timeout --signal=TERM --kill-after=15s "${MODEL_PATH_RESOLVE_TIMEOUT_SECONDS:-60}" \
    "$CLIENT_PYTHON" -S "$ROOT/scripts/resolve_datasphere_shared_model.py" \
      --shared-root "$DS_PROJECT_HOME/hallu_smiles/shared" --model-id "$MODEL_ID")" || {
    echo "[startup] shared-model resolution failed or exceeded ${MODEL_PATH_RESOLVE_TIMEOUT_SECONDS:-60}s." >&2
    exit 2
  }
  export MODEL_PATH
  echo "[startup] shared model path resolved."
fi

cleanup() {
  local exit_code=$?
  [[ -n "$GPU_PID" ]] && kill "$GPU_PID" 2>/dev/null || true
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  wait "${GPU_PID:-}" 2>/dev/null || true
  wait "${VLLM_PID:-}" 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

# A DataSphere project secret is injected into Job environments.  The staging
# session is the only consumer of HF_TOKEN; remove it before any subprocesses.
unset HF_TOKEN || true
mkdir -p "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$RUN_ROOT/hf-home}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/hallu/models}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$RUN_ROOT/dspy-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$DSPY_CACHEDIR"
available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
required_kb=$((MIN_WORK_FREE_GB * 1024 * 1024))
if [[ -z "$available_kb" || "$available_kb" -lt "$required_kb" ]]; then
  echo "Need at least ${MIN_WORK_FREE_GB} GiB free in RUN_ROOT; found ${available_kb:-unknown} KiB." >&2
  exit 2
fi

test -x "$CLIENT_PYTHON" || { echo "client interpreter is absent: $CLIENT_PYTHON" >&2; exit 2; }
test -x "$SERVER_PYTHON" || { echo "server interpreter is absent: $SERVER_PYTHON" >&2; exit 2; }
test -x "$VLLM_BIN" || { echo "vLLM executable is absent: $VLLM_BIN" >&2; exit 2; }
test -f "$RUNTIME_MANIFEST" || { echo "runtime manifest is absent: $RUNTIME_MANIFEST" >&2; exit 2; }
test -f "$EMBEDDING_MODEL_PATH/config.json" || { echo "offline SBERT snapshot is absent: $EMBEDDING_MODEL_PATH" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is absent; this must run on a GPU Job." >&2; exit 2; }
cp "$RUNTIME_MANIFEST" "$RUN_ROOT/runtime-manifest.json"
cp /opt/hallu/manifests/server.freeze.txt "$RUN_ROOT/server.freeze.txt"
cp /opt/hallu/manifests/client.freeze.txt "$RUN_ROOT/client.freeze.txt"
"$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_shared_assets.py" \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" \
  --model-id "$MODEL_ID" --report "$RUN_ROOT/shared-assets-preflight.json"
CUDA_VISIBLE_DEVICES=0 "$SERVER_PYTHON" "$ROOT/scripts/check_datasphere_gpu_runtime.py" \
  --expected-torch-cuda 11.8 --expected-device-substring V100 \
  --report "$RUN_ROOT/gpu-runtime.json"
"$CLIENT_PYTHON" - "$RUN_ROOT/shared-assets-preflight.json" "$RUN_ROOT/model_revision.txt" <<'PY'
import json
import sys
from pathlib import Path

revision = json.load(open(sys.argv[1], encoding="utf-8"))["model_revision"]
Path(sys.argv[2]).write_text(revision + "\n", encoding="utf-8")
PY
MODEL_REVISION="$(<"$RUN_ROOT/model_revision.txt")"
RUNTIME_IDENTITY="$RUN_ROOT/runtime-identity.json"
RUNTIME_FINGERPRINT="$("$CLIENT_PYTHON" - \
  "$RUNTIME_MANIFEST" "$RUNTIME_IDENTITY" "$EXPECTED_SOURCE_COMMIT" \
  "$DATASPHERE_DOCKER_IMAGE_ID" "$EMBEDDING_MODEL_PATH" "$EMBEDDING_MODEL_REVISION" \
  "$MODEL_DTYPE" "$MODEL_MAX_MODEL_LEN" "$GUIDED_DECODING_BACKEND" \
  "$STRUCTURED_OUTPUT_BACKEND" "$GPU_MEMORY_UTILIZATION" "$KGGEN_CONCURRENCY" \
  "$MODEL_REVISION" "$KGGEN_MAX_TOKENS" "$STRUCTURED_OUTPUT_REQUEST_BACKEND" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_protocol = "hallu-datasphere-vllm085-cu118-v1"
if payload.get("source_commit") != sys.argv[3]:
    raise SystemExit(
        f"Docker/source drift: image={payload.get('source_commit')} job={sys.argv[3]}"
    )
if payload.get("runtime_protocol") != expected_protocol:
    raise SystemExit(f"unexpected runtime protocol: {payload.get('runtime_protocol')!r}")
if payload.get("embedding_path") != sys.argv[5]:
    raise SystemExit("Docker embedding path differs from the Job configuration")
if payload.get("embedding_revision") != sys.argv[6]:
    raise SystemExit("Docker embedding revision differs from the Job configuration")
if payload.get("server_runtime", {}).get("torch_cuda") != "11.8":
    raise SystemExit("Docker manifest is not the required CUDA 11.8 build")
if payload.get("client_runtime", {}).get("torch_cuda") is not None:
    raise SystemExit("Docker client environment is not CPU-only")
server_launch = {
    "dtype": sys.argv[7],
    "max_model_len": int(sys.argv[8]),
    "guided_decoding_backend": sys.argv[9],
    "structured_output_transport": "response_format",
    "structured_output_backend": sys.argv[10],
    "gpu_memory_utilization": sys.argv[11],
    "max_num_seqs": 1,
    "enforce_eager": True,
    "seed": 42,
    "vllm_use_v1": "0",
    "kggen_concurrency": int(sys.argv[12]),
    "model_revision": sys.argv[13],
    "max_tokens": int(sys.argv[14]),
    "guided_decoding_request_backend": sys.argv[15],
    "xgrammar_any_whitespace": False,
    "cluster_context_mode": "source_text",
    "cluster_context_protocol": "kggen-native-source-text-v1",
}
canonical = json.dumps(server_launch, sort_keys=True, separators=(",", ":"))
generation_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
runtime_fingerprint = (
    f"{sys.argv[4]}:{payload['runtime_fingerprint']}:{generation_fingerprint}"
)
identity = {
    "source_commit": sys.argv[3],
    "datasphere_docker_image_id": sys.argv[4],
    "runtime_protocol": expected_protocol,
    "image_runtime_fingerprint": payload["runtime_fingerprint"],
    "generation_fingerprint": generation_fingerprint,
    "runtime_fingerprint": runtime_fingerprint,
    "server_launch": server_launch,
    "guided_decoding_request_backend": sys.argv[15],
    "xgrammar_any_whitespace": False,
    "cluster_context_mode": "source_text",
    "cluster_context_protocol": "kggen-native-source-text-v1",
}
Path(sys.argv[2]).write_text(
    json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(runtime_fingerprint)
PY
)"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$CLIENT_PYTHON" - "$METADATA" "$started" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$GPU_TIME_LIMIT_SECONDS" "$DATASPHERE_DOCKER_IMAGE_ID" "$RUNTIME_FINGERPRINT" "$RUNTIME_IDENTITY" <<'PY'
import json
import sys
from pathlib import Path

identity = json.load(open(sys.argv[9], encoding="utf-8"))
Path(sys.argv[1]).write_text(json.dumps({
    "started_at_utc": sys.argv[2],
    "state": "started",
    "model_id": sys.argv[3],
    "shared_model_path": sys.argv[4],
    "model_revision": sys.argv[5],
    "gpu_time_limit_seconds": int(sys.argv[6]) if sys.argv[6] else None,
    "datasphere_docker_image_id": sys.argv[7],
    "runtime_fingerprint": sys.argv[8],
    "source_commit": identity["source_commit"],
    "runtime_protocol": identity["runtime_protocol"],
    "server_launch": identity["server_launch"],
    "structured_output_transport": "response_format",
    "structured_output_backend": "xgrammar",
    "guided_decoding_request_backend": identity[
        "guided_decoding_request_backend"
    ],
    "xgrammar_any_whitespace": False,
    "cluster_context_mode": identity["cluster_context_mode"],
    "cluster_context_protocol": identity["cluster_context_protocol"],
    "runs": ["strict", "support", "cache-only-strict", "cache-only-support"],
}, indent=2) + "\n", encoding="utf-8")
PY

export OPENAI_API_KEY="${OPENAI_API_KEY:-local-datasphere-key}"
runtime_config_args=(
  --base-config "$ROOT/config.yaml"
  --output "$RUNTIME_CONFIG"
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --runtime-fingerprint "$RUNTIME_FINGERPRINT"
  --api-base "http://127.0.0.1:${PORT}/v1"
  --data-dir "$DATA_DIR"
  --max-tokens "$KGGEN_MAX_TOKENS"
  --cluster-context-mode "$CLUSTER_CONTEXT_MODE"
  --structured-output-transport response_format
  --structured-output-backend "$STRUCTURED_OUTPUT_BACKEND"
  --structured-output-request-backend "$STRUCTURED_OUTPUT_REQUEST_BACKEND"
  --embedding-model-path "$EMBEDDING_MODEL_PATH"
  --embedding-model-revision "$EMBEDDING_MODEL_REVISION"
  --explicit-clustering
  --concurrency "$KGGEN_CONCURRENCY"
  --serial-chunking
  --work-dir "$RUN_ROOT"
)
if [[ -n "$KGGEN_CLUSTER_MAX_ITEMS" ]]; then
  [[ "$KGGEN_CLUSTER_MAX_ITEMS" =~ ^[1-9][0-9]*$ ]] || { echo "KGGEN_CLUSTER_MAX_ITEMS must be positive when set." >&2; exit 2; }
  runtime_config_args+=(--cluster-max-items "$KGGEN_CLUSTER_MAX_ITEMS")
fi
if [[ -n "$QA_PILOT_LIMIT" ]]; then
  [[ "$QA_PILOT_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "QA_PILOT_LIMIT must be positive when set." >&2; exit 2; }
  runtime_config_args+=(--cluster-min-retention-ratio 0.20)
fi
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/make_datasphere_runtime_config.py" "${runtime_config_args[@]}"

CUDA_VISIBLE_DEVICES=0 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype "$MODEL_DTYPE" --max-model-len "$MODEL_MAX_MODEL_LEN" \
  --guided-decoding-backend "$GUIDED_DECODING_BACKEND" \
  --max-num-seqs 1 --enforce-eager --seed 42 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

vllm_ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    vllm_ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    vllm_status=1
    wait "$VLLM_PID" || vllm_status=$?
    echo "vLLM exited before its health endpoint became ready (status=$vllm_status)." >&2
    tail -n 240 "$VLLM_LOG" >&2 || true
    exit "$vllm_status"
  fi
  sleep 2
done
if [[ "$vllm_ready" != 1 ]]; then
  echo "vLLM health endpoint did not become ready within 240 seconds." >&2
  tail -n 240 "$VLLM_LOG" >&2 || true
  exit 124
fi
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vllm_completion.py" \
  --port "$PORT" --model-id "$MODEL_ID"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 10 >"$GPU_LOG" 2>&1 &
GPU_PID=$!

echo "[probe] verifying the exact KGGen relation schema through native response_format."
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_vllm_guided_json.py" \
  --port "$PORT" --model-id "$MODEL_ID" \
  --data-dir "$DATA_DIR" \
  --timeout "${STRUCTURED_OUTPUT_PROBE_TIMEOUT_SECONDS:-90}" \
  --repeat 1 \
  --max-tokens "$KGGEN_MAX_TOKENS" \
  --request-backend "$STRUCTURED_OUTPUT_REQUEST_BACKEND" \
  --report "$RUN_ROOT/vllm-response-format-probe.json"

# A plain completion is not enough: KGGen calls vLLM through DSPy's typed
# output adapter and optional LLM clustering. Both are required for a faithful
# KGGen run, so the probe keeps clustering on. GNU timeout protects the budget
# if the client blocks.
echo "[probe] checking KGGen/DSPy structured extraction before the QA pilot."
timeout --signal=TERM --kill-after=30s "${KGGEN_PROBE_TIMEOUT_SECONDS:-180}" \
  env CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_kggen_probe.py" \
  --port "$PORT" --model-id "$MODEL_ID" \
  --timeout "${KGGEN_PROBE_REQUEST_TIMEOUT_SECONDS:-60}" \
  --max-tokens "${KGGEN_PROBE_MAX_TOKENS:-$KGGEN_MAX_TOKENS}" \
  --cluster \
  --structured-output-transport response_format \
  --request-backend "$STRUCTURED_OUTPUT_REQUEST_BACKEND" \
  --report "$RUN_ROOT/kggen-probe.json"

echo "[probe] checking the support verifier's closed verdict schema."
timeout --signal=TERM --kill-after=30s "${VERIFIER_PROBE_TIMEOUT_SECONDS:-120}" \
  env CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_verifier_probe.py" \
  --config "$RUNTIME_CONFIG" \
  --report "$RUN_ROOT/verifier-probe.json"

# The synthetic probe above validates the typed-output protocol.  This second
# bounded probe exercises the exact first selected RAGTruth reference graph,
# prints phase-by-phase progress and seeds its content-addressed context cache.
# Do not spend a full pilot Job if this real input cannot clear extraction.
echo "[probe] checking first selected QA reference with post-stage diagnostics."
timeout --signal=TERM --kill-after=30s "${KGGEN_REFERENCE_PROBE_TIMEOUT_SECONDS:-180}" \
  env CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/check_datasphere_qa_reference_probe.py" \
  --config "$RUNTIME_CONFIG" \
  --report "$RUN_ROOT/qa-reference-probe.json"

run_extraction_with_gpu_watchdog() {
  # The liveness condition applies only to live KG extraction. Scoring and
  # reporting are intentionally CPU-heavy, so monitoring the whole pipeline
  # would produce false positives after extraction has completed.
  local idle_limit_seconds="${GPU_IDLE_ABORT_SECONDS:-600}"
  local poll_seconds=30
  local required_samples=$((idle_limit_seconds / 10))
  local child_pid=""
  if [[ ! "$idle_limit_seconds" =~ ^[1-9][0-9]*$ ]] || (( idle_limit_seconds < 60 )); then
    echo "GPU_IDLE_ABORT_SECONDS must be an integer of at least 60." >&2
    return 2
  fi

  "$@" &
  child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$poll_seconds"
    if ! kill -0 "$child_pid" 2>/dev/null; then
      break
    fi
    if tail -n "$required_samples" "$GPU_LOG" | awk -F ',' -v need="$required_samples" '
      NF >= 2 {
        samples++
        value = $2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        if (value + 0 > 0) active = 1
      }
      END { exit !(samples >= need && !active) }
    '; then
      echo "[watchdog] no GPU activity for ${idle_limit_seconds}s during KG extraction; terminating extraction." >&2
      kill -TERM "$child_pid" 2>/dev/null || true
      wait "$child_pid" || true
      return 124
    fi
  done
  wait "$child_pid"
}

start_epoch="$(date +%s)"
echo "[watchdog] strict KG extraction will abort after ${GPU_IDLE_ABORT_SECONDS:-600}s without GPU activity."
extract_args=(
  --config "$RUNTIME_CONFIG" --stage extract --relation-mode strict --qa-pilot
  --qa-pilot-manifest-out "$MANIFEST" --output-dir "$STRICT_OUT"
)
if [[ -n "$QA_PILOT_LIMIT" ]]; then
  extract_args+=(--qa-pilot-limit "$QA_PILOT_LIMIT")
fi
run_extraction_with_gpu_watchdog env CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/run.py" "${extract_args[@]}"
require_complete_extraction() {
  local output_dir="$1"
  local failures="$output_dir/failed_extractions.jsonl"
  local summary="$output_dir/extraction_summary.json"
  if [[ -s "$failures" ]]; then
    echo "[error] KG extraction was incomplete; refusing to mark this run successful." >&2
    cat "$failures" >&2
    return 1
  fi
  "$CLIENT_PYTHON" - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("protocol") != "hallu-extraction-summary-v1":
    raise SystemExit("extraction summary protocol mismatch")
if summary.get("status") != "ready" or summary.get("failures") != []:
    raise SystemExit("extraction summary is not complete")
if summary.get("completed_records") != summary.get("expected_records"):
    raise SystemExit("extraction summary record mismatch")
PY
}
require_complete_extraction "$STRICT_OUT"
if [[ -n "$QA_PILOT_LIMIT" ]]; then
  end_epoch="$(date +%s)"
  "$CLIENT_PYTHON" - "$METADATA" "$started" "$((end_epoch - start_epoch))" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$UNITS_PER_SECOND" "$GPU_TIME_LIMIT_SECONDS" "$QA_PILOT_LIMIT" "$DATASPHERE_DOCKER_IMAGE_ID" "$RUNTIME_FINGERPRINT" "$EXPECTED_SOURCE_COMMIT" "$GUIDED_DECODING_BACKEND" "$STRUCTURED_OUTPUT_REQUEST_BACKEND" "$CLUSTER_CONTEXT_MODE" "$CLUSTER_CONTEXT_PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

units_per_second = float(sys.argv[7]) if sys.argv[7] else None
wall_clock_seconds = int(sys.argv[3])
Path(sys.argv[1]).write_text(json.dumps({
    "started_at_utc": sys.argv[2],
    "state": "completed",
    "mode": "cluster-runtime-probe",
    "qa_pilot_limit": int(sys.argv[9]),
    "wall_clock_seconds": wall_clock_seconds,
    "model_id": sys.argv[4],
    "configured_units_per_second": units_per_second,
    "estimated_units_spent": wall_clock_seconds * units_per_second if units_per_second else None,
    "shared_model_path": sys.argv[5],
    "model_revision": sys.argv[6],
    "gpu_time_limit_seconds": int(sys.argv[8]) if sys.argv[8] else None,
    "datasphere_docker_image_id": sys.argv[10],
    "runtime_fingerprint": sys.argv[11],
    "source_commit": sys.argv[12],
    "structured_output_transport": "response_format",
    "structured_output_backend": "xgrammar",
    "guided_decoding_backend": sys.argv[13],
    "guided_decoding_request_backend": sys.argv[14],
    "xgrammar_any_whitespace": False,
    "cluster_context_mode": sys.argv[15],
    "cluster_context_protocol": sys.argv[16],
    "runs": ["strict-extract"],
}, indent=2) + "\n", encoding="utf-8")
PY
  echo "[done] cluster runtime probe extracted ${QA_PILOT_LIMIT} fixed QA rows; strict/support scoring was intentionally not run."
  exit 0
fi
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode strict --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$STRICT_OUT" --kg-cache-only
find "$RUN_ROOT/cache/kg" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/kg-cache-before-support.sha256"
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode support --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$SUPPORT_OUT" --kg-cache-only
find "$RUN_ROOT/cache/kg" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/kg-cache-after-support.sha256"
cmp "$RUN_ROOT/kg-cache-before-support.sha256" "$RUN_ROOT/kg-cache-after-support.sha256"
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/scripts/compare_qa_pilot_results.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --output "$RUN_ROOT/comparison.json"

# Prove that the archived graph/verdict caches are complete.  The endpoint is
# stopped first; a missing artifact must fail rather than silently infer.
find "$RUN_ROOT/cache" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-before.sha256"
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
REPLAY_STRICT="$RUN_ROOT/cache-replay/strict"
REPLAY_SUPPORT="$RUN_ROOT/cache-replay/support"
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode strict --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$REPLAY_STRICT" --cache-only
CUDA_VISIBLE_DEVICES="" "$CLIENT_PYTHON" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode support --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$REPLAY_SUPPORT" --cache-only
cmp "$STRICT_OUT/metrics.csv" "$REPLAY_STRICT/metrics.csv"
cmp "$SUPPORT_OUT/metrics.csv" "$REPLAY_SUPPORT/metrics.csv"
find "$RUN_ROOT/cache" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/cache-after.sha256"
cmp "$RUN_ROOT/cache-before.sha256" "$RUN_ROOT/cache-after.sha256"
end_epoch="$(date +%s)"

"$CLIENT_PYTHON" - "$METADATA" "$started" "$((end_epoch - start_epoch))" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$UNITS_PER_SECOND" "$GPU_TIME_LIMIT_SECONDS" "$DATASPHERE_DOCKER_IMAGE_ID" "$RUNTIME_FINGERPRINT" "$EXPECTED_SOURCE_COMMIT" "$GUIDED_DECODING_BACKEND" "$STRUCTURED_OUTPUT_REQUEST_BACKEND" "$CLUSTER_CONTEXT_MODE" "$CLUSTER_CONTEXT_PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

units_per_second = float(sys.argv[7]) if sys.argv[7] else None
wall_clock_seconds = int(sys.argv[3])
Path(sys.argv[1]).write_text(json.dumps({
    "started_at_utc": sys.argv[2],
    "state": "completed",
    "wall_clock_seconds": wall_clock_seconds,
    "model_id": sys.argv[4],
    "configured_units_per_second": units_per_second,
    "estimated_units_spent": wall_clock_seconds * units_per_second if units_per_second else None,
    "shared_model_path": sys.argv[5],
    "model_revision": sys.argv[6],
    "gpu_time_limit_seconds": int(sys.argv[8]) if sys.argv[8] else None,
    "datasphere_docker_image_id": sys.argv[9],
    "runtime_fingerprint": sys.argv[10],
    "source_commit": sys.argv[11],
    "structured_output_transport": "response_format",
    "structured_output_backend": "xgrammar",
    "guided_decoding_backend": sys.argv[12],
    "guided_decoding_request_backend": sys.argv[13],
    "xgrammar_any_whitespace": False,
    "cluster_context_mode": sys.argv[14],
    "cluster_context_protocol": sys.argv[15],
    "runs": ["strict", "support", "cache-only-strict", "cache-only-support"],
}, indent=2) + "\n", encoding="utf-8")
PY
