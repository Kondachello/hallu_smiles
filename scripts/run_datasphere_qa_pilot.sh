#!/usr/bin/env bash
# DataSphere GPU Job: shared read-only model -> strict QA pilot -> support QA pilot.
# Model/data staging happens separately from a c1.4 Jupyter session.  Do not add
# model downloads or runtime package installation here: both waste GPU units.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID to the Hugging Face model identifier}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a ready shared read-only model directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the shared RAGTruth directory}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to this Job writable output directory}"
PORT="${VLLM_PORT:-8000}"
MIN_WORK_FREE_GB="${MIN_WORK_FREE_GB:-5}"
# FP16 works on both V100 and A100.  In particular, a V100 cannot serve the
# model's native BF16 dtype.  8192 tokens comfortably covers the 6000-char KG
# chunks while avoiding a needless 128K KV-cache reservation on a 32-GiB V100.
MODEL_DTYPE="${MODEL_DTYPE:-half}"
MODEL_MAX_MODEL_LEN="${MODEL_MAX_MODEL_LEN:-8192}"
# KGGen 0.4 otherwise requests up to 16k completion tokens.  256 is sufficient
# for the selected short QA contexts and bounds an 8B model from returning a
# huge entity list which makes KGGen's dynamic Pydantic schemas CPU-bound.
KGGEN_MAX_TOKENS="${KGGEN_MAX_TOKENS:-256}"
# The official QA method keeps KGGen LLM clustering enabled and unbounded.
# Supplying a positive cap is reserved for an explicitly labelled diagnostic,
# never for the final strict/support comparison.
KGGEN_CLUSTER_MAX_ITEMS="${KGGEN_CLUSTER_MAX_ITEMS:-}"
# KGGen 0.4 and DSPy share mutable state inside a backend instance.  Nested
# chunk/response thread pools produced a local-vLLM deadlock and hours of idle
# V100 time.  The Job is intentionally serial; vLLM remains the only GPU work.
KGGEN_CONCURRENCY="${KGGEN_CONCURRENCY:-1}"
# lm-format-enforcer and vLLM 0.6.3's structured backends fail to resolve Pydantic ``$defs`` inside
# KGGen's nested Relation output schema.  The DSPy adapter therefore sends the
# mathematically equivalent inline schema; Outlines constrains that complete
# grammar.  Its broken pyairports dependency is supplied by the checked-in
# JSON-only runtime shim below.
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-outlines}"
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
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$RUN_ROOT/sentence-transformers}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$RUN_ROOT/dspy-cache}"
export PYTHONPATH="$ROOT/datasphere/runtime_shims${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME" "$DSPY_CACHEDIR"
available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
required_kb=$((MIN_WORK_FREE_GB * 1024 * 1024))
if [[ -z "$available_kb" || "$available_kb" -lt "$required_kb" ]]; then
  echo "Need at least ${MIN_WORK_FREE_GB} GiB free in RUN_ROOT; found ${available_kb:-unknown} KiB." >&2
  exit 2
fi

command -v vllm >/dev/null || { echo "vllm is absent; use the Job's manual requirements environment." >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is absent; this must run on a GPU Job." >&2; exit 2; }
"$PYTHON_BIN" "$ROOT/scripts/check_datasphere_shared_assets.py" \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" \
  --model-id "$MODEL_ID" --report "$RUN_ROOT/shared-assets-preflight.json"
"$PYTHON_BIN" "$ROOT/scripts/check_datasphere_gpu_runtime.py" \
  --report "$RUN_ROOT/gpu-runtime.json"
"$PYTHON_BIN" "$ROOT/scripts/patch_datasphere_lmfe_bool_schema.py" \
  --report "$RUN_ROOT/lmfe-bool-schema-patch.json"
"$PYTHON_BIN" "$ROOT/scripts/check_datasphere_outlines_backend.py" \
  --report "$RUN_ROOT/outlines-backend.json"
"$PYTHON_BIN" - "$RUN_ROOT/shared-assets-preflight.json" "$RUN_ROOT/model_revision.txt" <<'PY'
import json
import sys
from pathlib import Path

revision = json.load(open(sys.argv[1], encoding="utf-8"))["model_revision"]
Path(sys.argv[2]).write_text(revision + "\n", encoding="utf-8")
PY
MODEL_REVISION="$(<"$RUN_ROOT/model_revision.txt")"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON_BIN" - "$METADATA" "$started" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$GPU_TIME_LIMIT_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "started_at_utc": sys.argv[2],
    "state": "started",
    "model_id": sys.argv[3],
    "shared_model_path": sys.argv[4],
    "model_revision": sys.argv[5],
    "gpu_time_limit_seconds": int(sys.argv[6]) if sys.argv[6] else None,
    "runs": ["strict", "support"],
}, indent=2) + "\n", encoding="utf-8")
PY

export OPENAI_API_KEY="${OPENAI_API_KEY:-local-datasphere-key}"
runtime_config_args=(
  --base-config "$ROOT/config.yaml"
  --output "$RUNTIME_CONFIG"
  --model-id "$MODEL_ID"
  --api-base "http://127.0.0.1:${PORT}/v1"
  --data-dir "$DATA_DIR"
  --max-tokens "$KGGEN_MAX_TOKENS"
  --vllm-guided-json
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
fi
"$PYTHON_BIN" "$ROOT/scripts/make_datasphere_runtime_config.py" "${runtime_config_args[@]}"

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype "$MODEL_DTYPE" --max-model-len "$MODEL_MAX_MODEL_LEN" \
  --guided-decoding-backend "$GUIDED_DECODING_BACKEND" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
"$PYTHON_BIN" "$ROOT/scripts/check_datasphere_vllm_completion.py" \
  --port "$PORT" --model-id "$MODEL_ID"
echo "[probe] verifying vLLM native guided_json schema enforcement."
"$PYTHON_BIN" "$ROOT/scripts/check_datasphere_vllm_guided_json.py" \
  --port "$PORT" --model-id "$MODEL_ID" \
  --timeout "${GUIDED_JSON_PROBE_TIMEOUT_SECONDS:-60}" \
  --report "$RUN_ROOT/vllm-guided-json-probe.json"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 10 >"$GPU_LOG" 2>&1 &
GPU_PID=$!

# A plain completion is not enough: KGGen calls vLLM through DSPy's typed
# output adapter and optional LLM clustering. Both are required for a faithful
# KGGen run, so the probe keeps clustering on. GNU timeout protects the budget
# if the client blocks.
echo "[probe] checking KGGen/DSPy structured extraction before the QA pilot."
timeout --signal=TERM --kill-after=30s "${KGGEN_PROBE_TIMEOUT_SECONDS:-180}" \
  "$PYTHON_BIN" "$ROOT/scripts/check_datasphere_kggen_probe.py" \
  --port "$PORT" --model-id "$MODEL_ID" \
  --timeout "${KGGEN_PROBE_REQUEST_TIMEOUT_SECONDS:-60}" \
  --max-tokens "${KGGEN_PROBE_MAX_TOKENS:-256}" \
  --cluster \
  --vllm-guided-json \
  --report "$RUN_ROOT/kggen-probe.json"

# The synthetic probe above validates the typed-output protocol.  This second
# bounded probe exercises the exact first selected RAGTruth reference graph,
# prints phase-by-phase progress and seeds its content-addressed context cache.
# Do not spend a full pilot Job if this real input cannot clear extraction.
echo "[probe] checking first selected QA reference with post-stage diagnostics."
timeout --signal=TERM --kill-after=30s "${KGGEN_REFERENCE_PROBE_TIMEOUT_SECONDS:-180}" \
  "$PYTHON_BIN" "$ROOT/scripts/check_datasphere_qa_reference_probe.py" \
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
run_extraction_with_gpu_watchdog "$PYTHON_BIN" "$ROOT/run.py" "${extract_args[@]}"
require_complete_extraction() {
  local output_dir="$1"
  local failures="$output_dir/failed_extractions.jsonl"
  if [[ -s "$failures" ]]; then
    echo "[error] KG extraction was incomplete; refusing to mark this run successful." >&2
    cat "$failures" >&2
    return 1
  fi
}
require_complete_extraction "$STRICT_OUT"
if [[ -n "$QA_PILOT_LIMIT" ]]; then
  end_epoch="$(date +%s)"
  "$PYTHON_BIN" - "$METADATA" "$started" "$((end_epoch - start_epoch))" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$UNITS_PER_SECOND" "$GPU_TIME_LIMIT_SECONDS" "$QA_PILOT_LIMIT" <<'PY'
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
    "runs": ["strict-extract"],
}, indent=2) + "\n", encoding="utf-8")
PY
  echo "[done] cluster runtime probe extracted ${QA_PILOT_LIMIT} fixed QA rows; strict/support scoring was intentionally not run."
  exit 0
fi
"$PYTHON_BIN" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode strict --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$STRICT_OUT"
"$PYTHON_BIN" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode support --qa-pilot-manifest "$MANIFEST" \
  --output-dir "$SUPPORT_OUT"
"$PYTHON_BIN" "$ROOT/scripts/compare_qa_pilot_results.py" \
  --strict-dir "$STRICT_OUT" --support-dir "$SUPPORT_OUT" \
  --output "$RUN_ROOT/comparison.json"
end_epoch="$(date +%s)"

"$PYTHON_BIN" - "$METADATA" "$started" "$((end_epoch - start_epoch))" "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "$UNITS_PER_SECOND" "$GPU_TIME_LIMIT_SECONDS" <<'PY'
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
    "runs": ["strict", "support"],
}, indent=2) + "\n", encoding="utf-8")
PY
