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
# KGGen 0.4 otherwise requests up to 16k completion tokens.  That cannot fit
# alongside a prompt in the deliberately memory-safe 8k vLLM context window.
KGGEN_MAX_TOKENS="${KGGEN_MAX_TOKENS:-1024}"
RUNTIME_CONFIG="$RUN_ROOT/runtime_config.yaml"
MANIFEST="$RUN_ROOT/qa_pilot_manifest.json"
STRICT_OUT="$RUN_ROOT/strict"
SUPPORT_OUT="$RUN_ROOT/support"
VLLM_LOG="$RUN_ROOT/vllm.log"
GPU_LOG="$RUN_ROOT/gpu.csv"
METADATA="$RUN_ROOT/run_metadata.json"
UNITS_PER_SECOND="${DATASPHERE_UNITS_PER_SECOND:-}"
GPU_TIME_LIMIT_SECONDS="${GPU_TIME_LIMIT_SECONDS:-}"
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
mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME"
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
"$PYTHON_BIN" "$ROOT/scripts/make_datasphere_runtime_config.py" \
  --base-config "$ROOT/config.yaml" \
  --output "$RUNTIME_CONFIG" \
  --model-id "$MODEL_ID" \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --data-dir "$DATA_DIR" \
  --max-tokens "$KGGEN_MAX_TOKENS" \
  --work-dir "$RUN_ROOT"

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype "$MODEL_DTYPE" --max-model-len "$MODEL_MAX_MODEL_LEN" \
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

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 10 >"$GPU_LOG" 2>&1 &
GPU_PID=$!

start_epoch="$(date +%s)"
"$PYTHON_BIN" "$ROOT/run.py" --config "$RUNTIME_CONFIG" --stage all \
  --relation-mode strict --qa-pilot --qa-pilot-manifest-out "$MANIFEST" \
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
