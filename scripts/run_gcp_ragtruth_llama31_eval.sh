#!/usr/bin/env bash
# Controlled Llama-3.1-8B RAGTruth evaluation on one persistent GCE CPU VM.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HALLU_GCP_PYTHON:-/opt/hallu/client/bin/python}"
PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
REGION="${GCP_REGION:-europe-west4}"
GATEWAY_URL="${HALLU_GATEWAY_URL:-https://hallu-docred-vertex-gateway-qbfs4yp45q-ez.a.run.app}"
GATEWAY_SECRET="${HALLU_GATEWAY_SECRET:-hallu-docred-gateway-bearer}"
BUCKET="${GCP_RAGTRUTH_BUCKET:?Set GCP_RAGTRUTH_BUCKET}"
INPUT_PREFIX="${GCP_RAGTRUTH_INPUT_PREFIX:?Set GCP_RAGTRUTH_INPUT_PREFIX}"
INPUT_PROVENANCE_SHA256="${GCP_RAGTRUTH_INPUT_PROVENANCE_SHA256:?Set GCP_RAGTRUTH_INPUT_PROVENANCE_SHA256}"
DATA_PROVENANCE_SHA256="${GCP_RAGTRUTH_DATA_PROVENANCE_SHA256:?Set GCP_RAGTRUTH_DATA_PROVENANCE_SHA256}"
FROZEN_REFERENCE_OBJECT="${GCP_RAGTRUTH_FROZEN_REFERENCE_OBJECT:?Set GCP_RAGTRUTH_FROZEN_REFERENCE_OBJECT}"
FROZEN_REFERENCE_SHA256="${GCP_RAGTRUTH_FROZEN_REFERENCE_SHA256:?Set GCP_RAGTRUTH_FROZEN_REFERENCE_SHA256}"
WORK_ROOT="${HALLU_GCP_WORK_ROOT:-/work/hallu-ragtruth-llama31}"
RUN_ID="${GCP_RAGTRUTH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$WORK_ROOT/runs/gcp-ragtruth-llama31-$RUN_ID"
INPUT_ROOT="$WORK_ROOT/inputs/$INPUT_PREFIX"
ARCHIVE_OBJECT="terminal-archives/gcp-ragtruth-llama31-$RUN_ID.tar.gz"
ARCHIVE_LOCAL="$WORK_ROOT/archives/gcp-ragtruth-llama31-$RUN_ID.tar.gz"
STATUS="started"
EXTRACTION_MAX_TOKENS_CEILING="${HALLU_GCP_EXTRACTION_MAX_TOKENS_CEILING:-32768}"
RUNNER_SOURCE_COMMIT="${HALLU_RUNNER_SOURCE_COMMIT:-}"
export HALLU_GATEWAY_URL="$GATEWAY_URL"

[[ "$EXTRACTION_MAX_TOKENS_CEILING" =~ ^[0-9]+$ ]] || {
  echo "[gcp-runner] extraction token ceiling must be an integer" >&2
  exit 2
}
(( EXTRACTION_MAX_TOKENS_CEILING >= 4096 && EXTRACTION_MAX_TOKENS_CEILING <= 32768 )) || {
  echo "[gcp-runner] extraction token ceiling must be within 4096..32768" >&2
  exit 2
}

require_file() {
  [[ -f "$1" ]] || { echo "[gcp-runner] required file is missing" >&2; exit 2; }
}

download() {
  "$PYTHON" "$ROOT/scripts/gcp_transfer_object.py" download \
    --bucket "$BUCKET" --object "$1" --output "$2" --sha256 "$3"
}

archive_and_upload() {
  mkdir -p "$(dirname "$ARCHIVE_LOCAL")"
  local archive_args=()
  if [[ "$STATUS" != "success" ]]; then
    archive_args+=(--allow-partial)
  fi
  "$PYTHON" "$ROOT/scripts/archive_gcp_ragtruth_llama31_eval.py" \
    --run-root "$RUN_ROOT" --archive "$ARCHIVE_LOCAL" "${archive_args[@]}"
  "$PYTHON" "$ROOT/scripts/gcp_transfer_object.py" upload \
    --bucket "$BUCKET" --object "$ARCHIVE_OBJECT" --input "$ARCHIVE_LOCAL" --if-absent
}

write_terminal_metadata() {
  "$PYTHON" - "$RUN_ROOT/run_metadata.json" "$STATUS" "$PROJECT_ID" "$REGION" "$RUN_ID" "$EXTRACTION_MAX_TOKENS_CEILING" "$RUNNER_SOURCE_COMMIT" <<'PY'
import json
import os
import sys
from pathlib import Path

path, status, project, region, run_id, extraction_ceiling, runner_source_commit = map(str, sys.argv[1:])
payload = {
    "protocol": "hallu-gcp-ragtruth-llama31-run-v1",
    "status": status,
    "project_id": project,
    "region": region,
    "run_id": run_id,
    "gateway_url": os.environ["HALLU_GATEWAY_URL"],
    "gateway_manifest_sha256": os.environ.get("EXPECTED_GATEWAY_MANIFEST_SHA256"),
    "input_prefix": os.environ["GCP_RAGTRUTH_INPUT_PREFIX"],
    "frozen_reference_object": os.environ["GCP_RAGTRUTH_FROZEN_REFERENCE_OBJECT"],
    "cache_root": "persistent_vm_disk_only",
    "reference_graph_mode": "frozen_historical_artifact",
    "relation_modes": ["strict", "support-critical"],
    "concurrency": 1,
    "extraction_recovery": {
        "normal_max_tokens": 4096,
        "max_tokens_ceiling": int(extraction_ceiling),
        "protocol": "provider-length-only-adaptive-retry-v1",
    },
    "runner_source_commit": runner_source_commit or None,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ "$STATUS" != "success" ]]; then
    STATUS="failed"
  fi
  write_terminal_metadata || true
  archive_and_upload || true
  unset HALLU_GATEWAY_API_KEY EXPECTED_GATEWAY_MANIFEST_SHA256
  if [[ "$STATUS" == "success" ]]; then
    echo "RUN_COMPLETE archive_object=$ARCHIVE_OBJECT" >&2
  else
    echo "RUN_FAILED archive_object=$ARCHIVE_OBJECT" >&2
  fi
  exit "$exit_code"
}
trap cleanup EXIT

mkdir -p "$RUN_ROOT" "$INPUT_ROOT" "$WORK_ROOT/cache"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

require_file "$PYTHON"
require_file /opt/hallu/runtime-manifest.json

# The bearer is read through VM ADC into this process environment only.  It is
# never rendered into a config, written to disk, or supplied as a CLI argument.
HALLU_GATEWAY_API_KEY="$("$PYTHON" "$ROOT/scripts/read_gcp_gateway_secret.py" \
  --project "$PROJECT_ID" --secret "$GATEWAY_SECRET")"
export HALLU_GATEWAY_API_KEY

download "$INPUT_PREFIX/input_provenance.json" "$INPUT_ROOT/input_provenance.json" "$INPUT_PROVENANCE_SHA256"
input_json="$INPUT_ROOT/input_provenance.json"
input_value() {
  "$PYTHON" - "$input_json" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split('.'):
    value = value[part]
print(value)
PY
}

download "$INPUT_PREFIX/ragtruth_data_provenance.json" "$INPUT_ROOT/ragtruth_data_provenance.json" "$DATA_PROVENANCE_SHA256"
download "$INPUT_PREFIX/source_info.jsonl" "$INPUT_ROOT/source_info.jsonl" "$(input_value source_info_jsonl_sha256)"
download "$INPUT_PREFIX/response.jsonl" "$INPUT_ROOT/response.jsonl" "$(input_value response_jsonl_sha256)"
download "$INPUT_PREFIX/ragtruth_llama31_annotated.csv" "$INPUT_ROOT/ragtruth_llama31_annotated.csv" "$(input_value annotation_csv_sha256)"
download "$INPUT_PREFIX/historical_qa_manifest.json" "$INPUT_ROOT/historical_qa_manifest.json" "$(input_value historical_manifest_sha256)"
download "$INPUT_PREFIX/llama31_manifest.json" "$INPUT_ROOT/llama31_manifest.json" "$(input_value manifest_sha256)"
download "$FROZEN_REFERENCE_OBJECT" "$INPUT_ROOT/frozen_reference_graphs.json" "$FROZEN_REFERENCE_SHA256"

"$PYTHON" - "$INPUT_ROOT/input_provenance.json" "$INPUT_ROOT/ragtruth_data_provenance.json" <<'PY'
import json
import sys
from src.llama31_eval import DEFAULT_RAGTRUTH_COMMIT

inputs = json.load(open(sys.argv[1], encoding="utf-8"))
data = json.load(open(sys.argv[2], encoding="utf-8"))
if inputs.get("ragtruth_commit") != DEFAULT_RAGTRUTH_COMMIT:
    raise SystemExit("input provenance has an unpinned RAGTruth revision")
if data.get("commit") != DEFAULT_RAGTRUTH_COMMIT:
    raise SystemExit("RAGTruth data provenance has an unpinned revision")
for key in ("source_info_jsonl_sha256", "response_jsonl_sha256"):
    if inputs.get(key) != data.get(key):
        raise SystemExit("RAGTruth JSONL checksums differ between input provenance and source snapshot")
PY

"$PYTHON" "$ROOT/scripts/fetch_gcp_gateway_manifest.py" \
  --gateway-url "$GATEWAY_URL" --output "$RUN_ROOT/gateway_manifest.json"
EXPECTED_GATEWAY_MANIFEST_SHA256="$("$PYTHON" - "$RUN_ROOT/gateway_manifest.json" <<'PY'
import json
import sys
from gateway.core import canonical_manifest_sha256
print(canonical_manifest_sha256(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
export EXPECTED_GATEWAY_MANIFEST_SHA256

# Preserve the 4096-token normal request and cache namespace. Only a
# provider-confirmed finish_reason=length may use this bounded recovery headroom.
"$PYTHON" "$ROOT/scripts/make_gcp_vertex_config.py" \
  --base-config "$ROOT/config.yaml" --gateway-manifest "$RUN_ROOT/gateway_manifest.json" \
  --gateway-url "$GATEWAY_URL" --gcp-runtime-manifest /opt/hallu/runtime-manifest.json \
  --embedding-model-path /opt/hallu/models/all-MiniLM-L6-v2 \
  --output "$RUN_ROOT/runtime-config.yaml" --identity-output "$RUN_ROOT/config_identity.json" \
  --data-dir "$INPUT_ROOT" --work-dir "$RUN_ROOT" --cache-root "$WORK_ROOT/cache" \
  --max-tokens 4096 --extraction-max-tokens-ceiling "$EXTRACTION_MAX_TOKENS_CEILING" --concurrency 1 > "$RUN_ROOT/runtime_identity.json"

"$PYTHON" - "$WORK_ROOT/cache/checkpoint_identity.json" "$RUN_ROOT/config_identity.json" "$INPUT_ROOT/input_provenance.json" "$INPUT_ROOT/frozen_reference_graphs.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path, config_path, input_path, frozen_path = map(Path, sys.argv[1:])
def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
identity = json.load(open(config_path, encoding="utf-8"))
payload = {
    "protocol": "hallu-gcp-ragtruth-llama31-checkpoint-v1",
    "gateway_manifest_sha256": identity["gateway_manifest_sha256"],
    "runtime_fingerprint": identity["runtime_fingerprint"],
    "input_provenance_sha256": sha256(input_path),
    "frozen_reference_artifact_sha256": sha256(frozen_path),
    "concurrency": 1,
}
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and json.load(open(path, encoding="utf-8")) != payload:
    raise SystemExit("persistent cache checkpoint identity differs; refusing incompatible reuse")
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

cp "$INPUT_ROOT/input_provenance.json" "$RUN_ROOT/input_provenance.json"
cp "$INPUT_ROOT/llama31_manifest.json" "$RUN_ROOT/llama31_manifest.json"
"$PYTHON" - "$INPUT_ROOT/frozen_reference_graphs.json" "$RUN_ROOT/frozen_reference_provenance.json" <<'PY'
import json
import sys
from pathlib import Path
from src.llama31_eval import file_sha256

artifact = json.load(open(sys.argv[1], encoding="utf-8"))
payload = {
    "protocol": artifact.get("protocol"),
    "artifact_sha256": file_sha256(sys.argv[1]),
    "historical_provenance": artifact.get("historical_provenance"),
    "records": len(artifact.get("records", [])),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

base_args=(
  --config "$RUN_ROOT/runtime-config.yaml"
  --data-dir "$INPUT_ROOT"
  --llama31-csv "$INPUT_ROOT/ragtruth_llama31_annotated.csv"
  --qa-manifest "$INPUT_ROOT/llama31_manifest.json"
  --llama31-historical-manifest "$INPUT_ROOT/historical_qa_manifest.json"
  --frozen-reference-artifact "$INPUT_ROOT/frozen_reference_graphs.json"
  --exclude-source-id 12448
  --stage all
  --no-audit
)

"$PYTHON" "$ROOT/run.py" "${base_args[@]}" --relation-mode strict --output-dir "$RUN_ROOT/strict"
"$PYTHON" - "$RUN_ROOT/strict/extraction_summary.json" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "ready_with_explicit_exclusions" or p.get("responses_completed") != 749:
    raise SystemExit("strict did not complete all 749 non-quarantined answer graph extractions")
if p.get("reference_graph_provenance", {}).get("reference_origin") != "frozen_historical_artifact":
    raise SystemExit("strict did not record frozen-reference provenance")
PY

"$PYTHON" "$ROOT/run.py" "${base_args[@]}" --relation-mode support-critical --kg-cache-only --output-dir "$RUN_ROOT/support-critical"
"$PYTHON" - "$RUN_ROOT/support-critical/extraction_summary.json" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "ready_with_explicit_exclusions" or p.get("responses_completed") != 749:
    raise SystemExit("support-critical did not reuse a complete 749-answer graph cache")
if any(record.get("cache", {}).get("answer", {}).get("cache_origin") != "primary" for record in p.get("graph_records", [])):
    raise SystemExit("support-critical answer graph cache was not solely current-run primary cache")
PY

"$PYTHON" "$ROOT/scripts/write_cache_inventory.py" --cache-root "$WORK_ROOT/cache" --output "$RUN_ROOT/cache_before_replay.json" --protocol hallu-ragtruth-llama31-cache-inventory-v1

"$PYTHON" "$ROOT/run.py" "${base_args[@]}" --relation-mode strict --cache-only --output-dir "$RUN_ROOT/replay-strict"
"$PYTHON" "$ROOT/run.py" "${base_args[@]}" --relation-mode support-critical --cache-only --output-dir "$RUN_ROOT/replay-support-critical"
"$PYTHON" "$ROOT/scripts/verify_cache_replay.py" --live-dir "$RUN_ROOT/strict" --replay-dir "$RUN_ROOT/replay-strict"
"$PYTHON" "$ROOT/scripts/verify_cache_replay.py" --live-dir "$RUN_ROOT/support-critical" --replay-dir "$RUN_ROOT/replay-support-critical"
"$PYTHON" "$ROOT/scripts/write_cache_inventory.py" --cache-root "$WORK_ROOT/cache" --output "$RUN_ROOT/cache_after_replay.json" --protocol hallu-ragtruth-llama31-cache-inventory-v1
cmp -s "$RUN_ROOT/cache_before_replay.json" "$RUN_ROOT/cache_after_replay.json" || {
  echo "cache-only replay changed persistent cache inventory" >&2
  exit 1
}
"$PYTHON" - "$RUN_ROOT/replay_verification.json" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "protocol": "hallu-ragtruth-llama31-cache-replay-v1",
    "strict": "identical except verifier_cache_hit diagnostic",
    "support_critical": "identical except verifier_cache_hit diagnostic",
    "live_inference_calls": 0,
    "cache_inventory_unchanged": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON" "$ROOT/scripts/write_ragtruth_llama31_controlled_summary.py" \
  --manifest "$INPUT_ROOT/llama31_manifest.json" --strict-dir "$RUN_ROOT/strict" \
  --support-critical-dir "$RUN_ROOT/support-critical" --output "$RUN_ROOT/controlled_summary.json"

STATUS="success"
