#!/usr/bin/env bash
# Materialise immutable controlled-run input objects and the frozen C/Q artifact.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HALLU_STAGE_PYTHON:-$ROOT/.venv/bin/python}"
PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
BUCKET="${GCP_RAGTRUTH_BUCKET:?Set GCP_RAGTRUTH_BUCKET from provision output}"

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/stage_gcp_ragtruth_llama31_inputs.sh \
  --annotations-csv PATH --historical-manifest PATH \
  --historical-extraction-summary PATH --historical-run-metadata PATH \
  --historical-runtime-identity PATH --historical-cache-root PATH \
  --historical-cache-export PATH --historical-cache-export-sha256 PATH \
  --output PATH
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotations-csv|--historical-manifest|--historical-extraction-summary|--historical-run-metadata|--historical-runtime-identity|--historical-cache-root|--historical-cache-export|--historical-cache-export-sha256|--output)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ARGS+=("$1" "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

value() {
  local key="$1" index
  for ((index=0; index<${#ARGS[@]}; index+=2)); do
    [[ "${ARGS[index]}" == "$key" ]] && { printf '%s' "${ARGS[index+1]}"; return; }
  done
  echo "missing required $key" >&2; exit 2
}

ANNOTATIONS="$(value --annotations-csv)"
HISTORICAL_MANIFEST="$(value --historical-manifest)"
OUTPUT="$(value --output)"
for path in "$PYTHON" "$ANNOTATIONS" "$HISTORICAL_MANIFEST" "$(value --historical-extraction-summary)" "$(value --historical-run-metadata)" "$(value --historical-runtime-identity)" "$(value --historical-cache-export)" "$(value --historical-cache-export-sha256)"; do
  [[ -e "$path" ]] || { echo "required staging input does not exist" >&2; exit 2; }
done
command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 2; }

STAGE_ROOT="$(mktemp -d)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
DATA_DIR="$STAGE_ROOT/data"
INPUT_DIR="$STAGE_ROOT/input"
"$PYTHON" "$ROOT/scripts/fetch_ragtruth_pinned_data.py" --output-dir "$DATA_DIR" > "$STAGE_ROOT/ragtruth-fetch.json"
"$PYTHON" "$ROOT/scripts/prepare_ragtruth_llama31_inputs.py" \
  --annotations-csv "$ANNOTATIONS" --data-dir "$DATA_DIR" --historical-manifest "$HISTORICAL_MANIFEST" --output-dir "$INPUT_DIR" > "$STAGE_ROOT/prepare.json"
"$PYTHON" "$ROOT/scripts/import_ragtruth_frozen_reference_graphs.py" \
  --annotations-csv "$ANNOTATIONS" --data-dir "$DATA_DIR" --historical-manifest "$HISTORICAL_MANIFEST" \
  --historical-extraction-summary "$(value --historical-extraction-summary)" \
  --historical-run-metadata "$(value --historical-run-metadata)" \
  --historical-runtime-identity "$(value --historical-runtime-identity)" \
  --historical-cache-root "$(value --historical-cache-root)" \
  --historical-cache-export "$(value --historical-cache-export)" \
  --historical-cache-export-sha256 "$(value --historical-cache-export-sha256)" \
  --output "$STAGE_ROOT/frozen-reference.json" > "$STAGE_ROOT/import.json"
cp "$STAGE_ROOT/ragtruth-fetch.json" "$INPUT_DIR/ragtruth_data_provenance.json"
cp "$ANNOTATIONS" "$INPUT_DIR/ragtruth_llama31_annotated.csv"
cp "$HISTORICAL_MANIFEST" "$INPUT_DIR/historical_qa_manifest.json"
cp "$DATA_DIR/source_info.jsonl" "$INPUT_DIR/source_info.jsonl"
cp "$DATA_DIR/response.jsonl" "$INPUT_DIR/response.jsonl"

INPUT_SHA="$($PYTHON - "$INPUT_DIR/input_provenance.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
FROZEN_SHA="$($PYTHON - "$STAGE_ROOT/frozen-reference.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
DATA_SHA="$($PYTHON - "$INPUT_DIR/ragtruth_data_provenance.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
INPUT_PREFIX="inputs/llama31-controlled/$INPUT_SHA"
FROZEN_OBJECT="frozen-reference/llama31-controlled-$FROZEN_SHA.json"
for name in input_provenance.json ragtruth_data_provenance.json source_info.jsonl response.jsonl ragtruth_llama31_annotated.csv historical_qa_manifest.json llama31_manifest.json; do
  gcloud storage cp "$INPUT_DIR/$name" "gs://$BUCKET/$INPUT_PREFIX/$name" --project "$PROJECT_ID" --if-generation-match=0
done
gcloud storage cp "$STAGE_ROOT/frozen-reference.json" "gs://$BUCKET/$FROZEN_OBJECT" --project "$PROJECT_ID" --if-generation-match=0
"$PYTHON" - "$OUTPUT" "$PROJECT_ID" "$BUCKET" "$INPUT_PREFIX" "$INPUT_SHA" "$DATA_SHA" "$FROZEN_OBJECT" "$FROZEN_SHA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = dict(zip((
    'project_id', 'bucket', 'input_prefix', 'input_provenance_sha256',
    'data_provenance_sha256', 'frozen_reference_object', 'frozen_reference_sha256',
), sys.argv[2:]))
payload['protocol'] = 'hallu-gcp-ragtruth-llama31-staging-v1'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(payload, sort_keys=True))
PY
