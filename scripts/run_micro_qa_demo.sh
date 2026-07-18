#!/usr/bin/env bash
# Run the visible one-record KGGen demo with a local provider API key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Create the demo environment first:"
  echo "  bash scripts/setup_micro_qa_demo.sh"
  exit 2
fi
HOST_ARCH="$(uname -m)"
VENV_ARCH="$("$PYTHON" -c 'import platform; print(platform.machine())' 2>/dev/null || true)"
if [[ "$HOST_ARCH" == "arm64" && "$VENV_ARCH" != "arm64" ]]; then
  cat >&2 <<EOF
Refusing to run an x86_64/Rosetta .venv on this Apple-Silicon Mac.
Preserve it if wanted (for example: mv .venv .venv.x86_64-backup), then run:
  bash scripts/setup_micro_qa_demo.sh
EOF
  exit 2
fi
if ! "$PYTHON" -c 'from kg_gen import KGGen; assert KGGen' >/dev/null 2>&1; then
  echo "The full KGGen runtime is missing or cannot import. Run:"
  echo "  bash scripts/setup_micro_qa_demo.sh"
  exit 2
fi
if [[ ! -f "$ROOT/data/source_info.jsonl" || ! -f "$ROOT/data/response.jsonl" ]]; then
  echo "RAGTruth data is missing. Run: $PYTHON download_data.py --data-dir data"
  exit 2
fi

# config.yaml is the only source for the provider model and credential name,
# including this historical one-record demo.
config_values="$("$PYTHON" - "$ROOT/config.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

llm = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["llm"]
print(f"{llm['model']}\t{llm['api_key_env']}")
PY
)"
IFS=$'\t' read -r MODEL API_KEY_ENV <<<"$config_values"
[[ -n "$MODEL" && -n "$API_KEY_ENV" ]] || {
  echo "config.yaml must define llm.model and llm.api_key_env" >&2
  exit 2
}
echo "[demo] model=$MODEL (from config.yaml)"

# Keep DSPy's optional cache writable and local to this project. This avoids a
# warning when the caller's home cache is unavailable (for example in a
# sandboxed terminal); KGGen extraction and clustering are unchanged.
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$ROOT/.cache/dspy}"
mkdir -p "$DSPY_CACHEDIR"

# A local credential file makes it possible to rerun the demo non-interactively
# without putting keys in code, config.yaml, shell history, or Git. Parse only
# the provider-specific assignment rather than sourcing arbitrary shell code.
KEY_FILE="$ROOT/.env.micro_qa_demo"
if [[ -z "${!API_KEY_ENV:-}" && -f "$KEY_FILE" ]]; then
  parsed_key="$(sed -n -E "s/^${API_KEY_ENV}=([^[:space:]]+)[[:space:]]*$/\\1/p" "$KEY_FILE" | head -n 1)"
  if [[ -n "$parsed_key" ]]; then
    export "$API_KEY_ENV=$parsed_key"
  fi
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  read -r -s -p "$API_KEY_ENV (not saved): " parsed_key
  echo
  export "$API_KEY_ENV=$parsed_key"
fi

"$PYTHON" "$ROOT/scripts/micro_qa_demo.py" \
  --config "$ROOT/config.yaml" \
  --data-dir "$ROOT/data" \
  --output-dir "$ROOT/results/micro_qa_demo" \
  "$@"
