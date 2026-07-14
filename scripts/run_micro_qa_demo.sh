#!/usr/bin/env bash
# Run the visible one-record KGGen demo without persisting an OpenRouter key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MODEL="${1:-}"
MAX_TOKENS="${KGGEN_DEMO_MAX_TOKENS:-1024}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <LiteLLM-model-slug>"
  echo "Example: $0 openrouter/nvidia/nemotron-nano-9b-v2:free"
  exit 2
fi
if [[ ! "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "KGGEN_DEMO_MAX_TOKENS must be a positive integer (got: $MAX_TOKENS)" >&2
  exit 2
fi
export KGGEN_DEMO_MAX_TOKENS="$MAX_TOKENS"
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

# Keep DSPy's optional cache writable and local to this project. This avoids a
# warning when the caller's home cache is unavailable (for example in a
# sandboxed terminal); KGGen extraction and clustering are unchanged.
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$ROOT/.cache/dspy}"
mkdir -p "$DSPY_CACHEDIR"

# A local credential file makes it possible to rerun the demo non-interactively
# without putting the key in code, config.yaml, shell history, or Git. Parse
# only the one expected assignment rather than sourcing arbitrary shell code.
KEY_FILE="$ROOT/.env.micro_qa_demo"
if [[ -z "${OPENROUTER_API_KEY:-}" && -f "$KEY_FILE" ]]; then
  OPENROUTER_API_KEY="$(sed -n -E 's/^OPENROUTER_API_KEY=([^[:space:]]+)[[:space:]]*$/\1/p' "$KEY_FILE" | head -n 1)"
  export OPENROUTER_API_KEY
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  read -r -s -p "OpenRouter API key (not saved): " OPENROUTER_API_KEY
  echo
  export OPENROUTER_API_KEY
fi

# ``-t`` delegates the filename template to macOS/BSD ``mktemp`` and avoids
# suffix handling differences between macOS and GNU coreutils.
DEMO_CONFIG="$(TMPDIR="${TMPDIR:-/tmp}" mktemp -t micro-qa-demo)"
trap 'rm -f "$DEMO_CONFIG"' EXIT

"$PYTHON" - "$ROOT/config.yaml" "$DEMO_CONFIG" "$MODEL" <<'PY'
import sys
import os
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
model = sys.argv[3]
with source.open(encoding="utf-8") as fh:
    config = yaml.safe_load(fh)
config["llm"]["model"] = model
config["llm"]["api_key_env"] = "OPENROUTER_API_KEY"
config["llm"]["max_tokens"] = int(os.environ.get("KGGEN_DEMO_MAX_TOKENS", "1024"))
with destination.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
PY

"$PYTHON" "$ROOT/scripts/micro_qa_demo.py" \
  --config "$DEMO_CONFIG" \
  --data-dir "$ROOT/data" \
  --output-dir "$ROOT/results/micro_qa_demo"
