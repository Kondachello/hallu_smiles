#!/usr/bin/env bash
# Run the visible one-record KGGen demo without persisting an OpenRouter key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <LiteLLM-model-slug>"
  echo "Example: $0 openrouter/nvidia/nemotron-nano-9b-v2:free"
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Create the demo environment first."
  exit 2
fi
if [[ ! -f "$ROOT/data/source_info.jsonl" || ! -f "$ROOT/data/response.jsonl" ]]; then
  echo "RAGTruth data is missing. Run: $PYTHON download_data.py --data-dir data"
  exit 2
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
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
model = sys.argv[3]
with source.open(encoding="utf-8") as fh:
    config = yaml.safe_load(fh)
config["llm"]["model"] = model
config["llm"]["api_key_env"] = "OPENROUTER_API_KEY"
with destination.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
PY

"$PYTHON" "$ROOT/scripts/micro_qa_demo.py" \
  --config "$DEMO_CONFIG" \
  --data-dir "$ROOT/data" \
  --output-dir "$ROOT/results/micro_qa_demo"
