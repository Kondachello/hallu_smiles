#!/usr/bin/env bash
# Create the native Python 3.10-3.12 runtime required by the KGGen QA demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
REQUIREMENTS="$ROOT/requirements.micro_qa_demo.txt"
HOST_ARCH="$(uname -m)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  CANDIDATES=("$PYTHON_BIN")
elif [[ "$HOST_ARCH" == "arm64" && -x /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
  # Homebrew's native Apple-Silicon Python is preferred over a Rosetta PATH.
  CANDIDATES=(/opt/homebrew/opt/python@3.12/bin/python3.12 python3.12 python3.11 python3.10)
else
  CANDIDATES=(python3.12 python3.11 python3.10)
fi

BASE_PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))
PY
  then
    BASE_PYTHON="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "$BASE_PYTHON" ]]; then
  cat >&2 <<'EOF'
No supported Python 3.10-3.12 was found.
On Apple Silicon with Homebrew, run: brew install python@3.12
Then rerun this setup command.
EOF
  exit 2
fi

base_arch="$("$BASE_PYTHON" -c 'import platform; print(platform.machine())')"
if [[ "$HOST_ARCH" == "arm64" && "$base_arch" != "arm64" ]]; then
  cat >&2 <<EOF
Refusing to create a Rosetta environment on Apple Silicon.
Selected Python is $base_arch, but this Mac is $HOST_ARCH.
Use native Homebrew Python 3.12: /opt/homebrew/opt/python@3.12/bin/python3.12
EOF
  exit 2
fi

if [[ -e "$VENV" ]]; then
  venv_arch="$("$VENV/bin/python" -c 'import platform; print(platform.machine())' 2>/dev/null || true)"
  if [[ -z "$venv_arch" ]]; then
    echo "Existing .venv is not usable; move it aside and rerun this script." >&2
    exit 2
  fi
  if [[ "$HOST_ARCH" == "arm64" && "$venv_arch" != "arm64" ]]; then
    cat >&2 <<EOF
Existing .venv is $venv_arch under Rosetta, but this Mac is ARM64.
Preserve it if wanted (for example: mv .venv .venv.x86_64-backup), then rerun:
  bash scripts/setup_micro_qa_demo.sh
EOF
    exit 2
  fi
else
  "$BASE_PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --prefer-binary -r "$REQUIREMENTS"

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$ROOT/.cache/dspy}"
mkdir -p "$DSPY_CACHEDIR"

"$VENV/bin/python" - <<'PY'
import platform
import torch
from kg_gen import KGGen

print(f"[setup] Python architecture: {platform.machine()}")
print(f"[setup] PyTorch: {torch.__version__}")
print(f"[setup] KGGen import: {KGGen.__name__}")
PY
