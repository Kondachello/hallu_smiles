#!/usr/bin/env bash
# Start the local DocRED runner detached, preferring tmux when it is installed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${HALLU_DOCRED_ROOT:-/Volumes/mySSD/hallu_smiles/docred-kggen}"
RUN_ID="${HALLU_DOCRED_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="$ROOT/scripts/run_local_docred_kg_eval.sh"
LOG_DIR="$LOCAL_ROOT/launch-logs"
mkdir -p "$LOG_DIR"

bash "$RUNNER" --preflight

if command -v tmux >/dev/null; then
  session="docred-local-${RUN_ID}"
  tmux new-session -d -s "$session" \
    "HALLU_DOCRED_RUN_ID='$RUN_ID' bash '$RUNNER'"
  printf '[ok] started in tmux session %s\n' "$session"
  exit 0
fi

log="$LOG_DIR/${RUN_ID}.log"
nohup env HALLU_DOCRED_DETACHED=1 HALLU_DOCRED_RUN_ID="$RUN_ID" \
  bash "$RUNNER" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$LOG_DIR/${RUN_ID}.pid"
printf '[ok] started via nohup fallback: pid=%s log=%s\n' "$pid" "$log"
