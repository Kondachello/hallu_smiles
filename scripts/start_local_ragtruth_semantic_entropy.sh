#!/usr/bin/env bash
# Start the long local SemanticEntropy run detached and keep its monitor active.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${HALLU_SE_ROOT:-/Volumes/mySSD/hallu_smiles/semantic-entropy}"
RUN_ID="${HALLU_SE_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="$ROOT/scripts/run_local_ragtruth_semantic_entropy.sh"
LOG_DIR="$LOCAL_ROOT/launch-logs"
mkdir -p "$LOG_DIR"

bash "$RUNNER" --preflight

if command -v tmux >/dev/null; then
  session="semantic-entropy-${RUN_ID}"
  tmux new-session -d -s "$session" "HALLU_SE_RUN_ID='$RUN_ID' bash '$RUNNER'"
  printf '[ok] started in tmux session %s\n' "$session"
  exit 0
fi

if command -v screen >/dev/null; then
  session="semantic-entropy-${RUN_ID}"
  log="$LOG_DIR/${RUN_ID}.log"
  screen -dmS "$session" bash -c 'exec "$@" > "$0" 2>&1' "$log" \
    env HALLU_SE_RUN_ID="$RUN_ID" HALLU_SE_CAFFEINATED=1 \
    caffeinate -dimsu bash "$RUNNER"
  printf '[ok] started in screen session %s\n' "$session"
  exit 0
fi

log="$LOG_DIR/${RUN_ID}.log"
nohup env HALLU_SE_RUN_ID="$RUN_ID" bash "$RUNNER" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$LOG_DIR/${RUN_ID}.pid"
printf '[ok] started via nohup: pid=%s log=%s\n' "$pid" "$log"
