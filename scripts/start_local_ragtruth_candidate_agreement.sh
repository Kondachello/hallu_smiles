#!/usr/bin/env bash
# Start the fixed paired candidate-agreement evaluation detached.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${HALLU_CA_ROOT:-/Volumes/mySSD/hallu_smiles/candidate-agreement}"
RUN_ID="${HALLU_CA_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="$ROOT/scripts/run_local_ragtruth_candidate_agreement.sh"
mkdir -p "$LOCAL_ROOT/launch-logs"

bash "$RUNNER" --preflight
if command -v tmux >/dev/null; then
  session="candidate-agreement-${RUN_ID}"
  tmux new-session -d -s "$session" "HALLU_CA_RUN_ID='$RUN_ID' bash '$RUNNER'"
  printf '[ok] started in tmux session %s\n' "$session"
  exit 0
fi
if command -v screen >/dev/null; then
  session="candidate-agreement-${RUN_ID}"
  log="$LOCAL_ROOT/launch-logs/${RUN_ID}.log"
  screen -dmS "$session" bash -c 'exec "$@" > "$0" 2>&1' "$log" \
    env HALLU_CA_ROOT="$LOCAL_ROOT" HALLU_CA_RUN_ID="$RUN_ID" HALLU_CA_CAFFEINATED=1 \
    caffeinate -dimsu bash "$RUNNER"
  printf '[ok] started in screen session %s\n' "$session"
  exit 0
fi
log="$LOCAL_ROOT/launch-logs/${RUN_ID}.log"
nohup env HALLU_CA_RUN_ID="$RUN_ID" HALLU_CA_CAFFEINATED=1 caffeinate -dimsu bash "$RUNNER" \
  > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOCAL_ROOT/launch-logs/${RUN_ID}.pid"
printf '[ok] started via nohup: pid=%s\n' "$!"
