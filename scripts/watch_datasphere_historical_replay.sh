#!/usr/bin/env bash
# Periodically download whatever a historical-qa-cache-replay Job has written
# so far and re-render the local HTML progress view. Safe to run against a
# still-EXECUTING Job: DataSphere may report "no files yet" early on, which
# this loop treats as "try again next tick", not an error.
#
# Usage:
#   bash scripts/watch_datasphere_historical_replay.sh <JOB_ID> <RUN_ID> [interval_seconds]
#
# Requires: datasphere CLI on PATH, YC_AUTH=yc set, profile "default" configured.
set -Eeuo pipefail
JOB_ID="${1:?Usage: $0 <JOB_ID> <RUN_ID> [interval_seconds]}"
RUN_ID="${2:?Usage: $0 <JOB_ID> <RUN_ID> [interval_seconds]}"
INTERVAL="${3:-60}"
PROFILE="${PROFILE:-default}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/outputs/datasphere-results/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "Watching Job $JOB_ID (run-id=$RUN_ID), refreshing every ${INTERVAL}s. Ctrl+C to stop." >&2

while :; do
  # datasphere prints an INFO log line to stdout before the JSON object itself
  # (not just stderr), so extract from the first '{' rather than parsing raw stdin.
  STATUS="$(datasphere --profile "$PROFILE" project job get --id "$JOB_ID" --format json 2>/dev/null \
    | "$PYTHON_BIN" -c 'import json,sys
text = sys.stdin.read()
start = text.find("{")
print(json.loads(text[start:]).get("status", "?") if start >= 0 else "?")' 2>/dev/null || echo "?")"
  echo "[$(date -u +%H:%M:%S)] status=$STATUS" >&2

  # download-files can legitimately report "no files yet" on an early EXECUTING
  # Job; that is not fatal to the watch loop.
  datasphere --profile "$PROFILE" project job download-files \
    --id "$JOB_ID" --with-logs --with-diagnostics --output-dir "$OUT_DIR" 2>&1 | tail -5 || true

  RUN_DIR="$(find "$OUT_DIR" -type d -name "historical-cache-replay" -print -quit 2>/dev/null || true)"
  if [[ -n "$RUN_DIR" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/render_historical_replay_progress_html.py" \
      --run-dir "$RUN_DIR" --output "$OUT_DIR/progress.html" \
      && echo "  -> updated $OUT_DIR/progress.html" >&2
  else
    echo "  -> no historical-cache-replay/ directory downloaded yet" >&2
  fi

  case "$STATUS" in
    SUCCESS|ERROR|CANCELLED) echo "Job reached a terminal status ($STATUS); stopping watch." >&2; break ;;
  esac
  sleep "$INTERVAL"
done
