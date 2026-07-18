#!/usr/bin/env bash
# DataSphere CPU Job: Alibaba API contract -> bounded 3-QA gate or full 20-QA pilot.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=""
DATA_DIR=""
RUN_DIR=""
PROBE_ARTIFACT=""
CONFIG="$ROOT/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  echo "Usage: $0 --mode probe|pilot --data-dir DIR --run-dir DIR [--probe-artifact TAR] [--config YAML]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --probe-artifact) PROBE_ARTIFACT="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$MODE" == "probe" || "$MODE" == "pilot" ]] || { usage; exit 2; }
[[ -n "$DATA_DIR" && -n "$RUN_DIR" ]] || { usage; exit 2; }
[[ "$MODE" != "probe" || -z "$PROBE_ARTIFACT" ]] || {
  echo "Probe mode must not receive --probe-artifact." >&2
  exit 2
}
[[ "$MODE" != "pilot" || -n "$PROBE_ARTIFACT" ]] || {
  echo "Pilot mode requires --probe-artifact." >&2
  exit 2
}
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT to the exact pushed Git SHA}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python runtime not found: $PYTHON_BIN" >&2; exit 2; }

case "$RUN_DIR" in
  ""|/|"$ROOT"|"$DATA_DIR")
    echo "Unsafe or overlapping RUN_DIR: $RUN_DIR" >&2
    exit 2
    ;;
esac
[[ ! -L "$RUN_DIR" ]] || { echo "RUN_DIR must not be a symlink." >&2; exit 2; }

mkdir -p "$RUN_DIR"
chmod 700 "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
DATA_DIR="$(cd "$DATA_DIR" && pwd -P)"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd -P)/$(basename "$CONFIG")"
API_KEY_ENV="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

document = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = str((document.get("llm") or {}).get("api_key_env", ""))
if not name:
    raise SystemExit("config llm.api_key_env is empty")
print(name)
PY
)"
if [[ -n "$PROBE_ARTIFACT" ]]; then
  PROBE_ARTIFACT="$(cd "$(dirname "$PROBE_ARTIFACT")" && pwd -P)/$(basename "$PROBE_ARTIFACT")"
fi

ARTIFACT_ARCHIVE="${ARTIFACT_ARCHIVE:-${RUN_DIR}.tar.gz}"
mkdir -p "$(dirname "$ARTIFACT_ARCHIVE")"
ARTIFACT_ARCHIVE="$(cd "$(dirname "$ARTIFACT_ARCHIVE")" && pwd -P)/$(basename "$ARTIFACT_ARCHIVE")"
case "$ARTIFACT_ARCHIVE" in
  "$RUN_DIR"/*)
    echo "ARTIFACT_ARCHIVE must be outside RUN_DIR." >&2
    exit 2
    ;;
esac

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
unset HF_TOKEN || true

# Stream progress to DataSphere while retaining both channels in the archive.
# Explicit FIFOs/PIDs let the EXIT handler close and flush both tee processes
# before redaction and tar creation; this avoids a log/secret race.
exec 3>&1 4>&2
STDOUT_PIPE="$RUN_DIR/.job.stdout.pipe"
STDERR_PIPE="$RUN_DIR/.job.stderr.pipe"
mkfifo "$STDOUT_PIPE" "$STDERR_PIPE"
tee -a "$RUN_DIR/job.stdout.log" <"$STDOUT_PIPE" >&3 &
STDOUT_TEE_PID=$!
tee -a "$RUN_DIR/job.stderr.log" <"$STDERR_PIPE" >&4 &
STDERR_TEE_PID=$!
exec 1>"$STDOUT_PIPE" 2>"$STDERR_PIPE"

finalize() {
  local status=$?
  local temporary_archive="${ARTIFACT_ARCHIVE}.tmp.$$"
  trap - EXIT INT TERM

  # Stop writing the FIFOs, then wait until every logged byte is on disk.
  exec 1>&3 2>&4
  wait "$STDOUT_TEE_PID" || status=$?
  wait "$STDERR_TEE_PID" || status=$?
  rm -f "$STDOUT_PIPE" "$STDERR_PIPE"

  # Replace an exact secret value in any bounded regular artifact before tar.
  # The normal provider logger is allowlist-only; this is defense in depth for
  # third-party exception strings written during a failed Job.
  "$PYTHON_BIN" - "$RUN_DIR" "$API_KEY_ENV" <<'PY' || status=$?
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
secret = os.environ.get(sys.argv[2], "").encode()
if secret:
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if secret in data:
            temporary = path.with_name(f".{path.name}.redacted.{os.getpid()}")
            temporary.write_bytes(data.replace(secret, b"[REDACTED]"))
            os.replace(temporary, path)
PY

  if [[ ! -f "$RUN_DIR/run_metadata.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/run_metadata.json" "$MODE" "$status" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "protocol": "hallu-api-job-v1",
    "mode": sys.argv[2],
    "state": "error",
    "exit_code": int(sys.argv[3]),
    "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}, indent=2) + "\n", encoding="utf-8")
PY
  fi

  rm -f "$temporary_archive"
  tar -czf "$temporary_archive" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" || status=$?
  if [[ -f "$temporary_archive" ]]; then
    mv -f "$temporary_archive" "$ARTIFACT_ARCHIVE"
    echo "[artifact] $ARTIFACT_ARCHIVE"
  else
    echo "[artifact:error] archive creation failed" >&2
    status=1
  fi

  if [[ "$status" -eq 0 && "$MODE" == "probe" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/validate_api_probe_artifact.py" \
      --artifact "$ARTIFACT_ARCHIVE" \
      --expected-commit "$EXPECTED_SOURCE_COMMIT" \
      --config "$CONFIG" || status=$?
  fi
  exit "$status"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command=(
  "$PYTHON_BIN" "$ROOT/scripts/run_api_job.py"
  --mode "$MODE"
  --config "$CONFIG"
  --data-dir "$DATA_DIR"
  --run-dir "$RUN_DIR"
)
if [[ -n "$PROBE_ARTIFACT" ]]; then
  command+=(--probe-artifact "$PROBE_ARTIFACT")
fi
"${command[@]}"
