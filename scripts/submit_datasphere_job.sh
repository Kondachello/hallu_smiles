#!/usr/bin/env bash
# Render, validate, submit, monitor, and download one commit-pinned CPU API Job.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -x "$ROOT_DIR/.tools/yc/yc" ]]; then
  export PATH="$ROOT_DIR/.tools/yc:$PATH"
fi

usage() {
  cat <<'EOF'
Usage: scripts/submit_datasphere_job.sh --kind api-probe-c1|api-pilot-c1 --project-id ID --run-id ID [options]

Options:
  --branch NAME         Public origin branch containing the selected commit (default: current branch)
  --commit SHA          Full lowercase 40-character Git SHA (default: HEAD)
  --gate-artifact PATH  Required only for api-pilot-c1: successful matching probe tar
  --profile NAME        yc/datasphere profile (default: default)
  --python PATH         Python with PyYAML and the datasphere package
  --datasphere PATH     DataSphere CLI executable
  --no-wait             Return after duplicate-safe asynchronous submission

The model and API endpoint are read only from config.yaml. This command never accepts them.
EOF
}

KIND=""
PROJECT_ID=""
RUN_ID=""
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
GATE_ARTIFACT=""
PROFILE="default"
NO_WAIT=0
if [[ -x "$ROOT_DIR/.venv-datasphere/bin/python" ]]; then
  PYTHON_DEFAULT="$ROOT_DIR/.venv-datasphere/bin/python"
else
  PYTHON_DEFAULT="python3"
fi
if [[ -x "$ROOT_DIR/.venv-datasphere/bin/datasphere" ]]; then
  DATASPHERE_DEFAULT="$ROOT_DIR/.venv-datasphere/bin/datasphere"
else
  DATASPHERE_DEFAULT="datasphere"
fi
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
DATASPHERE_BIN="${DATASPHERE_BIN:-$DATASPHERE_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --gate-artifact) GATE_ARTIFACT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --datasphere) DATASPHERE_BIN="$2"; shift 2 ;;
    --no-wait) NO_WAIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$KIND" == "api-probe-c1" || "$KIND" == "api-pilot-c1" ]] || {
  echo "--kind must be api-probe-c1 or api-pilot-c1" >&2
  exit 2
}
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$BRANCH" ]] || {
  echo "--project-id, --run-id, and --branch are required" >&2
  exit 2
}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "--commit must be a lowercase full SHA" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-z0-9][a-z0-9-]{0,43}$ ]] || { echo "invalid --run-id" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
command -v "$DATASPHERE_BIN" >/dev/null || { echo "DataSphere CLI not found: $DATASPHERE_BIN" >&2; exit 2; }
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-native}"
retry_delay="${DATASPHERE_RETRY_DELAY_SECONDS:-2}"
[[ "$retry_delay" =~ ^[0-9]+$ ]] || { echo "DATASPHERE_RETRY_DELAY_SECONDS must be an integer" >&2; exit 2; }
execute_reconcile_attempts="${DATASPHERE_EXECUTE_RECONCILE_ATTEMPTS:-8}"
[[ "$execute_reconcile_attempts" =~ ^[1-9][0-9]*$ ]] || {
  echo "DATASPHERE_EXECUTE_RECONCILE_ATTEMPTS must be a positive integer" >&2
  exit 2
}

# The Job fetches this public remote by exact SHA. Never spend units on local-only code.
git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || {
  echo "Commit $COMMIT is not pushed to origin/$BRANCH; push it first." >&2
  exit 2
}
# DataSphere builds the manual Python environment from local submission files
# before the Job clones the selected commit.  Refuse a mixed-source submission:
# every tracked local file (including requirements and local-paths) must be the
# exact tree selected by --commit.  Untracked user artifacts remain untouched.
git diff --quiet "$COMMIT" -- || {
  echo "Tracked local files differ from --commit; submit from its exact clean tree." >&2
  exit 2
}

validate_probe_gate() {
  local config_values expected_model expected_api_base gate_json
  config_values="$("$PYTHON_BIN" - "$ROOT_DIR/config.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

llm = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["llm"]
print(f"{llm['model']}\t{llm['api_base']}")
PY
)"
  IFS=$'\t' read -r expected_model expected_api_base <<<"$config_values"
  gate_json="$("$PYTHON_BIN" scripts/validate_api_probe_artifact.py \
    --artifact "$1" \
    --expected-commit "$COMMIT" \
    --expected-model "$expected_model" \
    --expected-api-base "$expected_api_base" \
    --secret-env DASHSCOPE_API_KEY)"
  "$PYTHON_BIN" -c '
import json
import sys

gate = json.loads(sys.argv[1])
print(
    "[ok] probe gate: "
    f"contract={gate[\"contract_passed\"]}/3 "
    f"qa={gate[\"qa_completed\"]}/3 "
    f"provider_calls={gate[\"provider_calls\"]} "
    f"tokens={gate[\"total_tokens\"]} "
    f"(prompt={gate[\"prompt_tokens\"]}, completion={gate[\"completion_tokens\"]}) "
    f"artifact={sys.argv[2]}"
)
' "$gate_json" "$1"
}

if [[ "$KIND" == "api-pilot-c1" ]]; then
  [[ -n "$GATE_ARTIFACT" && -f "$GATE_ARTIFACT" ]] || {
    echo "api-pilot-c1 requires --gate-artifact pointing to a successful probe tar" >&2
    exit 2
  }
  validate_probe_gate "$GATE_ARTIFACT"
elif [[ -n "$GATE_ARTIFACT" ]]; then
  echo "--gate-artifact is only valid for api-pilot-c1" >&2
  exit 2
fi

rendered_dir="$ROOT_DIR/datasphere/jobs/rendered"
mkdir -p "$rendered_dir"
rendered="$rendered_dir/${KIND}-${RUN_ID}.yaml"
render_args=(--kind "$KIND" --commit "$COMMIT" --run-id "$RUN_ID" --output "$rendered")
if [[ "$KIND" == "api-pilot-c1" ]]; then
  render_args+=(--gate-artifact "$GATE_ARTIFACT")
fi
"$PYTHON_BIN" scripts/render_datasphere_job.py "${render_args[@]}"
"$PYTHON_BIN" scripts/validate_datasphere_job.py --job "$rendered" --repo-root "$ROOT_DIR"

retry_readonly() {
  local description="$1"
  shift
  local attempt rc
  for attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    else
      rc=$?
    fi
    if [[ "$attempt" == 5 ]]; then
      echo "$description failed after $attempt attempts" >&2
      return "$rc"
    fi
    echo "$description failed (attempt $attempt/5); retrying safely" >&2
    sleep $((attempt * retry_delay))
  done
}

retry_readonly "project get" "$DATASPHERE_BIN" --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null

jobs_json="$rendered_dir/jobs-${RUN_ID}.json"
list_jobs() {
  local attempt tmp rc
  tmp="$jobs_json.tmp"
  for attempt in 1 2 3 4 5; do
    rm -f "$tmp"
    if "$DATASPHERE_BIN" --profile "$PROFILE" project job list \
      -p "$PROJECT_ID" --format json -o "$tmp"; then
      if "$PYTHON_BIN" -m json.tool "$tmp" >/dev/null; then
        mv "$tmp" "$jobs_json"
        return 0
      else
        rc=$?
      fi
    else
      rc=$?
    fi
    if [[ "$attempt" == 5 ]]; then return "$rc"; fi
    echo "job list failed (attempt $attempt/5); retrying safely" >&2
    sleep $((attempt * retry_delay))
  done
}

JOB_NAME="hallu-${KIND}-${RUN_ID}"
find_exact_job() {
  "$PYTHON_BIN" - "$jobs_json" "$JOB_NAME" <<'PY'
import json
import sys

jobs = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [job for job in jobs if job.get("name") == sys.argv[2]]
if len(matches) > 1:
    raise SystemExit(f"duplicate exact-name Jobs already exist: {sys.argv[2]}")
if matches:
    print(f"{matches[0]['id']}\t{matches[0].get('status', 'UNKNOWN')}")
PY
}

reject_active_api_jobs() {
  "$PYTHON_BIN" - "$jobs_json" "$JOB_NAME" <<'PY'
import json
import sys

terminal = {"SUCCESS", "ERROR", "FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "ABORTED"}
jobs = json.load(open(sys.argv[1], encoding="utf-8"))
active = [
    job for job in jobs
    if str(job.get("name", "")).startswith("hallu-api-")
    and job.get("name") != sys.argv[2]
    and str(job.get("status", "UNKNOWN")).upper().removeprefix("JOB_STATUS_") not in terminal
]
if active:
    details = ", ".join(f"{job.get('name')} ({job.get('status')})" for job in active)
    raise SystemExit(f"another HalluGraph API Job is active: {details}")
PY
}

read_execution_job_id() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
job_id = payload.get("job_id") if isinstance(payload, dict) else None
if not isinstance(job_id, str) or not job_id.strip():
    raise SystemExit(1)
print(job_id)
PY
}

reconcile_ambiguous_execute() {
  local attempt exact list_rc find_rc
  for ((attempt = 1; attempt <= execute_reconcile_attempts; attempt++)); do
    if list_jobs; then
      :
    else
      list_rc=$?
      return "$list_rc"
    fi
    if exact="$(find_exact_job)"; then
      if [[ -n "$exact" ]]; then
        printf '%s\n' "$exact"
        return 0
      fi
    else
      find_rc=$?
      return "$find_rc"
    fi
    if (( attempt < execute_reconcile_attempts )); then
      echo "accepted Job not visible yet (reconcile $attempt/$execute_reconcile_attempts)" >&2
      sleep $((attempt * retry_delay))
    fi
  done
  return 3
}

list_jobs
exact="$(find_exact_job)"
if [[ -n "$exact" ]]; then
  IFS=$'\t' read -r JOB_ID JOB_STATUS <<<"$exact"
  echo "Reusing exact existing Job $JOB_ID ($JOB_STATUS); no duplicate execute request sent."
else
  reject_active_api_jobs
  execution_metadata="$rendered_dir/${KIND}-${RUN_ID}.execution.json"
  execute_log="$rendered_dir/${KIND}-${RUN_ID}.execute.log"
  JOB_ID=""
  for attempt in 1 2 3; do
    rm -f "$execution_metadata"
    set +e
    "$DATASPHERE_BIN" --profile "$PROFILE" project job execute --async \
      -p "$PROJECT_ID" -c "$rendered" -o "$execution_metadata" >"$execute_log" 2>&1
    execute_rc=$?
    set -e
    if [[ "$execute_rc" == 0 ]]; then
      if JOB_ID="$(read_execution_job_id "$execution_metadata")"; then
        break
      fi
      JOB_ID=""
      execute_rc=1
      echo "execute returned success but its metadata was absent or invalid; reconciling by exact name" >&2
    else
      echo "execute returned $execute_rc; reconciling by exact name before any retry" >&2
    fi

    # DataSphere's list endpoint can lag an accepted execute. Poll through a
    # bounded consistency window before deciding that a retry is safe.
    set +e
    exact="$(reconcile_ambiguous_execute)"
    reconcile_rc=$?
    set -e
    if [[ "$reconcile_rc" == 0 ]]; then
      IFS=$'\t' read -r JOB_ID JOB_STATUS <<<"$exact"
      echo "Recovered accepted Job $JOB_ID after an ambiguous execute response."
      break
    fi
    if [[ "$reconcile_rc" != 3 ]]; then
      echo "could not reconcile the ambiguous execute safely" >&2
      exit "$reconcile_rc"
    fi
    if [[ "$attempt" == 3 ]]; then
      sed -n '1,160p' "$execute_log" >&2
      echo "execute failed and no exact-name Job exists" >&2
      exit "$execute_rc"
    fi
    # Re-check the global single-Job invariant immediately before a retry.
    reject_active_api_jobs
  done
fi

[[ -n "$JOB_ID" ]] || { echo "could not resolve the DataSphere Job ID" >&2; exit 1; }
echo "DataSphere Job: $JOB_ID ($JOB_NAME)"
if [[ "$NO_WAIT" == 1 ]]; then
  echo "Asynchronous submission complete (--no-wait)."
  exit 0
fi

job_json="$rendered_dir/${KIND}-${RUN_ID}.job.json"
get_job() {
  local attempt tmp rc
  tmp="$job_json.tmp"
  for attempt in 1 2 3 4 5; do
    rm -f "$tmp"
    if "$DATASPHERE_BIN" --profile "$PROFILE" project job get --id "$JOB_ID" \
      --format json -o "$tmp"; then
      if "$PYTHON_BIN" -m json.tool "$tmp" >/dev/null; then
        mv "$tmp" "$job_json"
        return 0
      else
        rc=$?
      fi
    else
      rc=$?
    fi
    if [[ "$attempt" == 5 ]]; then return "$rc"; fi
    echo "job get failed (attempt $attempt/5); retrying safely" >&2
    sleep $((attempt * retry_delay))
  done
}

monitor_started="$(date +%s)"
if [[ "$KIND" == "api-probe-c1" ]]; then
  monitor_limit="${DATASPHERE_MONITOR_TIMEOUT_SECONDS:-10800}"
else
  monitor_limit="${DATASPHERE_MONITOR_TIMEOUT_SECONDS:-46800}"
fi
poll_seconds="${DATASPHERE_POLL_SECONDS:-30}"
while true; do
  get_job
  JOB_STATUS="$("$PYTHON_BIN" -c 'import json,sys; value=str(json.load(open(sys.argv[1])).get("status", "UNKNOWN")).upper(); print(value[len("JOB_STATUS_"):] if value.startswith("JOB_STATUS_") else value)' "$job_json")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $JOB_ID: $JOB_STATUS"
  case "$JOB_STATUS" in
    SUCCESS|ERROR|FAILED|CANCELLED|CANCELED|TIMEOUT|ABORTED) break ;;
  esac
  now="$(date +%s)"
  if (( now - monitor_started >= monitor_limit )); then
    echo "Local monitor timeout reached; Job was not cancelled: $JOB_ID" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done

download_dir="$rendered_dir/downloads-${RUN_ID}"
download_job() {
  local attempt attempt_dir rc
  for attempt in 1 2 3 4 5; do
    attempt_dir="$download_dir.attempt-$$-$attempt"
    mkdir -p "$attempt_dir"
    if "$DATASPHERE_BIN" --profile "$PROFILE" project job download-files --id "$JOB_ID" \
      --with-logs --with-diagnostics --output-dir "$attempt_dir"; then
      mv "$attempt_dir" "$download_dir"
      return 0
    else
      rc=$?
    fi
    if [[ "$attempt" == 5 ]]; then return "$rc"; fi
    echo "artifact download failed (attempt $attempt/5); retrying safely" >&2
    sleep $((attempt * retry_delay))
  done
}
if [[ -d "$download_dir" ]]; then
  echo "Reusing previously downloaded files: $download_dir"
else
  download_job
fi

if [[ "$KIND" == "api-probe-c1" ]]; then
  archive_name="api-probe-${RUN_ID}.tar.gz"
else
  archive_name="api-pilot-${RUN_ID}.tar.gz"
fi
archive_list="$rendered_dir/${KIND}-${RUN_ID}.archives.txt"
find "$download_dir" -type f -name "$archive_name" -print >"$archive_list"
archive_count="$(wc -l <"$archive_list" | tr -d ' ')"
[[ "$archive_count" == 1 ]] || {
  echo "Expected exactly one $archive_name in $download_dir, found $archive_count" >&2
  exit 1
}
archive_path="$(sed -n '1p' "$archive_list")"
"$PYTHON_BIN" - "$archive_path" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:gz") as tar:
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"unsafe artifact member: {member.name}")
print(f"[ok] safe readable artifact: {sys.argv[1]}")
PY

echo "Downloaded Job files: $download_dir"
echo "Primary artifact: $archive_path"
if [[ "$JOB_STATUS" != "SUCCESS" ]]; then
  echo "DataSphere Job ended with $JOB_STATUS; diagnostics were preserved." >&2
  exit 1
fi
if [[ "$KIND" == "api-probe-c1" ]]; then
  validate_probe_gate "$archive_path"
fi
echo "[ok] $KIND completed and its artifact passed local validation"
