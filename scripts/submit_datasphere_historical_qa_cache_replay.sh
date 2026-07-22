#!/usr/bin/env bash
# Submit only after the source commit and its CPU image have been published.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PROJECT_ID=""; RUN_ID=""; GATEWAY_URL=""; BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; PROFILE="default"; PYTHON_BIN="${PYTHON_BIN:-python3}"
while [[ $# -gt 0 ]]; do case "$1" in
  --project-id) PROJECT_ID="$2"; shift 2;; --run-id) RUN_ID="$2"; shift 2;; --gateway-url) GATEWAY_URL="$2"; shift 2;;
  --branch) BRANCH="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;; --profile) PROFILE="$2"; shift 2;;
  *) echo "Usage: $0 --project-id ID --run-id ID --gateway-url HTTPS_ORIGIN [--commit SHA] [--branch NAME]" >&2; exit 2;; esac; done
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$GATEWAY_URL" ]] || exit 2
git fetch --quiet origin "refs/heads/$BRANCH"; git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || { echo "Commit must be pushed." >&2; exit 2; }
IMAGE="$($PYTHON_BIN scripts/resolve_datasphere_runtime_image.py --repository "${DATASPHERE_VERTEX_CPU_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu}" --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}")"
RENDERED="datasphere/jobs/rendered/historical-qa-cache-replay-${RUN_ID}.yaml"
"$PYTHON_BIN" scripts/render_datasphere_historical_qa_cache_replay_job.py --commit "$COMMIT" --run-id "$RUN_ID" --gateway-url "$GATEWAY_URL" --docker-image "$IMAGE" --output "$RENDERED"
"$PYTHON_BIN" scripts/validate_datasphere_job.py --job "$RENDERED" --repo-root .
datasphere --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
datasphere --profile "$PROFILE" project job execute --async -p "$PROJECT_ID" -c "$RENDERED" -o "${RENDERED%.yaml}.execution.json"
