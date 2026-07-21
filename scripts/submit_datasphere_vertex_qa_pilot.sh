#!/usr/bin/env bash
# Validate the successful CPU 3-QA gate, then submit a cache-backed critical QA run.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-ares}"

usage() {
  cat <<'EOF'
Usage: scripts/submit_datasphere_vertex_qa_pilot.sh --project-id ID --run-id ID --gateway-url HTTPS_URL --gate-artifact PATH [options]

Options:
  --qa-sample-size N      Total QA sources (default: 100)
  --qa-test-fraction F    Held-out test fraction, e.g. 0.2 (default: 0.2)
  --cv-folds N            Train-only stratified CV folds (default: 5)
  --concurrency N         Gateway requests in flight (default: 1)
  --timeout-seconds N     Optional in-container wall-time ceiling; 0 uses the DataSphere deadline (default: 0)
  --commit SHA            Pushed source commit (default: HEAD)
  --branch NAME           Remote branch containing the commit (default: current)
  --skip-origin-fetch     Verify locally cached origin/BRANCH (only for a transient DNS outage)
  --docker-image REF      Optional immutable image; must match the committed build
  --profile NAME          DataSphere CLI profile (default: default)
EOF
}
PROJECT_ID=""; RUN_ID=""; GATEWAY_URL=""; GATE_ARTIFACT=""; BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; IMAGE=""; PROFILE="default"; SKIP_ORIGIN_FETCH=0
QA_SAMPLE_SIZE=100; QA_TEST_FRACTION="0.2"; CV_FOLDS=5; CONCURRENCY=1; TIMEOUT_SECONDS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gateway-url) GATEWAY_URL="$2"; shift 2 ;;
    --gate-artifact) GATE_ARTIFACT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --skip-origin-fetch) SKIP_ORIGIN_FETCH=1; shift ;;
    --docker-image) IMAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --qa-sample-size) QA_SAMPLE_SIZE="$2"; shift 2 ;;
    --qa-test-fraction) QA_TEST_FRACTION="$2"; shift 2 ;;
    --cv-folds) CV_FOLDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$GATEWAY_URL" && -n "$GATE_ARTIFACT" ]] || { usage >&2; exit 2; }
test -f "$GATE_ARTIFACT" || { echo "3-QA gate artifact is missing: $GATE_ARTIFACT" >&2; exit 2; }
if [[ "$SKIP_ORIGIN_FETCH" -eq 1 ]]; then
  REMOTE_REF="refs/remotes/origin/$BRANCH"
  git show-ref --verify --quiet "$REMOTE_REF" || {
    echo "No locally cached origin/$BRANCH ref; cannot use --skip-origin-fetch." >&2
    exit 2
  }
  git merge-base --is-ancestor "$COMMIT" "$REMOTE_REF" || {
    echo "Commit is not present in locally cached origin/$BRANCH." >&2
    exit 2
  }
  echo "Using locally cached origin/$BRANCH after --skip-origin-fetch."
else
  git fetch --quiet origin "refs/heads/$BRANCH"
  git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || { echo "Commit is not pushed to origin/$BRANCH." >&2; exit 2; }
fi
RESOLVED_IMAGE="$(python3 scripts/resolve_datasphere_runtime_image.py --repository "${DATASPHERE_VERTEX_CPU_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu}" --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}")"
if [[ -n "$IMAGE" && "$IMAGE" != "$RESOLVED_IMAGE" ]]; then
  echo "--docker-image does not match the immutable runtime published for source commit $COMMIT." >&2
  echo "Expected: $RESOLVED_IMAGE" >&2
  exit 2
fi
IMAGE="$RESOLVED_IMAGE"
GATE_JSON="$(python3 scripts/validate_datasphere_vertex_probe_artifact.py --artifact "$GATE_ARTIFACT" --gateway-url "$GATEWAY_URL")"
GATE_COMMIT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_commit"])' "$GATE_JSON")"
git merge-base --is-ancestor "$GATE_COMMIT" "$COMMIT" || { echo "3-QA gate source commit is not an ancestor of selected full-run commit." >&2; exit 2; }
GATE_MANIFEST_SHA256="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["gateway_manifest_sha256"])' "$GATE_JSON")"
RENDERED="datasphere/jobs/rendered/vertex-cpu-qa-pilot-${RUN_ID}.yaml"
python3 scripts/render_datasphere_vertex_qa_pilot_job.py --commit "$COMMIT" --run-id "$RUN_ID" \
  --gateway-url "$GATEWAY_URL" --gateway-manifest-sha256 "$GATE_MANIFEST_SHA256" \
  --docker-image "$IMAGE" --qa-sample-size "$QA_SAMPLE_SIZE" \
  --qa-test-fraction "$QA_TEST_FRACTION" --cv-folds "$CV_FOLDS" \
  --concurrency "$CONCURRENCY" --timeout-seconds "$TIMEOUT_SECONDS" --output "$RENDERED"
python3 scripts/validate_datasphere_job.py --job "$RENDERED" --repo-root .
datasphere --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
datasphere --profile "$PROFILE" project job execute --async -p "$PROJECT_ID" -c "$RENDERED" -o "${RENDERED%.yaml}.execution.json"
