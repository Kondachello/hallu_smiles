#!/usr/bin/env bash
# Render, validate, and submit the independent CPU Vertex three-QA probe.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Some home networks advertise IPv6 but cannot route it to the DataSphere API.
# The native gRPC resolver reliably falls back to the reachable IPv4 endpoint.
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-native}"

usage() { echo "Usage: $0 --project-id ID --run-id ID --gateway-url HTTPS_URL [--commit SHA] [--branch NAME] [--docker-image REF]"; }
PROJECT_ID=""; RUN_ID=""; GATEWAY_URL=""; BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; IMAGE=""; PROFILE="default"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gateway-url) GATEWAY_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --docker-image) IMAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$GATEWAY_URL" ]] || { usage >&2; exit 2; }
git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || { echo "Commit is not pushed to origin/$BRANCH." >&2; exit 2; }
RESOLVED_IMAGE="$(python3 scripts/resolve_datasphere_runtime_image.py --repository "${DATASPHERE_VERTEX_CPU_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu}" --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}")"
if [[ -n "$IMAGE" && "$IMAGE" != "$RESOLVED_IMAGE" ]]; then
  echo "--docker-image does not match the immutable runtime published for source commit $COMMIT." >&2
  echo "Expected: $RESOLVED_IMAGE" >&2
  exit 2
fi
IMAGE="$RESOLVED_IMAGE"
RENDERED="datasphere/jobs/rendered/vertex-cpu-probe-${RUN_ID}.yaml"
python3 scripts/render_datasphere_vertex_probe_job.py --commit "$COMMIT" --run-id "$RUN_ID" --gateway-url "$GATEWAY_URL" --docker-image "$IMAGE" --output "$RENDERED"
python3 scripts/validate_datasphere_job.py --job "$RENDERED" --repo-root .
datasphere --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
datasphere --profile "$PROFILE" project job execute --async -p "$PROJECT_ID" -c "$RENDERED" -o "${RENDERED%.yaml}.execution.json"
