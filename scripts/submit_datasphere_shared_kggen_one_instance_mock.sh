#!/usr/bin/env bash
# Render, validate, and submit the CPU-only two-pass shared-KGGen mock probe.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-ares}"

usage() { echo "Usage: $0 --project-id ID --run-id ID --response-id ID [--commit SHA] [--branch NAME] [--docker-image REF] [--profile NAME]"; }
PROJECT_ID=""; RUN_ID=""; RESPONSE_ID=""; BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; IMAGE=""; PROFILE="default"
PYTHON_BIN="${PYTHON_BIN:-python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --response-id) RESPONSE_ID="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --docker-image) IMAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$RESPONSE_ID" ]] || { usage >&2; exit 2; }

git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || { echo "Commit is not pushed to origin/$BRANCH." >&2; exit 2; }
RESOLVED_IMAGE="$($PYTHON_BIN scripts/resolve_datasphere_runtime_image.py --repository "${DATASPHERE_VERTEX_CPU_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu}" --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}")"
if [[ -n "$IMAGE" && "$IMAGE" != "$RESOLVED_IMAGE" ]]; then
  echo "--docker-image does not match the immutable runtime published for source commit $COMMIT." >&2
  exit 2
fi
RENDERED="datasphere/jobs/rendered/shared-kggen-one-instance-mock-${RUN_ID}.yaml"
$PYTHON_BIN scripts/render_datasphere_shared_kggen_one_instance_mock_job.py \
  --commit "$COMMIT" --run-id "$RUN_ID" --response-id "$RESPONSE_ID" \
  --docker-image "$RESOLVED_IMAGE" --output "$RENDERED"
$PYTHON_BIN scripts/validate_datasphere_job.py --job "$RENDERED" --repo-root .
datasphere --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
datasphere --profile "$PROFILE" project job execute --async -p "$PROJECT_ID" -c "$RENDERED" \
  -o "${RENDERED%.yaml}.execution.json"
