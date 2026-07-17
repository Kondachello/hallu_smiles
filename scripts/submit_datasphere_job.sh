#!/usr/bin/env bash
# Render, validate, and submit one commit-pinned DataSphere Job.
# This script never stages a model, creates a project, or injects HF_TOKEN.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -x "$ROOT_DIR/.tools/yc/yc" ]]; then
  export PATH="$ROOT_DIR/.tools/yc:$PATH"
fi

usage() {
  cat <<'EOF'
Usage: scripts/submit_datasphere_job.sh --kind preflight|cluster-probe-g1|qa-pilot-g1 --project-id ID --run-id ID [options]

Options:
  --branch NAME       Remote branch that must contain the selected commit (default: current branch)
  --commit SHA        Full 40-character commit SHA (default: HEAD)
  --model-id ID       Hugging Face model ID (default: Meta-Llama-3.1-8B-Instruct)
  --docker-image-id ID Immutable DataSphere project Docker resource
  --docker-image REF   Public OCI image pinned by @sha256
                       (default: resolve this commit's remote GHCR build)
  --gate-artifact PATH  Required for GPU Jobs: matching successful preflight tar
                        for cluster-probe-g1, or clean 3-QA tar for qa-pilot-g1
  --profile NAME      yc/datasphere profile (default: default)
  --python PATH       Python with PyYAML/datasphere dependencies
  --datasphere PATH   DataSphere CLI executable
EOF
}

KIND=""
PROJECT_ID=""
RUN_ID=""
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
DOCKER_IMAGE_ID=""
DOCKER_IMAGE=""
GATE_ARTIFACT=""
PROFILE="default"
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
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --docker-image-id) DOCKER_IMAGE_ID="$2"; shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --gate-artifact) GATE_ARTIFACT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --datasphere) DATASPHERE_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$KIND" == "preflight" || "$KIND" == "cluster-probe-g1" || "$KIND" == "qa-pilot-g1" ]] || { echo "--kind is required" >&2; exit 2; }
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$BRANCH" ]] || { echo "--project-id, --run-id, and a branch are required" >&2; exit 2; }
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "--commit must be a lowercase full SHA" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
command -v "$DATASPHERE_BIN" >/dev/null || { echo "DataSphere CLI not found: $DATASPHERE_BIN" >&2; exit 2; }
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-native}"

# A Job fetches the public remote by SHA. Fail locally if the requested branch
# does not yet contain that SHA instead of waiting for an image or consuming an instance.
git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || {
  echo "Commit $COMMIT is not pushed to origin/$BRANCH; push the branch first." >&2
  exit 2
}

if [[ -n "$DOCKER_IMAGE_ID" && -n "$DOCKER_IMAGE" ]]; then
  echo "Use only one of --docker-image-id and --docker-image" >&2
  exit 2
fi
if [[ -n "$DOCKER_IMAGE_ID" ]]; then
  [[ "$DOCKER_IMAGE_ID" =~ ^b[a-z0-9]{19}$ ]] || {
    echo "--docker-image-id must be a DataSphere project Docker resource ID (b + 19 lowercase letters/digits)" >&2
    exit 2
  }
  RUNTIME_IMAGE="$DOCKER_IMAGE_ID"
  IMAGE_ARG=(--docker-image-id "$DOCKER_IMAGE_ID")
elif [[ -n "$DOCKER_IMAGE" ]]; then
  [[ "$DOCKER_IMAGE" =~ ^[a-z0-9][a-z0-9.-]*(:[0-9]+)?(/[a-z0-9][a-z0-9._-]*)+@sha256:[0-9a-f]{64}$ ]] || {
    echo "--docker-image must be a lowercase registry reference pinned by @sha256" >&2
    exit 2
  }
  RUNTIME_IMAGE="$DOCKER_IMAGE"
  IMAGE_ARG=(--docker-image "$DOCKER_IMAGE")
else
  RUNTIME_IMAGE="$(
    "$PYTHON_BIN" scripts/resolve_datasphere_runtime_image.py \
      --repository "${DATASPHERE_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere}" \
      --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}"
  )"
  DOCKER_IMAGE="$RUNTIME_IMAGE"
  IMAGE_ARG=(--docker-image "$DOCKER_IMAGE")
  echo "Resolved immutable remote runtime: $RUNTIME_IMAGE"
fi

if [[ "$KIND" == "cluster-probe-g1" ]]; then
  [[ -n "$GATE_ARTIFACT" ]] || {
    echo "--gate-artifact with the successful matching preflight tar is required for cluster-probe-g1" >&2
    exit 2
  }
  "$PYTHON_BIN" scripts/validate_datasphere_gate_artifact.py \
    --gate preflight --artifact "$GATE_ARTIFACT" --commit "$COMMIT" \
    "${IMAGE_ARG[@]}" --model-id "$MODEL_ID"
elif [[ "$KIND" == "qa-pilot-g1" ]]; then
  [[ -n "$GATE_ARTIFACT" ]] || {
    echo "--gate-artifact with the clean matching 3-QA tar is required for qa-pilot-g1" >&2
    exit 2
  }
  "$PYTHON_BIN" scripts/validate_datasphere_gate_artifact.py \
    --gate cluster-probe-g1 --artifact "$GATE_ARTIFACT" --commit "$COMMIT" \
    "${IMAGE_ARG[@]}" --model-id "$MODEL_ID"
elif [[ -n "$GATE_ARTIFACT" ]]; then
  echo "--gate-artifact is not used for the CPU preflight" >&2
  exit 2
fi

rendered="datasphere/jobs/rendered/${KIND}-${RUN_ID}.yaml"
"$PYTHON_BIN" scripts/render_datasphere_job.py --kind "$KIND" --commit "$COMMIT" \
  --model-id "$MODEL_ID" "${IMAGE_ARG[@]}" \
  --run-id "$RUN_ID" --output "$rendered"
"$PYTHON_BIN" scripts/validate_datasphere_job.py --job "$rendered" --repo-root .
"$DATASPHERE_BIN" --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
if [[ "$KIND" == "cluster-probe-g1" || "$KIND" == "qa-pilot-g1" ]]; then
  job_list_json="datasphere/jobs/rendered/active-jobs-${RUN_ID}.json"
  "$DATASPHERE_BIN" --profile "$PROFILE" project job list \
    -p "$PROJECT_ID" --format json -o "$job_list_json"
  "$PYTHON_BIN" scripts/check_datasphere_active_jobs.py --jobs-json "$job_list_json"
fi
execution_metadata="datasphere/jobs/rendered/${KIND}-${RUN_ID}.execution.json"
"$DATASPHERE_BIN" --profile "$PROFILE" project job execute --async \
  -p "$PROJECT_ID" -c "$rendered" -o "$execution_metadata"
echo "DataSphere execution metadata: $execution_metadata"
"$PYTHON_BIN" -m json.tool "$execution_metadata"
