#!/usr/bin/env bash
# Render, validate, and submit one commit-pinned DataSphere Job.
# This script never stages a model, creates a project, or injects HF_TOKEN.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit_datasphere_job.sh --kind preflight|qa-pilot-g1 --project-id ID --run-id ID [options]

Options:
  --branch NAME       Remote branch that must contain the selected commit (default: current branch)
  --commit SHA        Full 40-character commit SHA (default: HEAD)
  --model-id ID       Hugging Face model ID (default: Meta-Llama-3.1-8B-Instruct)
  --profile NAME      yc/datasphere profile (default: default)
  --python PATH       Python with PyYAML/datasphere dependencies (default: python3)
  --datasphere PATH   DataSphere CLI executable (default: datasphere)
EOF
}

KIND=""
PROJECT_ID=""
RUN_ID=""
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
PROFILE="default"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASPHERE_BIN="${DATASPHERE_BIN:-datasphere}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --datasphere) DATASPHERE_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$KIND" == "preflight" || "$KIND" == "qa-pilot-g1" ]] || { echo "--kind is required" >&2; exit 2; }
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$BRANCH" ]] || { echo "--project-id, --run-id, and a branch are required" >&2; exit 2; }
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "--commit must be a lowercase full SHA" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
command -v "$DATASPHERE_BIN" >/dev/null || { echo "DataSphere CLI not found: $DATASPHERE_BIN" >&2; exit 2; }

# A Job fetches the public remote by SHA. Fail locally if the requested branch
# does not yet contain that SHA instead of consuming an instance for a bad ref.
git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || {
  echo "Commit $COMMIT is not pushed to origin/$BRANCH; push the branch first." >&2
  exit 2
}

rendered="datasphere/jobs/rendered/${KIND}-${RUN_ID}.yaml"
"$PYTHON_BIN" scripts/render_datasphere_job.py --kind "$KIND" --commit "$COMMIT" \
  --model-id "$MODEL_ID" --run-id "$RUN_ID" --output "$rendered"
"$PYTHON_BIN" scripts/validate_datasphere_job.py --job "$rendered" --repo-root .
"$DATASPHERE_BIN" --profile "$PROFILE" project get --id "$PROJECT_ID" >/dev/null
"$DATASPHERE_BIN" --profile "$PROFILE" project job execute --async -p "$PROJECT_ID" -c "$rendered"
