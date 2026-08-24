#!/usr/bin/env bash
# Build an immutable CPU runner image from the committed source only.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
REGION="${GCP_REGION:-europe-west4}"
REPOSITORY="${GCP_RAGTRUTH_ARTIFACT_REPOSITORY:-hallu-ragtruth-runner}"

usage() {
  echo "Usage: bash scripts/build_gcp_ragtruth_llama31_runner.sh" >&2
}

[[ $# -eq 0 ]] || { usage; exit 2; }
command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 2; }
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
if ! git -C "$ROOT" ls-remote origin | awk '{print $1}' | grep -qx "$SOURCE_COMMIT"; then
  echo "refusing GCP build: source commit is not pushed to origin" >&2
  exit 2
fi
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/ragtruth-llama31:$SOURCE_COMMIT"
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
git -C "$ROOT" archive --format=tar "$SOURCE_COMMIT" | tar -xf - -C "$BUILD_ROOT"
gcloud builds submit "$BUILD_ROOT" --project "$PROJECT_ID" --region "$REGION" \
  --ignore-file "$BUILD_ROOT/gcp/cloudbuild.gcloudignore" \
  --config "$BUILD_ROOT/gcp/cloudbuild.ragtruth-runner.yaml" \
  --substitutions "_SOURCE_COMMIT=$SOURCE_COMMIT,_IMAGE=$IMAGE"
DIGEST="$(gcloud artifacts docker images describe "$IMAGE" --project "$PROJECT_ID" --format='value(image_summary.digest)')"
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "failed to resolve immutable image digest" >&2; exit 1; }
printf '%s\n' "$IMAGE@$DIGEST"
