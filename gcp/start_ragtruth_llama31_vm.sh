#!/usr/bin/env bash
# COS startup script: run the pinned research container from VM metadata.
set -Eeuo pipefail

metadata() {
  curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

image="$(metadata runner-image)"
work_root="$(metadata work-root)"
mkdir -p "$work_root"
docker-credential-gcr configure-docker --registries=europe-west4-docker.pkg.dev
docker pull "$image"
docker run --rm --name hallu-ragtruth-llama31 \
  --mount "type=bind,src=$work_root,dst=/work" \
  --env "GCP_PROJECT_ID=$(metadata project-id)" \
  --env "GCP_REGION=$(metadata region)" \
  --env "GCP_RAGTRUTH_BUCKET=$(metadata bucket)" \
  --env "GCP_RAGTRUTH_INPUT_PREFIX=$(metadata input-prefix)" \
  --env "GCP_RAGTRUTH_INPUT_PROVENANCE_SHA256=$(metadata input-provenance-sha256)" \
  --env "GCP_RAGTRUTH_DATA_PROVENANCE_SHA256=$(metadata data-provenance-sha256)" \
  --env "GCP_RAGTRUTH_FROZEN_REFERENCE_OBJECT=$(metadata frozen-reference-object)" \
  --env "GCP_RAGTRUTH_FROZEN_REFERENCE_SHA256=$(metadata frozen-reference-sha256)" \
  --env "GCP_RAGTRUTH_RUN_ID=$(metadata run-id)" \
  "$image"
