#!/usr/bin/env bash
# Provision the minimum persistent GCP resources for one controlled CPU run.
set -Eeuo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
REGION="${GCP_REGION:-europe-west4}"
ZONE="${GCP_ZONE:-europe-west4-a}"
REPOSITORY="${GCP_RAGTRUTH_ARTIFACT_REPOSITORY:-hallu-ragtruth-runner}"
SERVICE_ACCOUNT_NAME="${GCP_RAGTRUTH_SERVICE_ACCOUNT:-hallu-ragtruth-runner}"
SECRET_NAME="${HALLU_GATEWAY_SECRET:-hallu-docred-gateway-bearer}"

command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 2; }
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
BUCKET="${GCP_RAGTRUTH_BUCKET:-hallu-ragtruth-llama31-$PROJECT_NUMBER-euw4}"
SERVICE_ACCOUNT="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
CLOUD_BUILD_SERVICE_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud services enable compute.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com logging.googleapis.com --project "$PROJECT_ID"
if ! gcloud artifacts repositories describe "$REPOSITORY" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" --project "$PROJECT_ID" --location "$REGION" --repository-format=docker
fi
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT_ID" --location="$REGION" --uniform-bucket-level-access
fi
# No lifecycle rule is installed: inputs, frozen reference artifacts and terminal
# archives are deliberately non-expiring. Public access is explicitly prevented.
gcloud storage buckets update "gs://$BUCKET" --project "$PROJECT_ID" --public-access-prevention
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" --project "$PROJECT_ID" --display-name="Hallu RAGTruth Llama31 CPU runner"
fi

gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/artifactregistry.reader --condition=None >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/logging.logWriter --condition=None >/dev/null
# Cloud Build is a separate build principal; writer access is repository-scoped
# and is intentionally not granted to the VM runner service account.
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --project "$PROJECT_ID" --location "$REGION" \
  --member="serviceAccount:$CLOUD_BUILD_SERVICE_ACCOUNT" --role=roles/artifactregistry.writer --condition=None >/dev/null
gcloud secrets add-iam-policy-binding "$SECRET_NAME" --project "$PROJECT_ID" --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/secretmanager.secretAccessor >/dev/null

# Object access is prefix-scoped. The runner has no project-level storage role,
# no Vertex role, and no permission to alter its service account or VM.
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/storage.objectViewer \
  --condition="expression=resource.name.startsWith('projects/_/buckets/$BUCKET/objects/inputs/') || resource.name.startsWith('projects/_/buckets/$BUCKET/objects/frozen-reference/'),title=ragtruth_llama31_inputs,description=Read-only controlled inputs and frozen references" >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/storage.objectCreator \
  --condition="expression=resource.name.startsWith('projects/_/buckets/$BUCKET/objects/terminal-archives/'),title=ragtruth_llama31_archives,description=Create-only redacted terminal archives" >/dev/null

cat <<EOF
project=$PROJECT_ID
region=$REGION
zone=$ZONE
bucket=$BUCKET
service_account=$SERVICE_ACCOUNT
cloud_build_service_account=$CLOUD_BUILD_SERVICE_ACCOUNT
artifact_repository=$REPOSITORY
machine_type=e2-medium
boot_disk=pd-ssd:30GB
EOF
