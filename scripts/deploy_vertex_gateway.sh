#!/usr/bin/env bash
# Build and deploy the Cloud Run gateway. It never creates or prints secrets.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/deploy_vertex_gateway.sh --project ID --service-account EMAIL --secret NAME --release ID [options]

Required Google Cloud setup (performed once): billing, Vertex AI, Cloud Run,
Cloud Build, Artifact Registry and Secret Manager APIs are enabled; SECRET
contains the gateway bearer key; SERVICE_ACCOUNT has roles/aiplatform.user and
roles/secretmanager.secretAccessor. The deployer needs permission to attach it.

Options:
  --region REGION       Cloud Run and Vertex location (default: europe-west4)
  --service NAME        Cloud Run service name (default: hallu-vertex-gateway)
  --repository NAME     Artifact Registry Docker repository (default: hallu-gateway)
  --image IMAGE         Use this immutable/pinned image instead of Cloud Build
EOF
}

PROJECT=""
SERVICE_ACCOUNT=""
SECRET=""
RELEASE=""
REGION="europe-west4"
SERVICE="hallu-vertex-gateway"
REPOSITORY="hallu-gateway"
IMAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --secret) SECRET="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --repository) REPOSITORY="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PROJECT" && -n "$SERVICE_ACCOUNT" && -n "$SECRET" && -n "$RELEASE" ]] || {
  echo "--project, --service-account, --secret, and --release are required." >&2
  exit 2
}
[[ "$REGION" == "europe-west4" ]] || {
  echo "This reproducible profile is pinned to europe-west4." >&2
  exit 2
}
command -v gcloud >/dev/null || { echo "gcloud is required." >&2; exit 2; }
gcloud artifacts repositories describe "$REPOSITORY" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1 || {
  echo "Artifact Registry Docker repository '$REPOSITORY' is missing in $REGION." >&2
  echo "Create it first: gcloud artifacts repositories create '$REPOSITORY' --repository-format=docker --location='$REGION' --project='$PROJECT'" >&2
  exit 2
}
LOGICAL_MODEL="$(python3 - <<'PY'
import yaml
with open('config.yaml', encoding='utf-8') as handle:
    print(yaml.safe_load(handle)['llm']['model'])
PY
)"
[[ "$LOGICAL_MODEL" == openai/gemini-2.5-flash ]] || {
  echo "config.yaml llm.model must be openai/gemini-2.5-flash for this profile." >&2
  exit 2
}

if [[ -z "$IMAGE" ]]; then
  IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${SERVICE}:${RELEASE}"
  gcloud builds submit --project "$PROJECT" --config gateway/cloudbuild.yaml \
    --substitutions "_IMAGE=${IMAGE}" .
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$SERVICE_ACCOUNT" --allow-unauthenticated \
  --concurrency 2 --max-instances 2 --timeout 120s \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=${REGION},HALLU_LOGICAL_MODEL=${LOGICAL_MODEL},HALLU_GATEWAY_RELEASE=${RELEASE}" \
  --set-secrets "HALLU_GATEWAY_API_KEY=${SECRET}:latest"

echo "Gateway URL (store as the non-secret --gateway-url for DataSphere):"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)'
