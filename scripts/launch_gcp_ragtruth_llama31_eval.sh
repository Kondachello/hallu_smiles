#!/usr/bin/env bash
# Launch the serial container runner, monitor its redacted terminal status, then stop the VM on success.
set -Eeuo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
REGION="${GCP_REGION:-europe-west4}"
ZONE="${GCP_ZONE:-europe-west4-a}"
VM_NAME="${GCP_RAGTRUTH_VM_NAME:-hallu-ragtruth-llama31-eval}"
SERVICE_ACCOUNT_NAME="${GCP_RAGTRUTH_SERVICE_ACCOUNT:-hallu-ragtruth-runner}"
REPOSITORY="${GCP_RAGTRUTH_ARTIFACT_REPOSITORY:-hallu-ragtruth-runner}"

usage() {
  echo "Usage: bash scripts/launch_gcp_ragtruth_llama31_eval.sh --staging-manifest PATH --image IMAGE@sha256:... [--run-id ID]" >&2
}

STAGING=""; IMAGE=""; RUN_ID=""; RUN_ID_WAS_EXPLICIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging-manifest) STAGING="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; RUN_ID_WAS_EXPLICIT=1; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$STAGING" && -n "$IMAGE" ]] || { usage; exit 2; }
[[ -f "$STAGING" ]] || { echo "staging manifest is missing" >&2; exit 2; }
[[ "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo "image must be pinned by immutable digest" >&2; exit 2; }
command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 2; }
[[ -n "$RUN_ID" ]] || RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

manifest_value() {
  python3 - "$STAGING" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))[sys.argv[2]])
PY
}
BUCKET="$(manifest_value bucket)"
INPUT_PREFIX="$(manifest_value input_prefix)"
INPUT_SHA="$(manifest_value input_provenance_sha256)"
DATA_SHA="$(manifest_value data_provenance_sha256)"
FROZEN_OBJECT="$(manifest_value frozen_reference_object)"
FROZEN_SHA="$(manifest_value frozen_reference_sha256)"
SERVICE_ACCOUNT="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

state="$(gcloud compute instances describe "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE" --format='value(status)' 2>/dev/null || true)"
if [[ "$state" == "RUNNING" ]]; then
  echo "refusing overlapping runner: $VM_NAME is already RUNNING" >&2
  exit 2
fi
if [[ -z "$state" ]]; then
  gcloud compute instances create-with-container "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE" \
    --machine-type=e2-medium --boot-disk-size=30GB --boot-disk-type=pd-ssd --no-boot-disk-auto-delete \
    --service-account="$SERVICE_ACCOUNT" --scopes=cloud-platform --container-image="$IMAGE" --container-restart-policy=never \
    --container-mount-host-path=mount-path=/work,host-path=/var/lib/hallu-ragtruth-llama31,mode=rw \
    --container-env="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION,GCP_RAGTRUTH_BUCKET=$BUCKET,GCP_RAGTRUTH_INPUT_PREFIX=$INPUT_PREFIX,GCP_RAGTRUTH_INPUT_PROVENANCE_SHA256=$INPUT_SHA,GCP_RAGTRUTH_DATA_PROVENANCE_SHA256=$DATA_SHA,GCP_RAGTRUTH_FROZEN_REFERENCE_OBJECT=$FROZEN_OBJECT,GCP_RAGTRUTH_FROZEN_REFERENCE_SHA256=$FROZEN_SHA,GCP_RAGTRUTH_RUN_ID=$RUN_ID"
else
  [[ "$RUN_ID_WAS_EXPLICIT" -eq 1 ]] || {
    echo "a stopped VM may be resumed only with its original explicit --run-id" >&2
    exit 2
  }
  # A stopped VM can resume only with its original immutable container/input
  # configuration. Use the same run ID; a different one requires a new VM name.
  gcloud compute instances start "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE"
fi

while true; do
  serial="$(gcloud compute instances get-serial-port-output "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE" --port=1 2>/dev/null || true)"
  if grep -q 'RUN_COMPLETE archive_object=' <<<"$serial"; then
    gcloud compute instances stop "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE"
    echo "[ok] runner archived successfully and VM was stopped; persistent pd-ssd disk retained"
    exit 0
  fi
  if grep -q 'RUN_FAILED archive_object=' <<<"$serial"; then
    echo "[failed] runner preserved its cache and redacted archive; VM left available for compatible inspection/resume" >&2
    exit 1
  fi
  sleep 30
done
