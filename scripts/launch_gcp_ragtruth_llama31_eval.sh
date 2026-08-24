#!/usr/bin/env bash
# Launch the serial container runner, monitor its redacted terminal status, then stop the VM on success.
set -Eeuo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-project-25d6be86-e826-471a-b6b}"
REGION="${GCP_REGION:-europe-west4}"
ZONE="${GCP_ZONE:-europe-west4-a}"
VM_NAME="${GCP_RAGTRUTH_VM_NAME:-hallu-ragtruth-llama31-eval}"
SERVICE_ACCOUNT_NAME="${GCP_RAGTRUTH_SERVICE_ACCOUNT:-hallu-ragtruth-runner}"
REPOSITORY="${GCP_RAGTRUTH_ARTIFACT_REPOSITORY:-hallu-ragtruth-runner}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

instance_metadata_value() {
  gcloud compute instances describe "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE" --format=json \
    | python3 -c '
import json
import sys
key = sys.argv[1]
items = json.load(sys.stdin).get("metadata", {}).get("items", [])
values = {str(item.get("key")): str(item.get("value", "")) for item in items}
print(values.get(key, ""))
' "$1"
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
  # ``create-with-container`` used a discontinued startup agent. COS retains
  # the same direct Compute Engine / persistent-disk design while this startup
  # script runs the immutable Docker digest through the VM service account.
  gcloud compute instances create "$VM_NAME" --project "$PROJECT_ID" --zone "$ZONE" \
    --machine-type=e2-medium --image-family=cos-stable --image-project=cos-cloud \
    --boot-disk-size=30GB --boot-disk-type=pd-ssd --no-boot-disk-auto-delete \
    --service-account="$SERVICE_ACCOUNT" --scopes=cloud-platform \
    --metadata="runner-image=$IMAGE,work-root=/var/lib/hallu-ragtruth-llama31,project-id=$PROJECT_ID,region=$REGION,bucket=$BUCKET,input-prefix=$INPUT_PREFIX,input-provenance-sha256=$INPUT_SHA,data-provenance-sha256=$DATA_SHA,frozen-reference-object=$FROZEN_OBJECT,frozen-reference-sha256=$FROZEN_SHA,run-id=$RUN_ID" \
    --metadata-from-file="startup-script=$ROOT/gcp/start_ragtruth_llama31_vm.sh"
else
  [[ "$RUN_ID_WAS_EXPLICIT" -eq 1 ]] || {
    echo "a stopped VM may be resumed only with its original explicit --run-id" >&2
    exit 2
  }
  # A stopped VM can resume only with its original immutable container/input
  # configuration. Validate that condition before starting it: ``gcloud
  # compute instances start`` ignores the image supplied to this launcher.
  existing_image="$(instance_metadata_value runner-image)"
  existing_run_id="$(instance_metadata_value run-id)"
  [[ "$existing_image" == "$IMAGE" ]] || {
    echo "refusing incompatible resume: stored runner image differs; use a new GCP_RAGTRUTH_VM_NAME" >&2
    exit 2
  }
  [[ "$existing_run_id" == "$RUN_ID" ]] || {
    echo "refusing incompatible resume: stored run ID differs; use a new GCP_RAGTRUTH_VM_NAME" >&2
    exit 2
  }
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
