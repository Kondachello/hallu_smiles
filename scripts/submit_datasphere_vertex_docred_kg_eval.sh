#!/usr/bin/env bash
# Validate a successful bounded gateway gate, then submit the fixed DocRED KG run.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-ares}"
# Never fall back to the subject-id profile: it can open an interactive browser
# login. DataSphere CLI consumes YC_TOKEN / YC_OAUTH_TOKEN directly and creates
# its short-lived IAM token internally.
export YC_AUTH=OAUTH

usage() {
  cat <<'EOF'
Usage: scripts/submit_datasphere_vertex_docred_kg_eval.sh --project-id ID --run-id ID --gateway-url HTTPS_URL --gate-artifact PATH [options]

The manifest is fixed: 50 train_annotated calibration documents (first 10 are
the smoke stage), 200 held-out dev documents, seed 42, serial extraction, and
a €10.5 maximum live-inference estimate.

Options:
  --budget-eur N      Conservative live Gemini budget, (0, 10.5] (default: 10.5)
  --commit SHA        Pushed source commit (default: HEAD)
  --branch NAME       Remote branch containing the commit (default: current)
  --docker-image REF  Optional immutable image; must match the committed build

Authentication is non-interactive. Set YC_TOKEN (or YC_OAUTH_TOKEN) through the
host secret environment; the token is never accepted as a command-line argument.
EOF
}

PROJECT_ID=""; RUN_ID=""; GATEWAY_URL=""; GATE_ARTIFACT=""
BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"
IMAGE=""; BUDGET_EUR="10.5"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gateway-url) GATEWAY_URL="$2"; shift 2 ;;
    --gate-artifact) GATE_ARTIFACT="$2"; shift 2 ;;
    --budget-eur) BUDGET_EUR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --docker-image) IMAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT_ID" && -n "$RUN_ID" && -n "$GATEWAY_URL" && -n "$GATE_ARTIFACT" ]] || {
  usage >&2; exit 2;
}
: "${YC_TOKEN:=${YC_OAUTH_TOKEN:?Set secret YC_TOKEN or YC_OAUTH_TOKEN for non-interactive DataSphere authentication}}"
test -f "$GATE_ARTIFACT" || { echo "gateway gate artifact is missing: $GATE_ARTIFACT" >&2; exit 2; }
python3 - "$BUDGET_EUR" <<'PY'
import sys
budget = float(sys.argv[1])
if not 0.0 < budget <= 10.5:
    raise SystemExit('budget must be in (0, 10.5]')
PY

git fetch --quiet origin "refs/heads/$BRANCH"
git merge-base --is-ancestor "$COMMIT" FETCH_HEAD || {
  echo "Commit is not pushed to origin/$BRANCH." >&2; exit 2;
}
RESOLVED_IMAGE="$(python3 scripts/resolve_datasphere_runtime_image.py \
  --repository "${DATASPHERE_VERTEX_CPU_RUNTIME_REPOSITORY:-ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu}" \
  --commit "$COMMIT" --wait-seconds "${DATASPHERE_RUNTIME_WAIT_SECONDS:-1800}")"
if [[ -n "$IMAGE" && "$IMAGE" != "$RESOLVED_IMAGE" ]]; then
  echo "--docker-image does not match the immutable runtime published for source commit $COMMIT." >&2
  echo "Expected: $RESOLVED_IMAGE" >&2
  exit 2
fi
IMAGE="$RESOLVED_IMAGE"
GATE_JSON="$(python3 scripts/validate_datasphere_vertex_probe_artifact.py --artifact "$GATE_ARTIFACT" --gateway-url "$GATEWAY_URL")"
GATE_COMMIT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["source_commit"])' "$GATE_JSON")"
git merge-base --is-ancestor "$GATE_COMMIT" "$COMMIT" || {
  echo "Gateway gate source commit is not an ancestor of selected DocRED commit." >&2; exit 2;
}
GATE_MANIFEST_SHA256="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["gateway_manifest_sha256"])' "$GATE_JSON")"
RENDERED="datasphere/jobs/rendered/vertex-cpu-docred-kg-${RUN_ID}.yaml"
python3 scripts/render_datasphere_vertex_docred_kg_eval_job.py \
  --commit "$COMMIT" --run-id "$RUN_ID" --gateway-url "$GATEWAY_URL" \
  --gateway-manifest-sha256 "$GATE_MANIFEST_SHA256" --budget-eur "$BUDGET_EUR" \
  --docker-image "$IMAGE" --output "$RENDERED"
python3 scripts/validate_datasphere_job.py --job "$RENDERED" --repo-root .
datasphere project get --id "$PROJECT_ID" >/dev/null
datasphere project job execute --async -p "$PROJECT_ID" -c "$RENDERED" \
  -o "${RENDERED%.yaml}.execution.json"
