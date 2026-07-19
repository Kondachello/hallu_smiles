# Vertex AI Cloud Run gateway and DataSphere 3-QA probe

This is the active route for the first real run. DataSphere only orchestrates
KGGen, local CPU S-BERT, cache replay and reports. Gemini inference is called
from Cloud Run through Vertex AI; neither DataSphere nor the repository gets a
Google service-account key.

## 1. One-time Google Cloud setup

Choose a billed project and use the same `europe-west4` region for Cloud Run
and Vertex AI. Enable the APIs:

```bash
export GCP_PROJECT=<your-project-id>
gcloud services enable --project "$GCP_PROJECT" \
  aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud artifacts repositories create hallu-gateway --project "$GCP_PROJECT" \
  --repository-format=docker --location=europe-west4
```

Create a dedicated Cloud Run service account and grant only the roles needed by
the gateway. The deployer also needs permission to attach this account.

```bash
gcloud iam service-accounts create hallu-vertex-gateway --project "$GCP_PROJECT"
export GATEWAY_SA="hallu-vertex-gateway@$GCP_PROJECT.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
  --member="serviceAccount:$GATEWAY_SA" --role=roles/aiplatform.user
```

Generate a high-entropy bearer value locally and create a Secret Manager
secret. Do not put the value in a shell profile, repository file, Job YAML, or
Cloud Run environment variable.

```bash
export HALLU_GATEWAY_API_KEY="$(openssl rand -hex 32)"
printf %s "$HALLU_GATEWAY_API_KEY" | gcloud secrets create hallu-gateway-api-key \
  --project "$GCP_PROJECT" --data-file=-
gcloud secrets add-iam-policy-binding hallu-gateway-api-key --project "$GCP_PROJECT" \
  --member="serviceAccount:$GATEWAY_SA" --role=roles/secretmanager.secretAccessor
```

Create a DataSphere Project secret named exactly `HALLU_GATEWAY_API_KEY` with
the **same value**. DataSphere injects Project secrets into Job environments;
the new Job template never lists it. If the project has isolated networking,
configure NAT so its CPU Job can reach the public Cloud Run HTTPS URL.
Keep the exported value only in this setup shell for the authenticated probe
below, then run `unset HALLU_GATEWAY_API_KEY` after that probe succeeds.

## 2. Deploy and verify the gateway

Run this from a commit that contains the implementation. `RELEASE` should be
the full Git commit, so the Cloud Run manifest and cache identity point to the
same source revision.

```bash
export RELEASE="$(git rev-parse HEAD)"
bash scripts/deploy_vertex_gateway.sh \
  --project "$GCP_PROJECT" --service-account "$GATEWAY_SA" \
  --secret hallu-gateway-api-key --release "$RELEASE"
```

The command prints the Cloud Run origin. Save it as `GATEWAY_URL`; it is not a
secret. Verify the bearer auth and immutable runtime manifest without verbose
curl output:

```bash
export GATEWAY_URL=https://<service>-<hash>.europe-west4.run.app
curl --fail --silent --show-error -H "Authorization: Bearer $HALLU_GATEWAY_API_KEY" \
  "$GATEWAY_URL/v1/hallu/manifest"
```

The gateway also implements authenticated `GET /healthz` for its application
contract, but do not use that path through a public Cloud Run `run.app` URL:
the Google Front End intercepts it before the request reaches the container.
The authenticated manifest is therefore the readiness and identity probe used
by the DataSphere Job.

Cloud Run is publicly routable only because DataSphere cannot present a Google
OIDC invocation token. Every app route checks the bearer secret before reading
or forwarding a prompt. The assigned Cloud Run service account uses ADC to
call Vertex AI; never set `GOOGLE_APPLICATION_CREDENTIALS` in the service.

## 3. Start the DataSphere CPU 3-QA probe

Push the selected commit to `new-metrics`, wait for the **Build DataSphere
Vertex CPU runtime image** GitHub Actions workflow, then submit:

```bash
bash scripts/submit_datasphere_vertex_probe.sh \
  --project-id <DATASPHERE_PROJECT_ID> \
  --run-id vertex-3qa-$(date +%Y%m%d) \
  --gateway-url "$GATEWAY_URL"
```

The submitter resolves the CPU image by commit digest. If `--docker-image
ghcr.io/...@sha256:...` is supplied, it must equal that resolved image; the
submitter rejects a stale digest before it can create a Job.

It also defaults `GRPC_DNS_RESOLVER=native`, so a laptop network with unusable
IPv6 reaches the DataSphere API over IPv4. No manual network setting is needed.

The Job validates its CPU runtime and authenticated gateway manifest, runs the
synthetic KGGen clustering probe, runs the verifier once live and once from its
verdict cache, extracts `G_c`, `G_q`, and `G_a` for the deterministic first
three entries of the fixed 20-QA manifest, then reruns extraction with
`--cache-only`.

Success requires an empty `failed_extractions.jsonl`, exactly three complete
graph pairs, a cache-only extraction summary identical to the live summary, and
unchanged cache hashes. The archive contains `gateway-manifest.json`,
`runtime_config.yaml` (without a secret), `kggen-vertex-probe.json`,
`verifier-vertex-probe.json`, both extraction summaries, cache hashes, and
`usage-counts.json` (including zero live calls for both cache-only replays),
and `run_metadata.json`.

This is not a tuning/evaluation run. The existing strict-vs-support comparison
is reserved for the later 20-QA manifest, whose 16 train and 4 test records are
needed to preserve the no-test-leakage protocol.
