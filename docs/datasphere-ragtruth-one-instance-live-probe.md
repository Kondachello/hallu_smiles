# Live one-instance RAGTruth probe in DataSphere

This is the only live smoke test for the paired experiment framework. It executes
exactly one explicit RAGTruth `response_id` through the real HalluGraph and GraphEval
paths, then archives enough redacted diagnostics to audit the outcome.

It is intentionally not a sample run, evaluation, threshold-tuning procedure, or
full-dataset execution. The rendered Job has one response id embedded in it and the
Python entrypoint exposes no `--limit`, sampling, evaluation, or fake-backend switch.

## Backend topology

```text
RAGTruth in Project storage (one response_id)
        │
        ├── HalluGraph: real KGGen -> Cloud Run gateway -> Vertex Gemini
        │                 + local pinned MiniLM embeddings
        │
        └── GraphEval: real gateway extractor -> Cloud Run gateway -> Vertex Gemini
                       + local pinned HHEM-2.1-Open NLI
```

Both methods receive the exact same materialized `DetectionInput`. RAGTruth `labels`
and `quality` are not copied into this input. There is no gold join or metric step.

## Secret and network contract

The code reads two environment variables only at runtime:

- `HALLU_GATEWAY_API_KEY` is the existing DataSphere **Project secret**. DataSphere
  injects it into the Job automatically; no `.env` file, CLI flag, YAML value, Git
  variable, or local secret copy is used.
- `HALLU_GATEWAY_URL` is the public Cloud Run origin and is written by the rendered
  Job template. The GraphEval gateway client appends `/v1` itself.

The shell wrapper confirms the secret can authenticate `GET /v1/hallu/manifest`, but
never prints the bearer token. The manifest is validated against the logical Gemini
model and hashed into the archive/cache identity. Python also redacts the secret from
audit details and prediction/stage files as defence in depth.

Do not use `set -x`, `curl -v`, a local `.env`, `GOOGLE_APPLICATION_CREDENTIALS`, a
Google service-account JSON, or a direct Vertex API call.

## Runtime prerequisites

Before submission all of these must be true:

1. The source commit has been pushed to the selected GitHub branch. DataSphere clones
   that immutable commit, not the local worktree.
2. CI has built the CPU runtime image for that same commit. The updated image includes
   `openai`, the local MiniLM snapshot, a pinned HHEM snapshot at
   `0e7edb3689e710c52ba120086e8f91ea3ee87f23`, and the HHEM custom model's required
   pinned `google/flan-t5-base` config/tokenizer directory at
   `d224e0d50f2fe7d975c973cf46d933e4dfaf2a3e`; runtime networking to Hugging Face is
   disabled.
   During the image build, HHEM's custom-model config is explicitly bound to that
   local directory; it never relies on a version-dependent Hugging Face cache layout.
3. The Project disk contains the approved read-only RAGTruth files at
   `$DS_PROJECT_HOME/hallu_smiles/shared/ragtruth/source_info.jsonl` and
   `$DS_PROJECT_HOME/hallu_smiles/shared/ragtruth/response.jsonl`.
4. Project secret `HALLU_GATEWAY_API_KEY` is available to Jobs. Its value must never
   be requested or copied into this repository.

The Job fails before inference if any prerequisite, gateway identity, HHEM snapshot,
or selected `response_id` is missing. A failure still produces an archive containing
the shell logs and all diagnostics completed before failure.

## Submission

Choose one exact ID from the shared `response.jsonl`, a fresh lowercase `RUN_ID`, and
the approved gateway origin. From the repository root:

```bash
bash scripts/submit_datasphere_ragtruth_one_instance_live_probe.sh \
  --project-id bt1i64odluitglbaj5st \
  --run-id paired-live-<short-sha>-<date> \
  --response-id <exact-response-id> \
  --gateway-url https://hallu-vertex-gateway-453887629111.europe-west4.run.app
```

The submitter verifies that the selected commit is already reachable from the remote
branch, waits for the immutable runtime image for that commit, renders a Job YAML,
and validates the YAML before invoking the DataSphere CLI. It does not submit a full
RAGTruth run.

## Audit artifacts

The downloaded `ragtruth-one-instance-live-<RUN_ID>.tar.gz` contains at least:

```text
live.stdout.log / live.stderr.log
cpu-runtime.json
hhem-offline-smoke.json
gateway-manifest.json
runtime_config.yaml                 # redacted: env-var name, never secret value
runtime-config-identity.json
paired-live/
  run_manifest.json
  instances.no_gold.jsonl
  audit/live_one_instance_events.jsonl
  audit/live_probe_inputs.json
  stages/stage_calls.jsonl
  predictions/raw_predictions.jsonl
  predictions/paired_predictions.jsonl
  prediction_seal.json
  reports/live_one_instance_probe_report.json
```

`audit/live_one_instance_events.jsonl` records, in order: environment check,
gateway-manifest validation, input hashes, detector construction, paired inference
statuses, and archive sealing. It records hashes and IDs rather than emitting a
credential. The input and prediction artifacts can contain RAGTruth text, so handle
the archive according to the dataset’s data-handling rules.

## Structured-output truncation policy

HalluGraph rejects every completion whose `finish_reason` is not `stop`; truncated
JSON is never parsed, cached, or treated as a graph. The live runtime starts KGGen
with `max_tokens=8192`. Only when the strict completion guard reports exactly
`finish_reason='length'`, it retries that graph extraction once with `max_tokens=12288`.
Other schema/parse failures still fail immediately, and transient network failures keep
their existing bounded transport retry policy. Each length retry writes the attempt,
current/next token budget, cache key, and outcome to
`audit/hallugraph_extraction_attempts.jsonl`; the runtime config and token policy are
also recorded in the paired archive audit trail.

## What local tests prove

`tests/experiments/test_live_one_instance_probe.py` does not call a live model. It
injects a dummy environment secret and mock detector factory to prove the production
entrypoint’s boundaries: required environment-variable names, manifest validation,
one-record no-gold materialization, audit ordering, sealing, and secret redaction.

The first actual DataSphere execution is the live verification of gateway access,
KGGen structured output, local HHEM loading, and both detector results. Do not expand
the scope beyond one response until that archive has been inspected.
