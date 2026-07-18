# DataSphere API pilot runbook

This workflow runs HalluGraph on a DataSphere CPU VM and calls Alibaba Model
Studio through its OpenAI-compatible endpoint. It does not start a local model,
request a GPU, use vLLM/CUDA, or write to the shared project disk.

## One-time project setup

1. In Alibaba Model Studio, select the Singapore region, enable **Free Quota
   Only**, and create an API key.
2. Open DataSphere project `bt1i64odluitglbaj5st`. In **Project resources**
   (`Ресурсы проекта`) select **Secret** (`Секрет`), click **Create**
   (`Создать`), set **Name** to `DASHSCOPE_API_KEY`, paste the Alibaba key into
   **Value**, and click **Create**. This is the workflow documented in
   [Yandex Cloud: Working with secrets](https://yandex.cloud/en/docs/datasphere/operations/data/secrets).
3. Do not put the key in Git, a YAML file, a command line, or a downloaded Job
   artifact. The Job checks only that the environment variable is non-empty.

The model and endpoint are configured only under `llm` in `config.yaml`. The Job
templates deliberately contain neither a model identifier nor an endpoint.

## Publish and run the three-QA probe

The submit wrapper accepts only an exact, already-published 40-character commit
SHA. From the repository root:

```bash
COMMIT="$(git rev-parse codex/new-metrics-api-clean)"
RUN_ID="api-probe-$(git rev-parse --short "$COMMIT")-$(date -u +%Y%m%d-%H%M)"

GRPC_DNS_RESOLVER=native bash scripts/submit_datasphere_job.sh \
  --kind api-probe-c1 \
  --project-id bt1i64odluitglbaj5st \
  --branch codex/new-metrics-api-clean \
  --commit "$COMMIT" \
  --run-id "$RUN_ID"
```

The wrapper renders and validates the YAML, checks that the commit belongs to
the remote branch, prevents concurrent HalluGraph API Jobs, submits at most one
Job for the unique run ID, monitors it to a terminal state, downloads the output
archive, and validates the archive before returning success. Read-only CLI
operations retry transient DNS/gRPC failures. An ambiguous submit is reconciled
by looking up the unique Job name before any retry.

The probe must establish all of the following:

- the pinned Python 3.11 API runtime and the RAGTruth shared directory are usable;
- three identical Swiss-chard requests return the exact root contract;
- synthetic KGGen extraction, official KGGen LLM clustering, and the support
  verifier complete;
- the deterministic 20-QA manifest is created, while only its first three rows
  are extracted and scored in strict and support modes;
- all three reference/answer pairs exist and `failed_extractions.jsonl` is empty;
- replaying extraction and verification from the caches performs zero API calls.

The downloaded tar contains the manifest, KG and verifier caches, contract
reports, strict/support scores, audits, provider-call metadata, usage totals,
runtime metadata, and stdout/stderr. Provider telemetry excludes API keys,
authorization headers, and full prompts.

## Full 20-QA pilot

`api-pilot-c1` is implemented but is intentionally not submitted by the probe
workflow. First review the probe report and token use. The pilot requires the
validated probe tar as a gate artifact, safely imports its manifest and caches,
continues the same 20-QA selection, and then runs the full train-only tuning and
single test evaluation for both strict and support metrics.

The pilot command is shown only after the probe is accepted. It follows the same
published-SHA and unique-run-ID rules and supplies the validated gate archive to
the submit wrapper. Never use an arbitrary or manually modified tar as the gate.

## Storage and quota

Both Jobs use `c1.4`. They read RAGTruth from
`$DS_PROJECT_HOME/hallu_smiles/shared/ragtruth` and write caches/results only in
their per-Job working directory. The existing shared Llama files are left
untouched and do not need to be removed. Alibaba pricing and limits can change;
check the current [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
and [rate-limit table](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
before starting the 20-QA pilot.
