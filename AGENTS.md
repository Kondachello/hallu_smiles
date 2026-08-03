# AGENTS.md

This is the operating guide for agents working on this repository. Read it before changing code, building an image, or submitting a DataSphere Job.

## Project state and active path

This repository reproduces the response-level **HalluGraph** hallucination detector on RAGTruth QA, with KGGen as the knowledge-graph extractor. The active production path is not the former local vLLM/GPU path. It is:

```text
RAGTruth (context, query, answer)
  → KGGen graphs G_c, G_q, G_a
  → local CPU S-BERT matching/retrieval
  → authenticated Cloud Run gateway
  → Vertex AI / Gemini 2.5 Flash
  → DataSphere CPU Job artifacts and persistent project-disk caches
```

The active experimental modes are:

- `strict` — historical HalluGraph-style graph alignment.
- `support` — text-supported graph relations.
- `support-critical` — the current experimental mode: graph evidence plus strict atomic-claim checking. It must not alter historical `strict` or `support` formulas or cache protocols.

The 3-QA gateway probe, the 100-QA support-critical pilot, and the fixed 750-QA scale-up have completed.  There is **no active DataSphere Job** at the time this guide was updated. Their caches and archives are historical inputs, not disposable scratch data. Do not resubmit the completed 750-QA experiment merely to reproduce its result.

Do not create a new branch unless the user explicitly asks. Do not delete prior run archives, rendered historical Jobs, or existing result folders as a cleanup step.

## `new-metrics` completed history and verified baseline

The `new-metrics` branch contains the reliability work and the final scale-up of the
`support-critical` experiment. This history is important context for follow-up work;
do not revert its safeguards or treat a terminal console status alone as a scientific
result.

- The fixed RAGTruth manifest has 750 QA rows: 600 train and 150 held-out test,
  with five folds used only inside train and source `12448` explicitly quarantined.
  Thus 749 sources were analysed; all completed without extraction failures. Three
  held-out responses had an empty answer graph, so all headline methods share
  147 scorable test responses (75 positive, 72 negative).
- The archive-verified result is documented in
  `docs/support-critical-750qa-results.tex`. On that fixed held-out denominator,
  support-critical obtained ROC-AUC `0.849` (bootstrap 95% interval
  `[0.784, 0.909]`) and F1 `0.798` (`[0.726, 0.862]`); strict obtained
  `0.755` / `0.721` and support `0.730` / `0.695`. These are promising
  single-manifest estimates, not a paired significance claim or independent
  replication.
- Train-only tuning selected `alpha=1.0`, `beta=0.5`, `k=3`, and `lambda=0.0`
  for support-critical. Consequently this particular final score is
  `0.5 * (1 - EG) + 0.5 * H_top3`: relation preservation was audited but has
  no direct numerical weight. Do not claim that this run demonstrates a direct
  benefit from relation scoring.
- R11 (`bt1et79n009goi7besij`) finished the live scientific computation, but its
  terminal acceptance check incorrectly compared the diagnostic
  `verifier_cache_hit` field in live versus replay records. The field is expected
  to change from false when filling a cache to true when replaying it; scientific
  scores, metrics, tuning, and cache inventories did not change. Commit `dc512b7`
  fixes the comparator to ignore **only** that diagnostic field.
- R12 (`bt1fud5f0v4sbr1ru4jo`) is the successful recovery. It ran with
  `--force-cache-only`, made zero live inference calls, and verified complete
  extraction, persistent-cache reuse, cache-only acceptance, and train/test
  isolation. Its downloaded archive is under
  `outputs/datasphere-vertex-750qa-20260729-r12-cache-only-success/`.
- The preceding reliability changes are intentional: content-addressed persistent
  caches, atomic writes and read-through, adaptive structured-output handling and
  segmentation, bounded transient retry with `Retry-After`, conservative serial
  request pacing, and redacted outer/inner progress. They address nested response
  limits and HTTP 429 quota pressure without turning extraction or review failures
  into silent `unknown` values. Preserve them when extending the pipeline.

The R12 acceptance comparison requires zero network inference, unchanged cache
inventories, and identical scientific outputs. For `scored.jsonl`, comparison may
ignore only `verifier_cache_hit`; it must continue to reject every other changed
field. Never weaken this rule further to make a run pass.

## Next queued scientific task: DocRED KG-extraction evaluation

This task is implemented on branch `kggen-docred`. Its first submission was
validated locally and reached DataSphere on 2026-08-01, but DataSphere rejected
creation before allocating a Job ID with `community is blocked by billing`.
Thus no DocRED live inference, cache mutation, or Gemini charge occurred, and there
is no active DocRED Job to monitor. This is a separate evaluation of the KGGen
extraction component, not another hallucination-detection run and not a reason to
modify the historical RAGTruth metrics.

For each DocRED document, extract a graph from the document text with the repository's
KG extraction path, align predicted triples to DocRED's document-level entity IDs and
relation inventory, then measure:

- gold-triple recall: the share of annotated DocRED triples recovered by extraction;
- gold-supported predicted-triple precision: the share of extracted triples that
  match an annotation; and
- their micro F1, plus counts, document-level uncertainty, extraction coverage, and
  diagnostic entity-pair/relation-alignment breakdowns.

DocRED labels are incomplete with respect to all facts that may be true in a document.
Therefore the second quantity must be named **gold-supported precision**, not an
absolute factual precision. A prediction absent from the annotation is not evidence
that it is false. Matching must preserve directed head/tail entity IDs, use documented
mention/title alias resolution, and use a relation-alignment policy frozen before the
held-out evaluation. Never inspect held-out gold triples to choose prompts, aliases,
thresholds, or a predicate mapping.

Use an annotated development protocol: develop/freeze extraction configuration and
the relation-alignment policy only on DocRED train data, then evaluate once on the
labeled public development split unless a separately labeled official test split is
available. Call the latter a held-out development result, not an unseen benchmark
test. Record the exact dataset release/checksum, document manifest, prompt/model and
runtime fingerprints, alignment policy, cache namespace, and failure/coverage policy.
Do not silently drop failed or unalignable documents; report them and make their
treatment in every denominator explicit.

The pre-registered first live run is fixed and budgeted:

- `train_annotated`: 50 deterministic calibration documents, seed `42`; the first
  10 are a live smoke stage and remain part of those 50.
- labelled public `dev`: 200 deterministic held-out documents; it is called a
  held-out development result, never a blind test benchmark.
- The official blind test and `train_distant` are not used. The source ID `12448`
  rule belongs only to RAGTruth and has no meaning for DocRED.
- The relation-description S-BERT threshold grid `{0.65, 0.75, 0.85}` is selected
  only from the 50 train documents, then frozen before the 200-document dev pass.
- The maximum estimated live Gemini budget is EUR 10.5. The guard records its
  pinned price snapshot, reserves each cold document, and after smoke verifies it
  can reserve all still-cold documents. A `budget_exhausted` checkpoint is an
  explicit non-result, not permission to extend the manifest or make another paid
  attempt.

The implemented path is:

```text
scripts/fetch_docred_data.py
  → persistent project-disk DocRED namespace (pinned revision/checksums)
  → scripts/run_docred_kg_eval.py (live 10/50/200, cache checkpoints, redacted progress)
  → train-only relation alignment freeze
  → held-out metrics
  → zero-network cache-only replay
  → scripts/write_docred_kg_results_tex.py (only from downloaded terminal archive)
```

Use `scripts/submit_datasphere_vertex_docred_kg_eval.sh`, never a hand-edited
rendered YAML. It validates the existing bounded gateway artifact, pushed source
commit, immutable CPU image and rendered Job. The CPU runner has `concurrency=1`,
four-second pacing, base output cap `4096`, adaptive extraction ceiling `8192`, and
the existing 30-minute continuous-429 deadline. It emits atomic redacted
`progress.json`/`progress.jsonl` and `[docred-progress]` lines. Persistent caches
live under a DocRED-specific project-disk root keyed by the gateway manifest; they
are incompatible with RAGTruth caches and must never be deleted. Archives deliberately
exclude cache keys, raw graphs, prompts, completions, and raw usage rows while retaining
aggregate cache inventories and scientific artifacts.

After a terminal completion, download into a new `outputs/docred-...` directory,
validate archive metadata, full 200-document coverage, no live replay calls, and
unchanged cache inventory, then write `docs/docred-kg-extraction-results.tex` with
`scripts/write_docred_kg_results_tex.py` and build it with XeLaTeX. If the Job errors,
first download and diagnose its archive. Only one compatible serial cache-resume is
allowed automatically; do not submit overlapping Jobs or a second retry without user
direction. A 15-minute monitor must treat redacted inner/retry heartbeats as work and
must not cancel merely because the outer document counter is momentarily unchanged.

The detailed, copyable hand-off for this task is
`docs/task-prompts/docred-kg-extraction-evaluation.md`. It carries the repository,
cache, testing, DataSphere-authentication, and reporting constraints that a new chat
must follow.

## Scientific object and fixed notation

For each RAGTruth QA triple `(context C, query Q, answer A)`:

```text
G_c = knowledge graph extracted from C
G_q = knowledge graph extracted from Q
G_a = knowledge graph extracted from A
G_ref = G_c ∪ G_q
```

Let `V_a` be answer entities and `E_a` answer edges. A higher `H` always means a higher hallucination risk.

### Shared entity score

```text
EG = (# answer entities that match an entity in G_ref) / |V_a|
```

An entity matches through normalized equality, allowed token-boundary substring matching, or local S-BERT cosine similarity at `τ_e`.

### Historical strict score

An answer edge aligns only if its subject and object match in the correct direction and its relation matches at `τ_r`.

```text
RP_strict = (# aligned answer edges) / |E_a|
CFI_strict = α · EG + (1 − α) · RP_strict
H_strict = 1 − CFI_strict
```

### Historical support score

For graph edges, a verifier checks source text rather than relying solely on graph alignment.

```text
RP_support = (# text-entailed answer edges) / |E_a|
CFI_support = α · EG + (1 − α) · RP_support
H_support = 1 − CFI_support
```

### Support-critical score

`support-critical` is a separate, versioned protocol. It uses four textual verdicts:

```text
entailed       → risk 0
unknown        → risk λ, where λ is tuned from {0.0, 0.25}
unsupported    → risk 1
contradicted   → risk 1
```

For graph edges:

```text
RP_critical = 1 − mean(edge risk)
H_graph = 1 − [α · EG + (1 − α) · RP_critical]
```

For atomic claims extracted from the whole answer and claims found by the full-context reviewer:

```text
H_topk = mean risk of the k worst claims
H_critical = (1 − β) · H_graph + β · H_topk
```

`α`, `β`, `k`, `λ`, matching thresholds, and classification threshold `θ` are tuned **only on train data**. Current grids are:

```text
α ∈ {0.0, 0.1, …, 1.0}
β ∈ {0.25, 0.5, 0.75}
k ∈ {1, 2, 3}
λ ∈ {0.0, 0.25}
```

`α` is the entity-versus-relation weight **inside the graph score**; it is not an LLM parameter. `β` is the graph-versus-worst-claims weight. The selected values are data-dependent and must never be copied from a previous run into a new experiment.

Important implementation fact: in the successful 100-QA run, train-only tuning selected `α=1.0`, `β=0.75`, `k=3`, `λ=0.0`. Thus that particular final score was:

```text
H_critical = 0.25 · (1 − EG) + 0.75 · H_top3
```

The knowledge graphs were still extracted, entity-grounded, relation-audited, and text-verified, but `RP_critical` had zero direct numerical weight in that selected configuration. Do not describe that result as proof that relation scoring improved detection. It is evidence for the claim-verification branch plus a small graph-entity contribution.

Also distinguish the current implementation from an intended future extension: graph edges are audited and separately verified, but `CriticalClaimPipeline.assess(answer, context, query)` currently builds `H_topk` from atomic claims and full-context review claims; it does not inject every strict-unmatched graph edge into that top-k claim list. Do not silently claim that this merge already happens.

## Splits and scientific invariants

- RAGTruth label `y=1` means the response has at least one annotated hallucination span.
- `implicit_true` is configurable; `due_to_null` always counts.
- A manifest is the experimental unit. It records the exact chosen QA rows, split, seed, and hash.
- A fixed manifest with test fraction `0.2` has 80% train and 20% held-out test. Five-fold CV is performed only within train.
- Tune `α`, `β`, `k`, `λ`, `τ_e`, `τ_r`, and `θ` on train only. Score test once with frozen parameters.
- Do not select examples, change thresholds, inspect labels to refine prompts, or choose a model using the held-out test set.
- `EG` and relation scores are expensive but independent of `α`; calculate them once and derive weights later.
- `|V_a|=0` means `unscorable`. The claim layer may produce a diagnostic score, but headline paired strict/support/support-critical metrics exclude it so denominators remain comparable.
- `|E_a|=0` means relation preservation is undefined; reduce graph fidelity to `EG`, never fabricate a relation score of zero.
- `|V_ref|=0` is flagged/excluded according to the established policy.

## Source of truth for model and backend

`config.yaml` is authoritative:

```yaml
llm:
  model: "openai/gemini-2.5-flash"
  api_key_env: "HALLU_GATEWAY_API_KEY"
  structured_output_backend: "vertex"
```

Do not hardcode a second model identifier elsewhere. Job-local configuration derives `api_base`, Vertex model revision, and runtime fingerprint only after it validates the authenticated gateway manifest.

The active public gateway is:

```text
https://hallu-vertex-gateway-453887629111.europe-west4.run.app
```

The gateway offers authenticated `/healthz`, `/v1/hallu/manifest`, and OpenAI-compatible `/v1/chat/completions`. It maps requests to Vertex AI Gemini through its Cloud Run service identity. DataSphere never needs a Google private key or `GOOGLE_APPLICATION_CREDENTIALS`.

`HALLU_GATEWAY_API_KEY` is a DataSphere project secret injected as an environment variable. It must never appear in source code, generated YAML, a command argument, an archive, a diagnostic, or logs. Do not log prompts, completions, or Authorization headers.

Before every live Job, fetch and validate the gateway manifest. Its hash, Cloud Run revision, location, logical model, and protocol version belong in runtime identity and cache fingerprints. A gateway/model/revision incompatibility must invalidate affected entries rather than silently reuse them.

## Current execution path

Use the CPU Vertex flow:

```text
.github/workflows/datasphere-vertex-cpu-runtime-image.yml
  → immutable GHCR CPU image digest
  → scripts/submit_datasphere_vertex_qa_pilot.sh
  → scripts/render_datasphere_vertex_qa_pilot_job.py
  → datasphere/jobs/vertex-cpu-qa-pilot.template.yaml
  → scripts/run_datasphere_vertex_cpu_qa_pilot.sh inside the Job
```

GitHub Actions only builds and publishes a reproducible Docker image. It does **not** create or start a DataSphere Job. The submit script resolves/validates the immutable digest, renders and validates Job YAML, then the DataSphere CLI creates the Job.

### DataSphere CLI authentication (mandatory)

R11/R12 were submitted through `datasphere --profile default`, using the existing
**Yandex Identity Hub subject-id profile** in
`~/.config/yandex-cloud/credentials/default`. That profile stores a renewable
federated refresh credential locally. The `datasphere` CLI obtains a short-lived IAM
token from it internally; do not invent or require a separate `YC_TOKEN`/
`YC_OAUTH_TOKEN` environment secret.

Never run `yc init`: it can reinitialise the profile, ask for a cloud/folder or an
unrelated account type, and replace the subject-specific flow that owns this project.
Never print the profile file, its refresh credential, IAM token, subject identifier,
or a browser callback.

Before any agent/monitor command, mint a short-lived IAM token through the stored
credential **without opening a browser**, then let the DataSphere CLI read that token
from its supported process-local environment variable. Run this exact preflight from
the repository root:

```bash
source .venv/bin/activate
PATH="$PWD/.venv/bin:$PWD/.tools/yc:/Users/maslovartemij/yandex-cloud:$PATH"
export YC_IAM_TOKEN="$(GRPC_DNS_RESOLVER=ares \
  "$PWD/.tools/yc/yc" --profile default --no-browser --no-user-output \
  iam create-token)"
GRPC_DNS_RESOLVER=ares \
  datasphere project get --id bt1i64odluitglbaj5st
unset YC_IAM_TOKEN
```

`YC_IAM_TOKEN` is a short-lived IAM credential, never an OAuth token. It is created
inside the running shell, is not put in the command line, must never be printed or
written to a file, and must be unset after the command. The `datasphere` package
checks this variable before its normal `yc` fallback. That distinction matters:
`datasphere --profile default` internally invokes `yc iam create-token` again
**without** `--no-browser`, which can reopen an unwanted subject-id page. For this
repository, use the in-memory `YC_IAM_TOKEN` pattern above for `project get`, Job
status/download, and Job submission; do not add `--profile` to the `datasphere`
command after the token has been exported.

The token-mint command is not `yc init` and does not alter the profile.
`--no-browser` is essential: it makes an expired or unusable federated session fail
closed rather than opening a login page. The normal working path therefore needs no
browser and no token pasted into a command or chat.

If this no-browser preflight fails, do not fall back to a personal Yandex ID, do not
guess federation parameters, and do not submit a Job. A user must renew the existing
Identity Hub session by the organisation's approved sign-in method, or the
organisation must provide a service account with `datasphere.community-projects.developer`.
Agents must not create accounts, keys, profiles, or an alternate OAuth flow.

Interpret authentication and submission errors separately. A successful no-browser
mint and `project get` prove that the local refresh-to-IAM path works. If a subsequent
`project job execute` returns a service-side
`PERMISSION_DENIED` such as `community is blocked by billing`, do **not** run
`yc init`, do not overwrite the profile, and do not repeatedly submit duplicate
jobs: this is not an expired IAM token. Preserve the rendered Job and immutable
image, report that no Job ID was allocated, and wait for the DataSphere community's
billing state to be restored before the single intended submission is retried.

Relevant commands/scripts:

```bash
pytest -q

# Inspect available parameters; QA count is deliberately a parameter.
bash scripts/submit_datasphere_vertex_qa_pilot.sh --help

# The submitter takes --qa-sample-size N, --qa-test-fraction 0.2,
# --cv-folds 5, --concurrency 1, gateway URL, gate artifact, and image data.
# Do not hand-submit an old file under datasphere/jobs/rendered/.

datasphere project job get --id JOB_ID --format json
datasphere project job download-files --id JOB_ID \
  --with-logs --with-diagnostics --output-dir outputs/RUN_NAME
```

Rendered files under `datasphere/jobs/rendered/` are immutable historical records or generated inspection artifacts. Never edit one and submit it as if it were the source template.

Old vLLM/GPU/XGrammar and Llama staging jobs are legacy artifacts. Do not revive them for the active Vertex CPU pipeline unless the user explicitly requests that separate work.

## DataSphere Job sequence

The current CPU QA Job intentionally does the following:

1. Clones the exact pushed source commit and uses a pinned image digest.
2. Attaches DataSphere project disk, which holds persistent cache/checkpoint roots.
3. Validates gateway authentication and manifest, then writes a job-local config without secrets.
4. Uses an offline resilience preflight for offsets, segmentation, and cache-key invariants.
5. Runs a small live/cache-only support-critical gateway schema probe.
6. Generates or loads the deterministic QA manifest.
7. Reproduces historical strict/support baselines cache-only when their cache lineage is available.
8. Runs support-critical, records usage by component, writes audits and metrics.
9. Replays strict/support/support-critical cache-only and rejects any live HTTP call or changed cache hash.
10. Archives manifest, runtime configuration, cache hashes, diagnostics, metrics, usage, and logs.

For a larger manifest, do not require all graph cache keys to exist before the first run: new QA rows legitimately need live KGGen extraction. Reuse old cache entries through read-through, fill only missing entries, then require fully cache-only replay after completion.

## Cache contract and restart behavior

There are separate, content-addressed cache families:

```text
kg/                 G_c, G_q, G_a from KGGen
verdicts/           historical support relation verdicts
critical_claims/    atomic claim extraction
critical_coverage/  full-context candidate review
critical_verdicts/  strict four-way claim verdicts and evidence
```

Keys include relevant text, model/parameters, prompt or schema version, structured-output contract, gateway/model revision, and runtime fingerprint. Cache reads are primary-first and then read-through historical roots. A corrupt primary entry must not hide a valid compatible read-through entry.

Writes are atomic (`os.replace`) and use unique temporary names. Do not replace this with a shared temporary filename: duplicated RAGTruth content can be processed concurrently.

Consequences:

- A repeated source text can reuse a cache entry even if its row ID changes.
- A changed prompt, schema, model revision, gateway manifest, or extraction protocol must not reuse an incompatible entry.
- If a Job fails after caching partial work, a resubmission with the same manifest/config must resume from completed entries.
- Never delete project-disk caches to “start fresh” unless the user explicitly asks and exact cache roots have been identified.
- `--cache-only` means no network inference is allowed. A cache miss is a fast, useful failure, not permission to make a live request.

The cache-only acceptance condition is strict: zero live API calls, byte-identical
metrics/CSV outputs, and unchanged SHA-256 cache inventories. The one approved
exception is that a replay comparison of `scored.jsonl` strips the diagnostic
`verifier_cache_hit` boolean before comparison; that boolean necessarily differs
between initial cache population and cache reads. No scientific score, verdict,
parameter, metric, or other record field may be ignored.

## Reliability requirements for live inference

The network and Vertex quota are expected to be imperfect. Treat `429`, temporary `5xx`, connection resets, temporary DNS failures, and read/connect timeouts as transient. Preserve completed cache entries and retry with bounded exponential backoff, jitter, and `Retry-After` where supplied. Do not turn a transient `429` into `failed_extractions.jsonl` merely because a small local retry counter elapsed.

Conversely, do not retry forever on a deterministic configuration error such as an invalid API key, unknown model, incompatible manifest, or malformed request. Fail early with a redacted diagnostic.

Structured JSON handling is part of the protocol:

- Use native `response_format` / JSON Schema through the gateway, not vLLM/XGrammar fields.
- Validate the returned JSON locally.
- Detect truncation separately from invalid schema.
- Use the existing adaptive token-budget retry and response segmentation paths.
- Validate every claim offset against the exact response text.
- Never silently convert a malformed LLM response into `entailed`.
- A repeatedly schema-invalid scalar verifier may return explicitly marked `unknown` through the existing conservative fallback; extraction/review must not silently drop a factual span.

Concurrency for the CPU Vertex QA Job is deliberately conservative (`1` by default) to avoid quota bursts. Do not increase it simply to make a large run faster without a measured quota plan.

Do not impose an arbitrary short wall-clock watchdog that discards a long-running Job. If a platform limit is real, rely on the persisted content-addressed checkpoints and provide a safe resume submission; never erase caches after a timeout.

## Important modules

- `run.py` — the single pipeline entrypoint and train/test orchestration.
- `src/config.py` — YAML config, environment-key resolution, strict attribute access.
- `src/data.py` — RAGTruth loading, serializer, deterministic QA manifest and split handling.
- `src/extract.py` — KGGen extraction, cache, chunking, retries, usage logging, `Graph`.
- `src/matching.py` — normalized entity matching, `RefGraph`, S-BERT embedding/matching.
- `src/metrics.py` — `ScoreResult`, strict/support scores, support-critical graph and claim aggregation.
- `src/critical.py` — atomic claim extraction, full-context review, evidence retrieval, four-way verifier, its caches and fallbacks.
- `src/tune.py` — pure train-only CV and threshold helpers. It must not access test labels.
- `src/evaluate.py` — held-out metrics, bootstrap diagnostics, plots.
- `src/audit.py` — per-response evidence trail.
- `scripts/make_datasphere_vertex_config.py` — validates gateway manifest and derives the secret-free runtime config/cache namespaces.
- `scripts/run_datasphere_vertex_cpu_qa_pilot.sh` — in-Job QA orchestration and cache-only acceptance checks.
- `scripts/preflight_support_critical_resilience.py` and `scripts/check_support_critical_gateway_probe.py` — preflight gates; keep them before a long support-critical loop.

## Tests and acceptance when changing code

Keep unit tests offline. Use fake extractors, fake verifiers, and deterministic embedders; no Vertex, DataSphere, torch, or gateway credentials should be necessary for `pytest`.

At minimum, changes touching the active path must preserve or add coverage for:

- train/test isolation and train-only tuning;
- cache key versioning and primary/read-through fallback;
- atomic cache writes and cache-only zero-network replay;
- strict/support byte compatibility;
- structured-output schema, offsets, segmentation, and fallback behavior;
- four support-critical verdicts and risk mapping;
- top-k aggregation, one-risky-claim behavior, and empty-graph handling;
- rendered Job validation and secret-free runtime config.

Before a live submission, run `pytest -q`, validate the rendered Job, verify the gateway probe, and confirm the source commit is pushed and the image digest is immutable. After a Job completes, download its archive before drawing conclusions; console output alone is not evidence.
