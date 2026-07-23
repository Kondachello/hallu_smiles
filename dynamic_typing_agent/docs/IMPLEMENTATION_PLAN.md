# Implementation plan for the next agent

## Objective and stopping boundary

Implement the standalone package described by this directory, verify it with deterministic
fakes, then integrate it through a narrow outer adapter. Do not modify HalluGraph matching
or scoring in this workstream. Do not run a live backend or DataSphere job without a new
explicit user request.

## Phase 0 — claim, versions and dependency lock

1. Read the project wiki and all active claims.
2. Create a W2 claim covering only `dynamic_typing_agent/` runtime files and its own wiki log.
3. Inspect current official LangGraph/LangChain/Pydantic/LiteLLM compatibility and create a
   locked dependency artifact for Python 3.11/3.12. Preserve major-version bounds in
   `pyproject.toml`; record exact resolved versions in the run manifest.
4. Add import smoke tests for the locked environments.
5. Bump the package pre-release version only when executable code starts.

Exit: clean isolated checkout, non-overlapping claim and reproducible dependency resolution.

## Phase 1 — immutable domain models

Create Pydantic v2 models under `src/hallugraph_dynamic_typing/domain/`:

- IDs and provenance: `EvidenceSpan`, `ModelFingerprint`, `PromptProvenance`, `GraphRef`;
- graph input: `EntityNode`, `RelationEdge`, `SourceInput`, `AnswerInput`;
- type system: `TypeDefinition`, `TypeRelation`, `ContextualRole`, `TypeAssignment`;
- decisions: discriminated unions for every `DecisionAction`;
- NLI: `NliRequest`, `NliResult`, `NliEvidenceLevel`;
- registry: `RegistryDraft`, `FrozenTypeRegistry`, `RegistryHistoryEvent`;
- outputs: `SourceRegistryRun`, `AnswerAnnotationRun`, `FailureRecord`.

Rules:

- `extra="forbid"`, strict enums and bounded strings/lists;
- canonical serializers exclude `None` only where specified by the contract;
- stable IDs derive from canonical semantic payloads, never object order or wall time;
- frozen models for persisted values;
- answer input cannot be converted to source input;
- no model has a gold-related field.

Add round-trip, stable-ID and forbidden-extra tests before graph code.

Exit: canonical contract fixtures validate and hash identically across repeated runs.

## Phase 2 — prompt registry and structured model transport

Implement `PromptRegistry`:

- load exactly one prompt version directory;
- reject path traversal, missing/extra variables and schema mismatch;
- render Jinja with `StrictUndefined` and autoescape disabled only because output is plain text;
- hash raw files and the canonical manifest;
- return rendered messages, output schema and provenance.

Implement `LiteLLMStructuredModelAdapter` behind `StructuredModelPort`:

- reuse logical model, gateway `api_base`, environment-secret name, timeout and retry policy;
- pass native `response_format` JSON Schema where supported;
- independently parse and validate the returned document;
- accept only one choice with clean stop reason;
- own all retries; inner clients use zero retries;
- classify retryable transport errors separately from terminal protocol errors;
- emit usage without prompt/response text in regular logs.

Implement `FakeStructuredModel` as an ordered operation-to-response script with call
records. Never make tests monkeypatch LiteLLM globally when injection suffices.

Exit: every prompt renders, valid fake output passes, malformed/truncated output fails.

## Phase 3 — evidence spans and deterministic policies

Implement exact source segmentation:

- preserve original text and character offsets;
- keep context/query provenance separate;
- stable sentence/paragraph span IDs;
- exact quote round-trip assertion;
- deterministic chunking with overlap metadata for long source input.

Implement deterministic utilities:

- graph normalization without changing graph content;
- entity profiles from graph participation and spans;
- bounded candidate retrieval;
- registry DAG and alias-component validation;
- deterministic routing predicates for NLI;
- canonical immutable filesystem/SQLite caches;
- atomic artifact writes and checksums.

Candidate similarity may shortlist but never confirm assignment, identity, alias or hierarchy.

Exit: pure tests cover Unicode, duplicate mentions, missing textual mention, cycles and order permutations.

## Phase 4 — source LangGraph

Create small node modules, one public callable per node. Node bodies receive immutable state
and runtime dependencies; they return partial state updates.

Implement in order:

1. `validate_source_input`;
2. `resolve_source_cache`;
3. `segment_source` and conditional chunk fan-out;
4. `schema_overview`;
5. `schema_reconcile` when multiple drafts exist;
6. `build_entity_profiles`;
7. `retrieve_type_candidates`;
8. batched `entity_type_decision`;
9. `route_source_nli` plus reusable NLI subgraph;
10. `registry_consistency_review` on shortlisted close pairs only;
11. `cluster_split_review` only for heterogeneous clusters;
12. deterministic `validate_registry`;
13. bounded `registry_repair` loop;
14. `freeze_registry` and immutable cache/artifact emission.

Use conditional edges for cache hit, chunking, NLI routing and repair. Use `Send` only for
bounded independent fan-out. Sort fan-in by stable IDs. Compile with an injected checkpointer;
provide in-memory checkpointer for tests and SQLite for local runs.

Exit: fake source runs cover every action, three NLI verdicts, cache hit and resume.

## Phase 5 — reusable NLI subgraph

Implement one subgraph shared by source and answer graphs:

1. validate one short hypothesis and non-empty exact source evidence;
2. compute evidence level before model invocation;
3. resolve immutable NLI cache;
4. render `nli_verification` prompt;
5. invoke injected NLI/model port;
6. validate `entailed|contradicted|neutral`;
7. persist premise/hypothesis hashes, provenance and calibrated confidence;
8. return verdict without applying caller-specific policy.

The caller maps verdict to assignment/hierarchy/answer/edge behavior. This prevents NLI
transport code from silently changing scientific policy.

Add safeguards:

- definition-only evidence cannot confirm alias merge;
- neutral never becomes contradiction;
- same generative model as proposer is recorded as correlated verification;
- external knowledge statements in explanations do not become evidence.

Exit: all route-specific policy tables have unit tests.

## Phase 6 — answer LangGraph

Implement:

1. `validate_answer_input`: require frozen registry and matching source identity;
2. `build_answer_profiles` and explicit type-phrase extraction;
3. `answer_typing`: existing registry IDs only or unknown;
4. `route_answer_nli`: explicit specializations/conflicts and weak assignments only;
5. one-to-one entity alignment with type as secondary evidence;
6. bounded `edge_candidate_resolution`;
7. `route_edge_nli` for uncertain relation/direction/role hypotheses;
8. `emit_annotation_bundle` with coverage and audit fields, not HalluGraph score.

Any proposed answer-only type is diagnostic and remains outside the frozen registry.

Exit: tests prove answer cannot add/reparent/rename types or mutate source assignments.

## Phase 7 — standalone facade and CLI

Implement a facade with `build_source_registry` and `annotate_answer`, plus the CLI commands
documented in `INTEGRATION.md`.

Configuration precedence:

1. explicit Python/CLI arguments;
2. selected config file;
3. safe non-secret defaults.

Secrets are read only by the live transport adapter from the configured environment-variable
name. CLI output contains run/artifact paths and status, never secret values. Add `--backend
fake|cache_only|live`; live requires explicit selection and complete validated configuration.

Exit: package can be installed in a fresh environment and run from outside the parent repo.

## Phase 8 — framework adapter and contract v2

Under a separate W3 claim:

1. define the outer typing archive schema v2 for hierarchy, multi-assignment, roles, evidence,
   decision history, NLI and failure records;
2. retain a reader for v1 stub artifacts but never pretend v1 contains v2 evidence;
3. implement the adapter translating `SharedGraphBundle` without mutation;
4. inject the provider into `build_three_way_shared_kggen_detectors`;
5. preserve the all-unknown score-equality test;
6. add model-backed fake tests that still leave typed scoring unchanged;
7. prove cache-only replay makes zero KGGen and typing-model calls.

Exit: richer annotations are archived and sealed while B0/scoring remains unchanged.

## Phase 9 — verification and scientific readiness

Run:

- package unit/graph/integration tests;
- outer three-way framework tests;
- prompt/schema/example validator;
- type checker and linter selected for the package;
- wiki validator.

Produce a deterministic fake run over all twenty examples. Report proposed registries,
unknowns, NLI routes and call counts, but do not label it a scientific result.

Only then request user approval for one live synthetic probe. B5 score integration is a new
W4 decision after agent-quality inspection.

## Required review checklist

- Source graph has no answer field at runtime or checkpoint level.
- Answer graph receives an already frozen registry.
- Every model node has prompt/schema provenance.
- Every accepted type/hierarchy decision has source evidence or an explicit weaker level.
- NLI is routed, three-way and cached; neutral is not contradiction.
- Unknown and failed are disjoint in contracts, metrics and logs.
- Fan-out/fan-in is deterministic.
- Registry repair is bounded.
- Cache-only cannot become live.
- Framework integration cannot mutate shared graphs.
- No gold label appears in agent API, state, prompt or fixture input.

