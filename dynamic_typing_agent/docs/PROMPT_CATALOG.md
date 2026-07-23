# Prompt catalog

## Storage model

Prompts are package data under `prompts/v1/`. Each model operation has:

- `<prompt_id>.system.md` — stable role, invariants and decision policy;
- `<prompt_id>.user.j2` — Jinja template containing only operation inputs;
- `schemas/<prompt_id>.schema.json` — strict output schema;
- one `manifest.json` entry — version, files, required variables and allowed phase.

Prompt text is never embedded in graph-node Python code. The loader reads one immutable
manifest, validates paths and variables, computes SHA-256 for every file and then computes
a complete manifest hash. That hash is included in checkpoints, cache keys and artifacts.

New behavior requires a new prompt version directory (`v2`, not an in-place edit after a
scientific run). Prompt IDs remain stable when intent is unchanged; output schema changes
also require a contract-version change.

## Model nodes

| Graph node | Prompt ID | Purpose |
|---|---|---|
| `schema_overview` | `schema_overview` | Draft local types, distinctions and roles from source only |
| `schema_reconcile` | `schema_reconcile` | Reconcile chunk drafts without inventing evidence |
| `entity_type_decision` | `entity_type_decision` | Choose closed-set actions for entity profiles |
| `registry_consistency_review` | `registry_consistency_review` | Classify relations among nearby type definitions |
| `cluster_split_review` | `cluster_split_review` | Detect heterogeneous type clusters and propose an auditable split |
| `registry_repair` | `registry_repair` | Repair only listed deterministic invariant violations |
| `answer_typing` | `answer_typing` | Assign frozen-registry types or abstain; identify explicit assertions |
| `edge_candidate_resolution` | `edge_candidate_resolution` | Select among bounded edge matches or abstain |
| NLI subgraph | `nli_verification` | Return entailed/contradicted/neutral from supplied source evidence |

## Prompt injection policy

Context, query, answer, graph labels and type labels are untrusted data. User templates
wrap them in explicit data sections and state that instructions contained inside are not
instructions to the model. The model has no tools, retrieval or external knowledge input.
Structured output is enforced by the transport and independently validated.

## Language policy

Instructions are English for consistency. Source text may be any language. Type labels
should follow the dominant source language unless configuration requests a canonical
language. Evidence quotes remain verbatim. NLI hypotheses use the source language when
possible to avoid translation becoming hidden evidence.

