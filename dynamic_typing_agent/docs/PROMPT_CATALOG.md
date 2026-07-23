# Prompt catalog

## Storage and versioning

The active prompt set is immutable package data under `prompts/v2/`. Every model
operation has:

- `<prompt_id>.system.md` — stable role, terminology, invariants and decision policy;
- `<prompt_id>.user.j2` — a Jinja template containing only operation inputs;
- `schemas/<prompt_id>.schema.json` — a strict structured-output schema;
- one `manifest.json` entry — version, paths, required variables and phase.

Prompt text is not embedded in graph-node Python code. The loader validates all paths
and required variables, hashes every file and then hashes the complete manifest. The
manifest hash is included in cache keys and run artifacts. A behavior or output-contract
change creates a new version directory rather than mutating a prompt set used in an
earlier scientific run. `prompts/v1/` is retained only to reproduce old artifacts.

## Active model operations

| Stage | Prompt ID | Purpose |
|---|---|---|
| Source overview | `schema_overview` | Produce non-binding category hints from context, query and their graph |
| One source entity | `entity_type_decision` | Reuse a current type or propose one reusable semantic category |
| Registry review | `registry_consistency_review` | Propose bounded parent or merge changes after all entities are assigned |
| One answer entity | `answer_typing` | Select only IDs from the frozen source registry |
| NLI verification | `nli_verification` | Judge one explicit assignment, hierarchy or merge hypothesis |

The overview cannot create final registry entries. The source decision prompt is called
once per unique entity surface and receives that entity's graph neighbourhood, relevant
source spans and the registry built so far. The answer prompt is also entity-local and
cannot extend the frozen registry.

## Meaning of “type”

The system prompts define a type as a reusable semantic category used later to align
source and answer graphs in HalluGraph. A type is not:

- the entity's proper name or textual alias;
- an arbitrary object reached by a graph relation;
- a transient event, numeric value, date or relation phrase;
- a claim copied from the context without category semantics.

The prompts include positive and negative examples. They favor stable categories such
as `organization`, `person`, `city`, `financial institution` and `scientific method`,
while rejecting identity mappings such as `North Bank -> North Bank` and mechanical
edge mappings such as `X --is--> Y`, unless `Y` is independently justified as a reusable
category.

## NLI and finality

NLI means natural-language inference: checking whether a short hypothesis follows from
the supplied source evidence. The model proposes a type; it does not certify its own
proposal. The runtime constructs a canonical hypothesis such as
`North Bank is an organization.` and records the NLI result.

- `entailed` gives strong source evidence;
- `neutral` may remain a final but weak source classification because a category can be
  useful without being stated verbatim;
- `contradicted` triggers one broader retry and then a structural root fallback;
- answer-only semantic matches require `entailed`, except exact reuse of a source entity
  surface, which preserves that entity's already frozen source type;
- parent links require one entailed hypothesis;
- merges require entailed hypotheses in both directions.

“Final” is a workflow state, not a claim that the evidence is strong. Every serialized
source and answer vertex must have at least one final type ID. Evidence strength remains
explicit in `evidence_level` and in the NLI records.

## Untrusted-input policy

Context, query, answer text, graph labels and type labels are untrusted data. User
templates delimit them as data and instruct the model not to execute instructions found
inside. The model has no tools or external retrieval. The transport enforces structured
output, and the runtime independently validates the returned JSON schema and all IDs.

Instructions are written in English for consistency. Source text may be in any language.
Type labels follow the dominant source language unless a future configuration explicitly
requests canonicalization. Evidence remains verbatim.
