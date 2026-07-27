# Context/query diagnostic dataset for dynamic typing

## Purpose

This directory contains a small, manually designed diagnostic corpus for inspecting a
dynamic hierarchical entity-typing agent. It is not a quality benchmark and has no
hallucination labels. Its purpose is to expose recurring failure modes in type discovery,
hierarchy construction, entity assignment, contextual roles, abstention and evidence
provenance.

The corpus contains exactly twenty source-only inputs:

- ten Russian cases;
- ten English cases;
- six short, eight medium and six long contexts;
- multiple domains and mixed-domain documents;
- no answer text, prepared graph, registry or expected label in the agent input.

## Files

### `cases.no_gold.jsonl`

The only file that may be supplied to the type agent. Every line is one JSON object:

```json
{
  "case_id": "cqdiag-ru-001-bank-hierarchy",
  "language": "ru",
  "context": "...",
  "query": "..."
}
```

The agent must receive only `context` and `query`. `case_id` is an external correlation
key, and `language` is routing metadata. Neither field is a target label.

### `review_expectations.jsonl`

A physically separate human-review guide. It must never be opened by the type agent,
included in a model prompt or used to construct the source registry. It may be joined by
`case_id` only after a run has completed and its artifacts have been sealed.

The expectations are semantic review targets, not an exact gold registry. A valid dynamic
system may choose different labels while preserving the required distinctions, roles,
hierarchy and abstentions.

Each line contains:

```text
case_id               linked input ID;
domain                review slice;
length_class          short / medium / long;
primary_challenges    mechanisms deliberately stressed;
expected_distinctions concepts that should remain distinguishable;
expected_roles        contextual roles to inspect;
expected_abstentions  details the agent should not invent;
failure_signals       concrete symptoms of a bad registry.
```

## Coverage matrix

| Cases | Main diagnostic pressure |
|---|---|
| RU-001, EN-011 | safe hierarchy and granularity |
| RU-002, EN-012 | homonyms and identity before typing |
| RU-003, EN-016 | relation signatures and incompatible branches |
| RU-004, EN-015, EN-017 | permanent types versus scoped roles |
| RU-005, EN-014 | organization versus product/service and metonymy |
| RU-006, EN-013 | abstract work versus physical/digital copy |
| RU-007 | finance, insurance, property, quantities and explicit exclusions |
| RU-008 | gene/protein ambiguity, drug typing and mention-level evidence |
| RU-009, EN-019 | model versus instance, component levels and documents |
| RU-010 | one surface form across place, government, team and ship |
| EN-018 | observatory, instrument, event, data product and publication |
| EN-020 | query-only entity, unsupported fact and required abstention |

## Length policy

Length classes describe diagnostic complexity rather than a strict token boundary:

- `short`: one compact fact group, normally below 60 whitespace-separated words;
- `medium`: several related facts, normally 60–150 words;
- `long`: a dense multi-entity context, normally above 150 words.

Every language contains short, medium and long cases. Long contexts include irrelevant or
negative details so that an agent cannot succeed by turning every noun phrase into a
confirmed permanent type.

## How to run a qualitative inspection

For every input:

1. Load one line from `cases.no_gold.jsonl`.
2. Pass only `context + query` through the source-registry workflow.
3. Freeze and save the resulting registry before reading any review metadata.
4. Save entity mentions, canonical entities, evidence spans, assigned types, parents,
   aliases, contextual roles, unknown decisions and NLI routes.
5. Only after the artifact is sealed, join the matching line from
   `review_expectations.jsonl`.
6. Review the result semantically. Do not require exact type wording.

The current agent's general `run-fixture` command expects context, query, response and
prepared graph fixtures, so this source-only dataset must be run through the source
registry API or a future source-only CLI adapter. It must not be padded with a synthetic
answer merely to satisfy that older fixture interface.

## Review checklist

Inspect at least the following for each case:

- Were distinct real entities merged because they share a name or type?
- Were aliases of one entity left as false separate entities?
- Are parent-child links directed correctly?
- Were parent and child incorrectly merged?
- Does an entity receive several compatible parents when required?
- Are temporary roles stored separately from permanent types?
- Does relation context disambiguate model/instance and whole/component mentions?
- Are explicit negative statements respected?
- Does the system abstain on unsupported subtypes?
- Are query-only entities and claims marked with their provenance?
- Does every confirmed assignment cite source evidence?
- Was NLI routed only where a real ambiguity exists?

## Suggested review outcome

Use one of these statuses per expectation item:

```text
pass        — distinction or abstention is preserved;
partial     — usable result with avoidable granularity or evidence issue;
fail        — wrong merge, wrong type, wrong role or unsupported claim;
not_tested  — the current agent stage does not expose the needed output.
```

Record the observed labels as generated. Do not rewrite them to match the expectation
wording, because label instability is itself a diagnostic signal.

## Leakage and scientific-use rules

- `review_expectations.jsonl` is evaluation-only and never agent input.
- The files contain no RAGTruth label or detector gold label.
- This corpus is synthetic and intentionally adversarial; it does not estimate real-world
  accuracy or replace a held-out benchmark.
- Prompt, model, NLI, threshold and registry versions must be recorded with every run.
- Repeated runs should preserve input order and immutable case IDs.

## Intended next step

First run the source-only type agent over all twenty cases and inspect the resulting
registries without changing prompts. Use recurring failures to define a small number of
prompt or algorithm hypotheses, then rerun the same immutable inputs under a new recorded
version. Do not tune on this corpus and later report it as an unbiased final benchmark.
