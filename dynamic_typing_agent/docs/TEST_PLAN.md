# Test plan

## Test pyramid

### Pure domain tests

- canonical serialization and stable IDs;
- source-span extraction and exact offset round-trip;
- hierarchy cycle detection;
- alias/parent incompatibility;
- merge/split append-only history;
- evidence-level policy;
- `unknown` versus `failed` separation;
- answer cannot extend a frozen registry;
- no forbidden gold keys in accepted input/state.

### Prompt contract tests

- every model/NLI node has exactly one prompt manifest entry;
- all referenced files exist and remain inside the version directory;
- system/user files are non-empty English instructions;
- Jinja variables equal the manifest allowlist;
- every schema is strict JSON Schema with `additionalProperties: false`;
- examples embedded in prompts contain no RAGTruth labels;
- prompt hashes and complete manifest hash are deterministic.

### Node tests with fakes

- success, abstention and protocol-error response for every model node;
- routing sends only ambiguous/high-impact decisions to NLI;
- `neutral` maps to unknown/preliminary, never contradiction;
- retries do not duplicate artifacts or decisions;
- parallel results are deterministically ordered;
- bounded repair terminates;
- checkpoint resume makes zero calls for completed nodes.

### Graph tests

- source graph state cannot contain answer text;
- answer graph requires `frozen=true` and valid registry checksum;
- registry cache hit skips all source model nodes;
- NLI cache hit skips NLI transport;
- transport/protocol failure terminates with `failed`;
- same semantic input and versions produce byte-identical registry output;
- entity-order and completion-order permutations preserve the registry.

### Integration tests

- local JSON run with deterministic fakes;
- existing three-way experiment receives the same shared graph bundle;
- unknown-only provider remains score-preserving;
- model-backed adapter writes richer artifacts without mutating graphs;
- cache-only replay performs zero model/NLI calls;
- no-gold fixture loader never reads the expectation file.

### Live contract probe

Only after explicit authorization: one synthetic case, concurrency 1, bounded requests,
no DataSphere submission by default. Validate structured output support, gateway manifest,
usage accounting and cache replay before any dataset pilot.

## Twenty-case corpus

`examples/dynamic_typing_20.no_gold.jsonl` contains only agent inputs.
`examples/dynamic_typing_20.expectations.jsonl` contains human review expectations and is
never loaded by the agent. The specification tests prove identical case IDs and absence of
expectation-only keys in the no-gold file.

