# Integration contract

## Standalone API

The future facade will expose synchronous methods first:

```python
registry_run = agent.build_source_registry(source_input)
answer_run = agent.annotate_answer(answer_input, registry_run.registry)
```

An asynchronous facade may wrap the same compiled graphs, but synchronous behavior and
artifact identity remain canonical for the first pilot.

The local CLI will use JSON/JSONL inputs and an explicit output directory:

```text
hallugraph-type-agent build-registry --input source.json --output run/
hallugraph-type-agent annotate-answer --registry run/registry.json --input answer.json --output run/
hallugraph-type-agent run --input no-gold.jsonl --output runs/ --backend fake
hallugraph-type-agent validate --run runs/<id>/
```

The implemented CLI accepts `--config config/live-gateway-hhem.yaml` before the command.
It requires an explicit environment-supplied endpoint/key/model and has no silent fallback
from cache/fake to live. The source typing LLM uses the OpenAI-compatible endpoint; NLI is
performed by the independently loaded local HHEM snapshot. HHEM returns a consistency
score, so the adapter maps only the high and low tails to `entailed`/`contradicted` and
keeps the middle range `neutral`; the thresholds are explicit configuration.

## Experiment-framework adapter

The outer adapter will implement the existing `experiments.dynamic_typing.DynamicTypingProvider`:

1. translate `SourceTypingInput` and graph payloads into standalone source contracts;
2. call `build_source_registry` or load an exact compatible cached registry;
3. translate the standalone registry into the versioned outer archive schema;
4. call `annotate_answer` with the frozen registry;
5. expose standalone artifacts through `artifact_records()`;
6. map run failures to an explicit detector failure without producing a score.

The adapter owns no prompts, model policy or type decisions. The standalone package does
not import `experiments`, `src.extract` or `detector_adapters`.

## Graph conversion

The standalone graph contract accepts:

```json
{
  "graph_id": "...",
  "role": "context|query|answer",
  "entities": ["..."],
  "relations": [["subject", "relation", "object"]],
  "input_sha256": "..."
}
```

The outer adapter converts `SharedGraphArtifact` to this shape. Evidence spans are derived
from raw source text by the standalone agent and never inserted into the shared KGGen
graph, preserving the controlled comparison.

## Version handshake

Integration validates:

- standalone input/output contract version;
- outer typing-contract version;
- prompt manifest hash;
- model and NLI fingerprints;
- frozen registry checksum;
- source/context/query graph IDs;
- answer graph ID for answer annotations.

Unsupported versions fail closed. The adapter must not silently drop hierarchy, roles,
evidence or `unknown` reasons to fit the current minimal outer schema; outer schema v2 is
a planned W3 change.
