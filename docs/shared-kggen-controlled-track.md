# Controlled track: shared KGGen graphs

`controlled_shared_kggen_response_v1` is an additional experiment track. It does not
replace `kggen_untyped_adaptation`.

For every immutable no-gold RAGTruth instance, `SharedKGGraphProvider` seals the three
KGGen roles once as `(G_context, G_query, G_answer)`. The original response-controlled
comparison consumes the common `G_answer`: HalluGraph receives it through a drop-in
extractor proxy, while `SharedKGGenGraphEvalExtractor` sends its deterministically sorted
relations to the normal GraphEval parser, verbalizer and HHEM NLI over raw context.

```text
response -> shared KGGen graph -> HalluGraph graph alignment
                              -> GraphEval triples -> HHEM(context, triple)
```

HalluGraph retrieves the pre-materialized context/query graphs from the same provider.
GraphEval still uses raw context as NLI evidence. A context-graph GraphEval verifier
would be a separate ablation.

## Construction

GraphEval must state the injected backend explicitly:

```yaml
extractor:
  backend: shared_kggen
```

```python
detectors, provider = build_controlled_shared_kggen_detectors(
    hallugraph_config="config.yaml", grapheval_config=graph_config,
    gateway_manifest_sha256=manifest_sha256, cache_sources=sources,
    cache_mode="read_through")
run_paired(archive, instances_path=instances, detectors=detectors,
           shared_graph_provider=provider)
```

The factory rejects `shared_kggen` without an injected extractor. GraphEval does not
import HalluGraph or make a second extraction request.

## Audit and failure contract

`run_paired` creates `shared_graph_bundle_materialization` before detector calls. Both
prediction records contain the same answer graph reference and the same
`shared_graph_bundle_id`; the latter also fixes the context/query graph identities. The
legacy pair is valid only when `shared_response_graph_consistent=true`; the three-way
track additionally requires `all_variants_same_graph_bundle=true`.

The response graph is common preprocessing cost; HalluGraph context/query extraction and
GraphEval HHEM remain method-specific. A common extraction failure creates no surrogate
graph or score: both pending method records are failed. 429/5xx/network failures remain
transport failures.

The offline test `tests/experiments/test_shared_kggen_track.py` proves identical graph
references and a zero-KGGen-call cache-only replay. It is plumbing verification, not a
scientific result.

## Three-way extension for dynamic typing

The new `controlled_shared_all_graphs_three_way_stub_v1` track materializes all three
roles before calling any detector, then compares GraphEval, untyped HalluGraph and a
contract-only typed HalluGraph variant. The type layer is a separate annotation over
the immutable graphs; it does not modify KGGen output. Its current `unknown` provider
is deliberately score-preserving and has no model or prompt. The precise contract,
archive files and cache-only requirements are in
`docs/dynamic-typing-experiment-infrastructure.md`.

## One-response two-pass mock probe

`examples/mock_shared_kggen_one_instance.py` accepts local RAGTruth files and one
explicit `response_id`. It performs two complete framework runs with FakeKGGen/FakeNLI:

1. `read_write`: builds the shared response/context/query graphs and seals an archive;
2. `cache_only`: runs the same response against the same `cache_root`, seals a second
   archive and asserts zero KGGen calls.

```bash
python examples/mock_shared_kggen_one_instance.py \
  --source-info /path/source_info.jsonl \
  --responses /path/response.jsonl \
  --response-id 6845 \
  --output-root /path/probe-output \
  --cache-root /path/shared-kg-cache
```

The probe does not use a gateway, DataSphere secrets, real KGGen or HHEM. It exercises
the real adapters, no-gold projection, graph provenance, cache lookup and archive sealing
on exactly one selected RAGTruth record.
