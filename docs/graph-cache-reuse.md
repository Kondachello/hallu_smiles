# Reusing historical KGGen graph caches

Historical graphs are scientific artifacts. They are read-only inputs, never copied into
or silently relabelled by a run.

`vertex-100qa-hypothesis-report.md` confirms the earlier 100-QA strict/support comparison
used one KG cache and successful cache-only replays, but intentionally does not prescribe
a portable filesystem path. The framework therefore exposes an explicit `cache_root`
instead of hard-coding a DataSphere mount path.

## Source contract and modes

Each `GraphCacheSource` identifies an immutable directory of `hallu-kg-cache-v2` envelopes,
an optional lineage manifest, priority and (when needed) an explicit compatibility key schema.
Every envelope is validated for protocol,
filename/key agreement, canonical graph SHA-256 and graph round-trip. Two sources that
return distinct graph hashes for the same key cause a hard conflict.

The only currently supported historical schema is `kggen-v11-pre-length-retry`: it is
available exclusively to a declared read-only source and recomputes the former v11 key
without `length_retry_attempts` and `length_retry_max_tokens`. New cache writes continue
to use `kggen-v11-current`; no generic fallback or change to current keys is allowed.

| Mode | Behaviour |
|---|---|
| `cache_only` | Every expected graph must validate; a miss is an integrity error and no KGGen backend is constructed. |
| `read_through` | Reuse hits; write misses only into the current writable namespace. |
| `read_write` | Use the current namespace normally. |
| `live_fresh` | Ignore historical sources. |
| `inspect_only` | Validate source structure/conflicts only. |

## Inspect existing graphs

The command below never downloads models, calls a gateway, reads secrets or runs a
detector. With `--instances`, it recomputes cache keys from no-gold input and reports
`compatible_hit`, `miss` or `corrupt` per response/role.

```bash
python -m experiments.cli cache inspect \
  --hallugraph-config config.yaml \
  --source vertex-100qa=/path/to/historical/kg \
  --instances /path/to/instances.no_gold.jsonl \
  --role response \
  --output cache-compatibility-report.json
```

Preserve the original extraction identity/lineage alongside historical caches. The cache
key includes extractor protocol, runtime fingerprint, token limits, clustering and
chunking; matching a model name alone does not prove compatibility.

## Storage interface for DataSphere

Pass a Project-storage directory as `cache_root` to a two-pass probe or controlled run.
The first materializing run may write only under that root; the second uses the same root
with `cache_only`. Historical trees are attached as `GraphCacheSource(source_id, root)` and
remain read-only. The job wrapper, rather than Python code, chooses the Project-storage
mount path, so local development may use any writable temporary directory.

Before live reuse of the historical ~100 graphs, run `experiments.cli cache inspect` against
the mounted source and selected `instances.no_gold.jsonl`; only a compatible report permits
`cache_only` reuse.

## Historical 100-QA replay Job

`docs/datasphere-historical-qa-cache-replay.md` describes the DataSphere proof run.
It resolves the recorded lineage from Project storage, reconstructs the historical
selection, then chooses the first record with compatible context/query/response
graphs. It runs both detectors in `cache_only` and records zero KGGen and GraphEval
extractor calls. The `cache_only` preflight supports an explicitly non-strict
inventory mode only for this selection step; actual graph materialization remains
strict and fails on a miss. Its registered historical source additionally declares the
pre-length-retry key schema, so a hit is recorded with both the requested current key
and the resolved historical key schema in the coverage and graph-resolution reports.

## Auditable output

A controlled run records `cache/cache_inventory.json`, `cache/cache_resolution.jsonl` and
`shared_graphs/graph_index.jsonl`. The prediction seal checksums these files when present.
Secrets and gateway credentials must never be written into cache metadata or artifacts.
