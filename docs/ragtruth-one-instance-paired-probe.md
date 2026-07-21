# RAGTruth: one-instance paired offline probe

`scripts/ragtruth_one_instance_paired_probe.py` is the bounded integration check for
one explicitly named local RAGTruth response. It answers a narrow question: can the
framework pass the same no-gold object through the current HalluGraph and GraphEval
adapters, record their outputs, and produce a valid sealed archive?

It is **not** a scientific experiment, a metric calculation, threshold tuning, or a
full-dataset run.

## What is real and what is fake

The probe invokes the real repository adapters and their production scoring flow:

- `detector_adapters.hallugraph_adapter.HalluGraphAdapter` with the normal
  `run.build_refgraph` and `src.metrics.score_response` path;
- `graph_eval.detector.GraphEvalDetector` with the normal answer-only extraction,
  triple parsing, NLI scoring, and aggregation path;
- the common `experiments.runner.run_paired` and archive sealing path.

Model-facing dependencies are deliberately deterministic substitutes:

- HalluGraph uses `FakeKGGen` and `DictEmbedder`;
- GraphEval uses `FakeExtractor` and `FakeNLI`.

Successful output proves interface wiring, score-direction transport, no-gold routing,
paired prediction storage, and checksums. It does **not** prove the quality or
numerical behaviour of either method with live models.

## Safety properties

- `--response-id` is required; exactly one response is materialized.
- The script only consumes already local `source_info.jsonl` and `response.jsonl`.
  It has no dataset-download option and makes no network calls.
- Raw RAGTruth `labels` and `quality` are never copied into the detector input,
  probe manifest, predictions, or terminal summary. `assert_no_gold` protects this
  boundary before inference.
- There is no `--live` option. Live inference remains a separately reviewed
  DataSphere workflow after its gateway, secret, cache, and preflight gates are ready.
- No gold join, metric computation, tuning, or report comparison is performed.

## Use

Run only after the two raw RAGTruth files are already present locally:

```bash
python scripts/ragtruth_one_instance_paired_probe.py \
  --data-dir data/raw/<pinned-revision> \
  --response-id <exact-response-id> \
  --output-root results/ragtruth_one_instance_probe
```

Equivalently, provide both paths directly:

```bash
python scripts/ragtruth_one_instance_paired_probe.py \
  --source-info /path/to/source_info.jsonl \
  --responses /path/to/response.jsonl \
  --response-id <exact-response-id>
```

The output is a new archive directory; the script refuses to overwrite an existing
run id. The concise terminal summary includes detector statuses and raw risk scores
only; it never prints gold labels.

```text
results/ragtruth_one_instance_probe/<run-id>/
  run_manifest.json
  probe_manifest.json
  instances.no_gold.jsonl
  stages/stage_calls.jsonl
  predictions/raw_predictions.jsonl
  predictions/paired_predictions.jsonl
  prediction_seal.json
  reports/one_instance_probe_report.json
```

The automated test `tests/experiments/test_one_instance_probe.py` creates a tiny
RAGTruth-shaped fixture whose raw response intentionally includes labels and quality.
It verifies that both real adapters finish on their fake backends, both predictions
are paired and sealed, and those evaluation-only fields cannot leak into the no-gold
object or the human-readable output.
