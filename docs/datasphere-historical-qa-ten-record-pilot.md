# Historical 100-QA cache: ten-record pilot

This is a CPU-only pilot over ten reproducibly selected records from the immutable
historical 100-QA graph cache. It is a cache-reuse and paired-adapter integration check,
not a quality evaluation over the full dataset.

Use `--replay-count 10 --replay-selection-seed 917`. The selection is random but
reproducible, and it contains only records with complete response/context/query graph
triplets. `cache_only` remains mandatory: a cache miss is an error, never a live fallback.

## Progress and logs

The Job writes an event after selection and after each record's graph read, HalluGraph
prediction, GraphEval prediction, and completion. Every event begins with
`HISTORICAL_REPLAY_PROGRESS` and has record index/total, identifiers, status, score and
elapsed time. It excludes gold fields and raw RAGTruth text.

The task command sends these events to stdout via `tee`. Open the DataSphere Job and its
**Logs** tab to view them while the service exposes new output; output can be buffered by
DataSphere, so this is an observation aid rather than a control mechanism. The guaranteed,
complete sources after completion are:

- `historical-cache-replay/reports/progress.jsonl` — machine-readable event stream;
- `replay.console.log` — combined stdout/stderr;
- `predictions/raw_predictions.jsonl`, `paired_predictions.jsonl` and
  `stages/stage_calls.jsonl` — per-method and per-stage evidence;
- `historical_qa_cache_replay_report.json`, coverage report, run manifest and seal.

The archived report must still show zero KGGen API calls, zero GraphEval extractor calls,
zero gateway LLM calls, `historical_100qa` graph provenance and two `ok` predictions for
every selected record.
