# Unified local test framework

## Goal

One command runs the type agent regardless of whether a case already contains knowledge
graphs or contains only raw text that must first pass through KGGen. Input adaptation
changes, but the saved artifacts, dashboard and case explorer do not.

The framework is no-gold by contract. Top-level gold/label fields are rejected before
model or KGGen execution.

## Input

The input is UTF-8 JSONL with stable, filesystem-safe `case_id` values.

Raw text:

```json
{
  "case_id": "bank-001",
  "source_id": "source-001",
  "context": "North Bank is a commercial bank.",
  "query": "Which organization is described?",
  "response": "North Bank is the organization."
}
```

Supplied graphs add:

```json
{
  "graphs": {
    "context": {"entities": ["North Bank"], "relations": []},
    "query": {"entities": ["organization"], "relations": []},
    "answer": {"entities": ["North Bank"], "relations": []}
  }
}
```

`--input-mode auto` uses supplied graphs when a row has `graphs`; otherwise it routes the
row through KGGen. A single JSONL may mix both forms. `--input-mode text` deliberately
ignores supplied graphs and extracts fresh graphs. A non-empty response requires an
answer graph in supplied-graph mode.

Use `--case-id <id>` to select an exact immutable row from a larger suite. The option may
be repeated; it is preferable to copying a diagnostic record into a temporary input file.

## Commands

Prepared graph or historical-cache fixture:

```powershell
$env:PYTHONPATH = "src"
python -m hallugraph_dynamic_typing `
  --config config\live-gateway-hhem.yaml `
  test `
  --input local_resources\historical_100qa_graph_cache\provenance\dynamic_typing_fixture.no_gold.jsonl `
  --input-mode graphs `
  --limit 1 `
  --output runs\historical-one-v3
```

Raw text with offline fake KGGen and fake type model:

```powershell
$env:PYTHONPATH = "src"
python -m hallugraph_dynamic_typing test `
  --input examples\text_kggen_smoke.no_gold.jsonl `
  --input-mode text `
  --kggen fake `
  --limit 1 `
  --output runs\unified-text-kggen-smoke
```

Raw text with real KGGen, gateway type model and local HHEM:

```powershell
.\.venv-local-live\Scripts\Activate.ps1
. .\env.local.ps1
$env:PYTHONPATH = "src"
python -m hallugraph_dynamic_typing `
  --config config\live-gateway-hhem.yaml `
  test `
  --input examples\text_kggen_smoke.no_gold.jsonl `
  --input-mode text `
  --kggen live `
  --kggen-config ..\..\hallu_smiles\config.yaml `
  --kggen-cache-root .cache\kggen-unified-smoke `
  --limit 1 `
  --output runs\unified-text-kggen-live-one
```

The real command is intentionally bounded to one case. Do not use it for 20/100 cases
until the one-case artifacts and gateway cost are reviewed.

For example, inspect only the long Russian industrial model/instance case:

```powershell
$env:PYTHONPATH = "src"
python -m hallugraph_dynamic_typing test `
  --input datasets\context_query_diagnostic_20\cases.no_gold.jsonl `
  --case-id cqdiag-ru-009-industrial-model-instance-long `
  --input-mode text `
  --kggen fake `
  --output runs\cqdiag-ru-009-industrial-model-instance-fake-utf8
```

The fake mode verifies the local pipeline and viewer only. Its Unicode fallback is a toy
co-occurrence graph, not an evaluation of KGGen or semantic typing quality.

The output directory is immutable: it must be new or empty. Use a new run name for every
attempt. This prevents stale failures or annotations from an earlier attempt from being
mixed with current results.

## Canonical output

```text
<output>/
  run_manifest.json
  summary.json
  <case-id>/
    input_snapshot.json
    source_registry.json
    answer_annotations.json       # only when a response exists
    execution_trace.json
    manifest.json
    failure.json                  # only for adapter/runtime failure
  viewer/
    index.html
    ...
```

`run_manifest.json` is the canonical run-level entry. It records the requested adapter,
KGGen protocol, no-gold flag, per-status counts, per-case mode, metrics, failure and
relative artifact directory. `input_snapshot.json` is always normalized to
`typing-test-input-v2` and records graph provenance as `supplied` or `kggen`.

`execution_trace.json` starts with input-adaptation events, so the viewer shows whether a
graph was loaded or extracted by KGGen before displaying every source, hierarchy, NLI
and answer stage.

The same case files are produced for both modes. Consumers and the viewer never need to
know which command generated a case before reading its snapshot.

## Failure behavior

An adapter, KGGen, model, NLI or validation error writes a redacted `failure.json` for
that case and marks it failed in the run manifest. Remaining cases continue. The command
returns a non-zero process status if at least one case failed. Semantic uncertainty is
not converted into a transport failure, and a transport failure is never converted into
an `unknown` type.
