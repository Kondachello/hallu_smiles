# HalluGraph Dynamic Typing Agent

This directory is the autonomous package boundary for the model-backed dynamic entity
typing agent. It is intentionally separate from the experiment runner, KGGen and
HalluGraph scoring code.

Current status: **functional standalone core, verified offline**. It contains source and
answer LangGraph graphs, immutable file cache and auditable artifacts, a deterministic
fake model/NLI mode, strict contracts, a local CLI, and an optional LiteLLM transport
that fails closed when its dependency or explicit live configuration is absent. The
outer experiment-framework adapter and HalluGraph B5 scoring remain deliberately
separate workstreams.

The package will expose two operations:

1. `build_source_registry(context, query, context_graph, query_graph)` — produces and
   freezes a source-only hierarchical registry.
2. `annotate_answer(response, answer_graph, frozen_registry)` — assigns only types from
   that registry or abstains.

Neither operation accepts gold labels. Transport/protocol failures are distinct from
epistemic `unknown` decisions.

Key documents:

- [Architecture](docs/ARCHITECTURE.md)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Prompt catalog](docs/PROMPT_CATALOG.md)
- [Integration contract](docs/INTEGRATION.md)
- [Test plan](docs/TEST_PLAN.md)

Offline checks (from this directory):

```bash
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python scripts/validate_spec.py
```

Run independently after installation (the default is deterministic fake mode; it makes
no network request):

```bash
python -m pip install .[dev]
hallugraph-type-agent run-fixture --input examples/dynamic_typing_20.no_gold.jsonl --output runs/fake
```

During source-tree development, use `PYTHONPATH=src` before `python -m
hallugraph_dynamic_typing ...`.

## Live LLM plus local HHEM NLI

`config/live-gateway-hhem.yaml` uses a real OpenAI-compatible LLM endpoint for source
typing and a local, pinned HHEM snapshot for answer-side NLI. It deliberately contains no
secret. Copy `env.example.ps1` to `env.local.ps1`, put the values in the copy, then load it
into the current PowerShell process. On this Windows workspace use Python 3.12 for the
live virtual environment: the installed Python 3.13 does not provide a reliable `torch`
runtime for HHEM.

```powershell
& "C:\Users\Kolya\AppData\Local\Programs\Python\Python312\python.exe" -m venv --system-site-packages .venv-local-live
.\.venv-local-live\Scripts\Activate.ps1
python -m pip install -e ".[live]"
Copy-Item env.example.ps1 env.local.ps1
# Edit env.local.ps1 locally; do not commit it.
. .\env.local.ps1
$env:PYTHONPATH = "src"
python -m hallugraph_dynamic_typing --config config\live-gateway-hhem.yaml run-fixture `
  --input examples\dynamic_typing_20.no_gold.jsonl --output runs\live-one --limit 1
```

Required environment variables: `HALLU_GATEWAY_URL`, `HALLU_GATEWAY_API_KEY`,
`HALLU_TYPING_MODEL`, and `HALLU_HHEM_MODEL_PATH`. The HHEM path must already contain the
pinned local snapshot, including `config.json`; runtime downloads are intentionally
disabled. `HALLU_GATEWAY_URL` is the Cloud Run origin used by the existing extractors;
the agent adds `/v1` exactly once before calling the OpenAI-compatible API. The configured
revision and thresholds are recorded in the configuration, while the run artifacts contain
no secret value. Temporary 429/5xx/network failures are retried at most five times; a
failed fixture returns a non-zero process code and its `summary.json` has a redacted cause.
The project Vertex gateway accepts a simple native schema but rejects the agent's deeply
nested source contract. The live profile therefore puts the complete schema into the system
instruction and applies the same complete schema as a local, fail-closed response validator.

Provision the ignored snapshot and official corpus once, then verify them offline:

```powershell
$env:PYTHONPATH = "src"
python scripts\provision_local_resources.py
python scripts\provision_local_resources.py --verify-only
```

This stores HHEM-2.1-Open at `local_resources\hhem-2.1-open` and the official pinned
RAGTruth files at `local_resources\ragtruth`, plus the pinned FLAN-T5 tokenizer/configuration
files that HHEM needs at `local_resources\flan-t5-base`. The generated `manifest.json`
records source commits, file sizes and SHA-256 values; all payloads remain ignored by Git.

## Interactive result viewer

Create one self-contained, local HTML explorer for a completed no-gold run:

```powershell
$env:PYTHONPATH = "src"
python scripts\render_run_viewer.py --run runs\live-test
```

It presents context, query and answer text, evidence spans, entity assignments, frozen type
hierarchy and NLI result. A run created before this viewer was added needs the fixture
fallback once: `--fixture examples\dynamic_typing_20.no_gold.jsonl`. Details:
[run viewer](docs/RUN_VIEWER.md).
