# HalluGraph-KGGen — HalluGraph hallucination detection with KGGen, evaluated on RAGTruth

A faithful, runnable reproduction of the **HalluGraph** response-level hallucination detector
(Noël et al., 2025, [arXiv:2512.01659](https://arxiv.org/abs/2512.01659)) with **one substitution**:
the knowledge-graph extractor is **KGGen** (Mo et al., 2025,
[arXiv:2502.09956](https://arxiv.org/abs/2502.09956), `pip install kg-gen`) instead of
spaCy NER + a custom SLM triple extractor. Evaluated on **RAGTruth**
(Niu et al., 2024, [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)).

## What it does

For each RAGTruth `(context C, query Q, response A)`:

1. Extract three KGs with KGGen: `G_c`, `G_q`, `G_a`.
2. Against the reference graph `G_ref = G_c ∪ G_q`, compute
   **Entity Grounding (EG)**, **Relation Preservation (RP)** (edge-aware), and
   **Composite Fidelity Index** `CFI = α·EG + (1−α)·RP`.
3. Hallucination score `H = 1 − CFI` (higher = more likely hallucinated).
4. Evaluate `H` as a detector vs. RAGTruth human labels: ROC-AUC (primary) + P/R/F1 @ a
   train-tuned threshold, per-task and per-model, with bootstrap CIs and ablations.
5. Emit a per-response **audit trail** naming the exact ungrounded entities / unsupported
   relations that produced the score.

---

## 1. Install

**Use Python 3.10–3.12 for a live run.** The light deps (numpy/scipy/scikit-learn/pandas/…)
work on 3.10–3.14, but `torch` (pulled by `sentence-transformers`) and `kg-gen` have the most
reliable wheels on 3.10–3.12. The **unit tests run fully offline** and need none of the heavy deps.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download RAGTruth

```bash
python download_data.py            # -> data/source_info.jsonl, data/response.jsonl (~36 MB)
```

## 3. Set the LLM model + API key

The backend LLM is defined in **exactly one place**: `llm.model` in `config.yaml`
(a LiteLLM-style string). Nothing else hardcodes a model.

**OpenRouter**
```bash
export OPENROUTER_API_KEY=sk-or-...
```
```yaml
# config.yaml
llm:
  model: "openrouter/openai/gpt-4o-mini"   # or openrouter/meta-llama/llama-3.1-8b-instruct
  api_key_env: "OPENROUTER_API_KEY"
```

**NVIDIA NIM**
```bash
export NVIDIA_API_KEY=nvapi-...
```
```yaml
llm:
  model: "nvidia_nim/meta/llama-3.1-8b-instruct"
  api_base: "https://integrate.api.nvidia.com/v1"   # optional; LiteLLM also infers it
  api_key_env: "NVIDIA_API_KEY"
```

## 4. Run

```bash
python run.py --config config.yaml --stage all
```

Stages (each resumable; artifacts persisted under `results/`):

| stage | does | writes |
|---|---|---|
| `extract` | build `G_c`,`G_q` (once per source) + `G_a` (per response); fills disk cache | `.cache/kg/*.json`, `failed_extractions.jsonl` |
| `score` | EG / RP / audit lists for every response | `results/scored.jsonl` |
| `tune` | **train only**: α (5-fold CV), θ (F1), τ_e×τ_r sweep | `results/tuning.json` |
| `evaluate` | score **test** once: AUC/P/R/F1/ablation/bootstrap/diagnostics | `results/metrics.csv`, `report.md`, `plots/` |
| `all` | all of the above | everything + `results/audit/{id}.json` |

### Offline plumbing check (no API key, no torch)

```bash
python tests/make_fixture.py tests/fixture_data          # tiny synthetic RAGTruth-format data
python run.py --stage all --fake-extractor --data-dir tests/fixture_data --output-dir results_smoke
```
`--fake-extractor` swaps in a dependency-free `FakeKGGen` + a deterministic `DictEmbedder`, so the
**entire pipeline** (extract → cache → score → tune → evaluate → audit) runs with zero network and
zero torch. It exercises plumbing only — the numbers are meaningless.

### Fixed 20-source QA relation pilot

The pilot selects one response per QA `source_id`: 16 train rows (8/8 labels) and 4 test rows
(2/2 labels). It writes a manifest; reuse that exact file for the support run so strict and
text-verified metrics see identical `(C,Q,A)` triples.

```bash
# Baseline: existing graph-edge RP semantics, no verifier LLM calls.
python run.py --stage all --relation-mode strict --qa-pilot \
  --qa-pilot-manifest-out results/qa_pilot_manifest.json \
  --output-dir results/qa_pilot_strict

# Support variant: verifier checks each grounded answer edge against C/Q text.
python run.py --stage all --relation-mode support \
  --qa-pilot-manifest results/qa_pilot_manifest.json \
  --output-dir results/qa_pilot_support
```

`RP_strict` remains the historical edge-alignment score. The support run additionally reports
`RP_grounded`, `RP_entailed_cond`, and `RP_support`; it caches text-verifier verdicts under
`.cache/verdicts/`. The verifier uses the same `llm.model` as KGGen and returns only
`entailed`, `contradicted`, or `unknown` for a canonical triple plus up to four source sentences.

### DataSphere batch job

DataSphere Jobs mount Project storage for reads.  Therefore Llama 3.1 8B is staged **once**
from a cheap `c1.4` Jupyter session into shared storage, and GPU Jobs never download it or install
packages at runtime.  The strict and support modes then run sequentially under one vLLM server on
the same 20-QA manifest and job-local KG/verdict caches.

```bash
# In DataSphere Jupyter. Start at the Jupyter Project directory, then use the
# repository's shared/ folder. DS_PROJECT_HOME is a Job-only mount variable.
# Llama is gated: HF_TOKEN is read only here; the GPU Job never receives it.
cd /home/jupyter/project/hallu_smiles
export DS_SHARED_ROOT="$PWD/shared"
python scripts/stage_datasphere_shared_assets.py --shared-root "$DS_SHARED_ROOT" \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct

# Locally, pin, validate and submit a read-only preflight followed by the one-GPU pilot.
# The helper refuses an unpushed commit and catches DataSphere CLI/YAML mistakes before
# it creates a cloud Job.  Use a Python environment that contains the `datasphere` package.
PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh --kind preflight --project-id <PROJECT_ID> \
  --run-id preflight-20260716 --branch new-metrics

PYTHON_BIN=.venv-datasphere/bin/python \
DATASPHERE_BIN=.venv-datasphere/bin/datasphere \
bash scripts/submit_datasphere_job.sh --kind qa-pilot-g1 --project-id <PROJECT_ID> \
  --run-id new-metrics-20260716 --branch new-metrics
```

The GPU configuration is one `g1.1` V100 in FP16, with `max-model-len=8192` and a hard
three-hour limit (777,600 units plus at most 60 seconds of graceful shutdown).  Per-run outputs
contain the strict/support reports, comparison, audits, vLLM log, GPU utilization trace, and
metadata.  See [the team DataSphere runbook](docs/datasphere-team-runbook.md) for the shared-storage
contract and monitoring procedure.

## 5. Outputs (`results/`)

- `metrics.csv` — per-response scored rows (EG, RP, H, graph sizes, split, task, model…).
- `summary_metrics.csv` — flat headline numbers.
- `report.md` — every §6 table: headline AUC + 95% CI, P/R/F1 @ θ, AUC by task, AUC by
  generator model, P/R/F1 by task, **EG-only / RP-only / CFI ablation**, AUC-vs-context-length
  buckets, graph statistics, Mann-Whitney significance, degenerate-case counts, tuning trace.
- `plots/h_distributions.png`, `plots/auc_vs_context_length.png`.
- `audit/{response_id}.json` — the HalluGraph audit trail (see §7 of the spec).
- `failed_extractions.jsonl`, `usage.jsonl` (per-call timing + cumulative cost/tokens).

---

## Reproducibility & determinism

- `temperature: 0.0` everywhere; fixed `eval.seed`.
- **Disk cache** keyed by `sha256(model + prompt_params + input_text)` at `.cache/kg/`.
  A warm cache makes **zero API calls** and reproduces **byte-identical** `metrics.csv`
  (verified: only `usage.jsonl` / the API-usage line of `report.md` differ, since they honestly
  report that 0 calls were made on the cached run).
- Cache writes are atomic (`os.replace`) with **per-writer unique temp names**, so concurrent
  extraction of identical texts (RAGTruth has duplicate responses) is crash-safe and race-free.
- **`G_c` is extracted once per `source_id`** (see `unique_sources` in `run.py`), reused across
  that source's responses; identical texts also dedupe through the content-addressed cache.
- **No test leakage:** α, θ, and the τ sweep are computed on the **train split only**; the test
  split is scored exactly once in `evaluate`.

## Adaptation to the real `kg-gen` API (per spec §3 instruction — no fork / no monkey-patch)

The installed `kg-gen` is used directly; two details differ from the §3 snippet and are handled
in `src/extract.py`:

1. **Chunking is native.** Long contexts use `kg.generate(input_data=..., chunk_size=..., cluster=True)`,
   which chunks → aggregates across chunks → runs one clustering pass internally — exactly the
   behavior §3 asks for. We therefore map `extraction.context_chunk_chars` → `chunk_size` instead
   of merging chunks by hand.
2. **Graph object.** `graph.entities` (`set[str]`) and `graph.relations` (`set[(s,p,o)]`) are used
   as documented; the object also exposes `.edges` / `*_clusters`, which we don't need.

## Matching (HalluGraph `match`/`align` adapted to KGGen's untyped strings)

KGGen entities are untyped strings, so `match(v,w)` is: exact normalized equality **or**
token-boundary substring (shorter side ≥ `min_substring_chars`, non-stopword) **or**
S-BERT cosine ≥ `τ_e`. `align((s,r,o),(s',r',o'))` requires `match(s,s') ∧ match(o,o')`
(direction-sensitive; `inverse_edge_match` ablation available) and relation-label compatibility
(exact **or** cosine ≥ `τ_r`). All thresholds are config parameters; a τ_e∈{0.85,0.90,0.95} ×
τ_r∈{0.65,0.75,0.85} sensitivity sweep is reported on train.

## Notes on statistics

The §6.1 "Wilcoxon signed-rank" comparison is between two **independent, unequal-length** groups
(factual vs. hallucinated responses), where the signed-rank test does not apply. We report the
correct rank-based test for that design — **Mann-Whitney U / Wilcoxon rank-sum** — and label it as
such in the report.

## Expected runtime & cost

Dominated by KGGen LLM calls. RAGTruth has ~2.7k test responses over 450 sources plus the train
split (~15k responses / ~2.6k sources), and each instance needs up to 3 extractions
(`G_c` once per source, `G_q`, `G_a`). Order-of-magnitude with a small hosted model
(e.g. `gpt-4o-mini` / an 8B model) at `concurrency: 4`:

- **Extractions:** ≈ (#unique sources) + (#unique responses) + (#QA queries) LLM calls.
  Ballpark tens of thousands of calls for the full corpus; a few dollars to low-tens of dollars
  depending on the model and context lengths. `usage.jsonl` logs per-call timing and a cumulative
  cost estimate (best-effort via LiteLLM).
- **Everything after extraction** (embedding, matching, metrics, tuning, evaluation, plots) is
  local and runs in minutes on CPU.
- Re-runs are ~free: the cache serves all previously-seen inputs.

Tip: start with `--limit 200` and/or a single task to validate config + budget before a full run.

## Tests

```bash
pytest -q      # 36 tests, fully offline (serializer, normalization, match/align, EG/RP/CFI edge cases)
```

## Layout

```
hallugraph-kggen/
  config.yaml            # all knobs; llm.model is the only place the backend is named
  download_data.py       # fetch RAGTruth JSONL
  run.py                 # single entrypoint (stages: extract|score|tune|evaluate|all)
  src/
    config.py            # YAML -> attribute config
    data.py              # RAGTruth load, (C,Q,A) construction, Data2txt serializer
    extract.py           # KGGen wrapper: cache, retries, chunking, cost log, FakeKGGen
    matching.py          # normalize, Embedder, RefGraph, match(), align()
    metrics.py           # EG, edge-aware RP, CFI, H, degenerate policies
    tune.py              # alpha CV, F1 threshold, tau sweep helpers (train only)
    evaluate.py          # AUC/P/R/F1, ablation, bootstrap, diagnostics, plots, report.md
    audit.py             # per-response audit-trail writer
  tests/                 # offline unit tests + synthetic-fixture generator
```

## Mapping to the spec

| spec | where |
|---|---|
| §1 config / determinism / cache / cost log | `config.yaml`, `src/config.py`, `src/extract.py` |
| §2 (C,Q,A) construction + Data2txt serializer + labels | `src/data.py` |
| §3 KGGen extraction, chunking, once-per-source `G_c`, failures | `src/extract.py`, `run.py:extract_all` |
| §4 `match()` / `align()`, thresholds, τ sweep | `src/matching.py`, `run.py:tau_sweep` |
| §5 EG / RP (edge-aware) / CFI / H, degenerate cases | `src/metrics.py` |
| §6 AUC/P/R/F1, per-task/model, ablation, bootstrap, diagnostics | `src/evaluate.py` |
| §7 audit trail | `src/audit.py` |
| §8 structure, entrypoint, tests, acceptance criteria | this repo |
