# Deviations from GRAPHEVAL_IMPLEMENTATION_PLAN.md

Running log of where the implementation intentionally departs from the plan, with
rationale. Requested by the project owner ("запомни все свои отклонения и выпиши сюда").

## D1 — Repo layout: no `HaluVSGraph_Eval/` reparenting
**Plan (§11):** create a new parent `HaluVSGraph_Eval/` containing `hallu_smiles/`
and `graph_eval/`.
**Done instead:** `graph_eval/` is a new **top-level package inside the existing
`hallu_smiles` repo** (this repo), on branch `sasha`. HalluGraph stays at repo root
(`src/`).
**Why:** the DataSphere submitter, GitHub Actions image build, gateway, and cache
all resolve paths from the current repo root. Reparenting would break every rooted
path for no scientific gain. Plan §13 itself says to *reuse the existing pipeline
rather than create an incompatible second path* — this honors that. The package is
still independently installable (`src/` layout, own `pyproject.toml`), so it can be
split out later without code changes.

## D2 — Typed config via stdlib dataclasses, not pydantic
**Plan (§11):** `config.py  # typed config + validation` (pydantic implied by house style).
**Done instead:** stdlib `@dataclass` + an explicit `validate()`.
**Why:** the offline core must run and unit-test on the machine's Python 3.14 venv,
where pydantic (and torch/openai) are not installed. Keeping the core dependency-free
is the plan's own Stage-1 criterion ("works without network, transformers, DataSphere").
Pydantic can be layered on later behind the `yaml`/`gateway` extras if richer schema
validation is wanted.

## D3 — Extractor emits JSON triples (not the paper's freeform text list) — settled in Stage 3
**Plan (§6.2, §7.3):** `paper_prompt` returns the paper's raw list-of-triples text format.
**Done (Stage 3):** the Appendix-A prompt instructs the model to return a **JSON array**
of `[s,r,o]` string triples, and `structured_json` mode enforces it via the gateway's
`json_schema`. The parser accepts that array (or a `{"triples": [...]}` object), always
preserves raw output, reports malformed output, and flags invalid/duplicate triples.
**Why:** JSON is a stricter, machine-checkable contract than free text and is exactly what
the gateway's structured mode guarantees; it removes a class of brittle text parsing while
keeping Appendix-A *semantics* (entity/coref/relation steps, exactly three non-empty
strings, full coverage). Raw output is retained, so a paper-format parser could be added
later without losing provenance. Not a deviation from the algorithm, only its wire format.

## D4 — Shared contract lives in `graph_eval.types`; adapters in `detector_adapters/`
**Plan (§5, §11):** a common `DetectionInput/DetectionResult` contract and a
`detector_adapters/` layer created at integration.
**Done (Stage 5):** the contract is defined once in `graph_eval/types.py` (GraphEval is
its reference implementation) and the HalluGraph adapter imports it. Adapters live in a
top-level `detector_adapters/` package, separate from both frameworks. GraphEval does not
import HalluGraph; HalluGraph (`run.py`/`src/`) is unchanged; neither imports the future
experiment framework. The adapter reuses `run.build_refgraph` + `score_response` verbatim,
so it is a thin wrapper and parity with `run.build_rows` holds by construction.
**Why:** one source of truth for the contract prevents drift between the two detectors; a
separate contract-only micro-package would add packaging overhead with no benefit.

## Followed exactly (not deviations, noted to avoid confusion)
- `H(response) = max_i (1 - p_consistent_i)`; paper threshold `0.5` strict `>` (plan §6.3).
- Empty/all-invalid answer graph => `empty_graph` with `raw_score=None`, never 0/1 (plan §9).
- Transport/model/parse failure => `failed` with `raw_score=None` (never a score) (plan §5, §9).
- Extractor sees the **answer only**; NLI premise is the **context**; query preserved
  in the input but never used as evidence (plan §4, §6.1).
- Atomic content-addressed cache (`os.replace` + per-writer temp), `cache_only` miss
  is an error (plan §8).

## Pending decisions (to freeze before the first live test — plan §16)
- Reuse the existing repo gateway client vs. a fresh `openai`-based adapter in
  `extraction/gateway.py` (Stage 3). Leaning: thin `openai`-client adapter matching
  the tutorial's contract, so GraphEval stays decoupled from HalluGraph's dspy path.
- Exact HHEM Hugging Face commit SHA (`nli.revision`) — must be pinned, not `main`.
- Primary extractor mode for the frozen test: `paper_prompt` vs `structured_json`.
