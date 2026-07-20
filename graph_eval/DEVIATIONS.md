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

## D3 — Parser canonicalizes to JSON; paper's raw list-of-triples handled in Stage 3
**Plan (§6.2, §7.3):** `paper_prompt` returns the paper's raw list-of-triples format.
**Done so far:** the Stage-1 parser accepts a JSON array of `[s,r,o]` (or an object
with a `triples` list). The exact Appendix-A textual format parser lands with the
gateway extractor in Stage 3; raw output is always preserved (`ParseOutcome.raw_output`)
and malformed output is reported, never guessed.
**Why:** Stage 1 is backend-agnostic core; the paper-format specifics belong with the
real extractor. Not a semantic deviation — the contract (ordered triples, keep raw,
flag invalid/duplicate) is already met.

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
