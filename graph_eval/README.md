# GraphEval — standalone hallucination detector

Answer-only knowledge-graph extraction + per-triple NLI verification, per
[GraphEval (Sansford et al., 2024)](https://arxiv.org/abs/2407.10793), adapted to a
Cloud Run / Vertex Gemini extractor and a local HHEM NLI verifier.

This is **framework #2** of three (see `../GRAPHEVAL_IMPLEMENTATION_PLAN.md`):

| # | Framework | Owner | Location |
|---|---|---|---|
| 1 | HalluGraph (EG/RP/CFI, 3 graphs) | existing | repo root `src/` |
| 2 | **GraphEval (this package)** | us | `graph_eval/` |
| 3 | Experiment runner (RAGTruth, split, tuning, metrics) | a teammate | future `experiments/` |

GraphEval never loads RAGTruth and never sees gold labels. It maps one
`(context, query, response)` to a raw `DetectionResult` (higher score = more likely
hallucination). All research metrics/thresholds are the experiment framework's job.

## Algorithm
1. Extract triples from the **answer only**.
2. Verbalize each triple `"<s> <r> <o>."` and run NLI with premise = **context**,
   getting `p_consistent`.
3. `p_unsupported = 1 - p_consistent`; response score `H = max_i p_unsupported_i`.
4. Paper decision: hallucinated iff `H > 0.5` (a train-tuned threshold is applied
   separately, by the experiment framework).

## Status
- **Stage 1 — offline core: done.** Types/contract, parser, verbalizer, scoring,
  atomic cache, fake extractor + fake NLI, `GraphEvalDetector`. 25 unit tests, fully
  offline (no torch/openai/network).
- Stage 2 — local HHEM NLI adapter (pinned revision): pending.
- Stage 3 — Cloud Run gateway extractor (+ `structured_json` variant, retries, cache): pending.
- Stage 4 — standalone CLI (JSONL in/out): pending.
- Stage 5 — `HalluGraphDetectorAdapter` + parity check: pending.

## Run the tests (offline)
```bash
PYTHONPATH="$PWD/graph_eval/src" .venv/bin/python -m pytest graph_eval/tests -q --noconftest
```

## Library usage
```python
from graph_eval import GraphEvalDetector, DetectionInput
from graph_eval.extraction.fake import FakeExtractor
from graph_eval.nli.fake import FakeNLI

det = GraphEvalDetector(FakeExtractor(), FakeNLI())
res = det.predict(DetectionInput(
    response_id="r1", source_id="s1",
    context="Paris is the capital of France.",
    response="Paris is the capital of France.",
    query="What is the capital of France?",
))
print(res.status, res.raw_score)   # ok <low score>
```
Real backends (`backend: gateway` / `backend: hhem`) arrive in Stages 2–3 and require
the `gateway` / `hhem` extras. The gateway secret is never stored in config — only the
env-var name (`HALLU_GATEWAY_API_KEY`) is referenced. See `config.example.yaml`.

## Layout
```
graph_eval/
  pyproject.toml            # src-layout, dependency-free core; gateway/hhem extras
  config.example.yaml       # redacted (no secret)
  DEVIATIONS.md             # departures from the plan, with rationale
  src/graph_eval/
    types.py                # DetectionInput/Result, Triple, statuses
    parser.py verbalize.py scoring.py cache.py usage.py config.py
    detector.py             # GraphEvalDetector facade
    artifacts.py            # prediction record + JSONL writer
    extraction/ (base, fake; gateway/vllm later)
    nli/        (base, fake; hhem later)
  tests/
```
