# HalluGraph-KGGen

Response-level hallucination detection for RAG answers using knowledge graphs extracted by **KGGen**. The project adapts the HalluGraph idea to RAGTruth QA and adds two evidence-aware variants: text-supported relations and atomic-claim verification.

> **Project status — archived experimental implementation.** The final RAGTruth result is an archive-verified 750-response experiment with a zero-network cache-only replay. The separate DocRED work is a smaller, explicitly preliminary extractor diagnostic; it is not a completed 200-document benchmark evaluation.

## Main result

On one fixed RAGTruth QA manifest, `support-critical` has the strongest held-out point estimates. All parameters and classification thresholds were selected using training data only.

| Method | Held-out ROC-AUC | Held-out F1 | Precision / recall |
|---|---:|---:|---:|
| `strict` | 0.755 [0.676, 0.835] | 0.721 [0.639, 0.793] | 0.639 / 0.827 |
| `support` | 0.730 [0.648, 0.816] | 0.695 [0.611, 0.773] | 0.640 / 0.760 |
| `support-critical` | **0.849 [0.784, 0.909]** | **0.798 [0.726, 0.862]** | **0.704 / 0.920** |

The held-out denominator is 147 responses: the deterministic manifest contains 750 responses (600 train, 150 held out), but three held-out answers have an empty answer graph and are unscorable under the shared graph policy. Source `12448` was quarantined before extraction, leaving 749 analysed sources; all completed without extraction failures.

These are promising point estimates from a single fixed manifest, not evidence of an independent replication or a general statistical superiority claim. The full, archive-derived report is [available here](docs/support-critical-750qa-results.tex).

## Method

For every RAGTruth tuple `(context C, query Q, answer A)`, the pipeline extracts three graphs:

```text
G_c = KGGen(C)       G_q = KGGen(Q)       G_a = KGGen(A)
G_ref = G_c ∪ G_q
```

It then measures how well entities and directed relations in the answer graph are grounded in `G_ref`. `H` is a hallucination-risk score: higher values mean greater risk.

| Mode | Scoring rule |
|---|---|
| `strict` | Historical HalluGraph-style directed graph alignment: entity grounding plus relation preservation. |
| `support` | Keeps entity grounding, but gives a relation credit only when a verifier finds textual entailment in the context/query. |
| `support-critical` | Adds atomic claims from the answer and claims found by full-context review. Each is assessed as `entailed`, `unknown`, `unsupported`, or `contradicted`; the worst `k` risks contribute to the final score. |

For `support-critical`, the training-only selected configuration was `alpha=1.0`, `beta=0.5`, `k=3`, and `lambda=0.0`. Thus the reported final score simplifies to:

```text
H_critical = 0.5 × (1 − EG) + 0.5 × H_top3
```

Relation evidence was still extracted and audited, but `alpha=1.0` means relation preservation had no direct numerical weight in this selected result. The experiment therefore does **not** show a direct benefit from relation scoring; it supports the implemented claim-verification branch together with entity grounding.

## Experimental safeguards

- The 750-response manifest, train/test split, five-fold training CV, thresholds, and runtime identity are recorded in the archive.
- `strict`, `support`, and `support-critical` use separate content-addressed cache namespaces. Cache keys include input content, model/runtime fingerprint, prompt/schema version, and extraction settings.
- Cache writes are atomic; compatible historical entries can be read through without mixing cache families.
- The successful R12 recovery reused the persistent cache and made zero live inference calls. Its replay preserved cache inventories and scientific outputs; only the diagnostic `verifier_cache_hit` flag is permitted to differ between cache population and replay.
- Unit tests use fakes and deterministic local embedders, so the test suite requires neither gateway access nor GPU inference.

## DocRED extraction diagnostic

DocRED evaluates the KG extraction component, not the RAGTruth hallucination detector. The completed time-limited pass used 50 `train_annotated` documents only to freeze the relation-alignment threshold, followed by a deterministic prefix of 66 public development documents with full extraction coverage.

| Metric on the 66-document held-out-development prefix | Result |
|---|---:|
| Typed DocRED triple recall | 15 / 816 = 1.84% |
| Gold-supported precision | 11.72% |
| Micro F1 | 3.18% |
| Endpoint-node F1 | 66.1% |
| Directed entity-pair F1 | 23.7% |

“Gold-supported precision” is deliberately not called factual precision: DocRED does not annotate every true fact in a document, so an unannotated extracted triple is not thereby false. The low exact typed-edge score reflects both limited directed-pair recovery and the mismatch between KGGen’s open predicates and DocRED’s closed relation inventory. A post-hoc, same-model semantic audit found conditional predicate agreement for 154 of 218 predictions whose directed entity pair already matched gold; it is diagnostic evidence, not an independent factual-precision estimate.

Read the [preliminary result](docs/docred-kg-time-limited-preliminary-results.tex), [topology diagnostic](docs/docred-kg-topology-technical-note.tex), and [semantic-predicate audit](docs/docred-kg-semantic-audit-technical-note.tex) for the protocol and limitations.

## Reproduce locally

The offline test and smoke paths are sufficient to check the implementation. A live run needs an authenticated gateway and must never put its API key in configuration, command arguments, logs, or archives.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

For an end-to-end offline plumbing check:

```bash
python tests/make_fixture.py tests/fixture_data
python run.py --stage all --fake-extractor \
  --data-dir tests/fixture_data --output-dir results_smoke
```

The active configuration is in [config.yaml](config.yaml): its sole logical model selector is `openai/gemini-2.5-flash`. Production runs derive the gateway URL, model revision, and runtime fingerprint from an authenticated gateway manifest. See the [Vertex/DataSphere runbook](docs/vertex-datasphere-team-runbook.md) for the archived RAGTruth workflow.

The DocRED launcher uses an external persistent root and Keychain-provided gateway access. Its safe, network-free prerequisite check is:

```bash
bash scripts/run_local_docred_kg_eval.sh --preflight
```

See [the DocRED evaluation hand-off](docs/task-prompts/docred-kg-extraction-evaluation.md) for its reproducibility contract and cache-only replay requirements.

## Repository map

```text
run.py                         RAGTruth orchestration and train/test evaluation
src/extract.py                 KGGen extraction, retries, structured output, caches
src/matching.py                Entity and directed-edge matching with local S-BERT
src/metrics.py                 Strict/support/support-critical aggregation
src/critical.py                Atomic claims, coverage review, evidence, verdicts
src/tune.py                    Train-only parameter and threshold selection
src/evaluate.py                Held-out metrics, bootstrap diagnostics, reports
src/docred.py                  DocRED entity/relation alignment and metrics
scripts/run_docred_kg_eval.py  Separate DocRED evaluator and cache-only replay
tests/                         Offline regression and protocol tests
docs/                          Result reports, diagnostics, and operational runbooks
```

## References

- Noël et al. — [HalluGraph](https://arxiv.org/abs/2512.01659)
- Mo et al. — [KGGen](https://arxiv.org/abs/2502.09956)
- Niu et al. — [RAGTruth](https://arxiv.org/abs/2401.00396)
- Tan et al. — [Re-DocRED](https://aclanthology.org/2022.acl-long.580/)
