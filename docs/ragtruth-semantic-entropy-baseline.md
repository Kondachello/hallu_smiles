# RAGTruth QA SemanticEntropy baseline

This branch adds a deliberately non-graph baseline based on TOHA's
likelihood-weighted semantic entropy.  It does not extract a KG and does not
read the candidate response text while constructing its score.

For each selected RAGTruth source prompt, Gemini 2.5 Flash produces `N=15`
independent completions at temperature `1.0` and a fixed 8192-token cap. The Cloud
Run gateway returns the selected-token log probability for each generated
completion token.  We use the mean selected-token log probability, matching
TOHA's causal-LM loss reduction, as the sample log likelihood.  A local
`microsoft/deberta-v2-xlarge-mnli` model then greedily groups completions under
TOHA's mutual-entailment rule: neither direction can be contradiction, and the
two directions cannot both be neutral.  Semantic-class likelihood is the sum
of normalized member likelihoods; entropy of this class distribution is the
score.  Higher entropy means higher hallucination risk.

This is a prompt-level uncertainty baseline: a RAGTruth label annotates one
historical response, while entropy measures uncertainty of new Gemini samples
for the same source prompt.  It is therefore reported separately from the
graph scores and is not described as an audit of the historical response.

The scale-up recreates the verified historical 750-row QA manifest (SHA-256
`19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86`).
It then applies the established source-level quarantine for `12448`, yielding
599 train and 150 held-out test rows.  The manifest is balanced before that
historical exclusion.  The decision threshold is selected by F1 on train only;
held-out test metrics are computed once with the frozen threshold.

All downloads, NLI weights, sample caches, checkpoints, logs, snapshots and
archives live under `/Volumes/mySSD/hallu_smiles/semantic-entropy` by default.
The repository stores no prompts, completions, bearer credentials, or NLI
weights.  A completed live stage is followed by a cache-only recomputation;
the public `scores.jsonl` files must be byte-identical.

Before a live run, deploy the updated gateway contract and fetch its
authenticated manifest.  The runner reads `HALLU_GATEWAY_API_KEY` only from
macOS Keychain and starts with a 10-QA measurement.  It then selects the
largest multiple-of-10, balanced 80/20 manifest no larger than 750 that fits a
conservative remaining 24-hour wall-clock estimate:

```bash
bash scripts/start_local_ragtruth_semantic_entropy.sh
```

The launcher is serial (four-second admission pacing), cache-resumable, and
records redacted 15-minute monitor snapshots.  A 429 or temporary gateway
failure first uses the existing bounded retry policy, then performs up to two
compatible cache-resumes without changing the manifest or score protocol.
