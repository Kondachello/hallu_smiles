# DataSphere: shared-KGGen two-pass one-instance mock

This is a CPU-only, offline framework check for one explicit RAGTruth response.
It uses `FakeKGGen` and `FakeNLI`; it does not call the Vertex gateway, use a
Project secret, download a model, or produce a scientific detector-quality result.

## Two passes and Project storage

The rendered Job gives the probe a new dedicated Project-storage namespace:

```text
$DS_PROJECT_HOME/hallu_smiles/checkpoints/shared-kggen-mock/
  controlled-shared-kggen-mock-v1/<source-commit>/<response-id>/
```

The first pass (`read_write`) materializes response/context/query graphs into this
namespace. The second pass (`cache_only`) uses exactly the same namespace and fails
on any cache miss. Thus the first pass is intentionally cold and the second is
intentionally cache-backed; neither pass mutates historical 100-QA checkpoints.

The payload also lists directories named `kg` under the historical base
`$DS_PROJECT_HOME/hallu_smiles/checkpoints/vertex-qa/qa-100-test-20-cv-5` into
`historical-kg-cache-candidates.json`. This is read-only discovery only. The 100-QA
report proves that prior cache/replay existed but does not give a portable root or
prove compatibility with the controlled fake track. Do not label a discovered
directory as reusable until `python -m experiments.cli cache inspect` validates its
envelopes and exact no-gold cache keys.

## Submission

From PowerShell in the repository root, use the wrapper through Git Bash:

```powershell
$env:YC_AUTH="yc"; $env:PYTHON_BIN="python"; $env:PATH="$env:USERPROFILE\yandex-cloud\bin;$env:PATH"; & "C:\Program Files\Git\bin\bash.exe" -lc "cd /c/Users/Kolya/Desktop/SMILES/HaluVSGraph_Eval/hallu_smiles && source .venv-datasphere/Scripts/activate && bash scripts/submit_datasphere_shared_kggen_one_instance_mock.sh --project-id bt1i64odluitglbaj5st --branch codex/experiment-framework-spec --run-id shared-kggen-mock-<date> --response-id 6845"
```

The submitter requires the source commit to be pushed and resolves the matching
immutable GHCR CPU image before it creates the Job. The Job is `c1.4` only.

## Success criteria

The archive must contain two sealed, valid archives and a two-pass report proving:

- both detectors have `status=ok` in each pass;
- all prediction records share one `shared_graph_sha256`;
- materialize `KGGen` calls are greater than zero;
- cache replay `KGGen` calls are exactly zero;
- no gold fields appear in detector inputs;
- `historical-kg-cache-candidates.json` is descriptive only and contains no secret.
