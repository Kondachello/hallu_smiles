# DataSphere: controlled live shared-KGGen two-pass probe

This is one explicit RAGTruth response (default operational target: `response_id=6845`),
not a fake test and not a full dataset run. It is CPU-only (`c1.4`): real KGGen uses the
existing Cloud Run/Vertex gateway and GraphEval runs pinned local HHEM.

The DataSphere Project must inject exactly `HALLU_GATEWAY_API_KEY`. The rendered YAML
contains only `HALLU_GATEWAY_URL`; it contains no bearer value. The runner neither echoes
the secret nor retains the raw manifest response, and the Python entrypoint scans both
archives for the secret before success.

## Two passes

The Job creates a dedicated Project-storage `CACHE_ROOT` keyed by source commit and
response ID. First it invokes real `build_controlled_shared_kggen_detectors` in
`read_write`; HalluGraph extracts real context/query graphs and both methods receive the
same real response graph. Second it uses the same root in `cache_only`. KGExtractor then
raises on any miss and cannot construct KGGen; GraphEval has an injected `shared_kggen`
extractor, so it cannot call its gateway extraction backend. HHEM remains local and may
execute in the second pass.

The report requires both methods `ok`, one shared response hash across methods and passes,
materialize KGGen calls > 0, replay KGGen/gateway calls = 0, two valid sealed archives and
hidden gold. 429/5xx remain transport failures.

## Submit from PowerShell

Do this only after committing and pushing the changes, ensuring the CPU image for that
commit exists and configuring the Project secret. This command submits one job:

```powershell
$env:YC_AUTH="yc"; $env:PYTHON_BIN="python"; $env:PATH="$env:USERPROFILE\yandex-cloud\bin;$env:PATH"; & "C:\Program Files\Git\bin\bash.exe" -lc "cd /c/Users/Kolya/Desktop/SMILES/HaluVSGraph_Eval/hallu_smiles && source .venv-datasphere/Scripts/activate && bash scripts/submit_datasphere_shared_kggen_one_instance_controlled_live.sh --project-id bt1i64odluitglbaj5st --branch codex/experiment-framework-spec --run-id controlled-live-<date> --response-id 6845 --gateway-url https://<cloud-run-origin>"
```
