# Copy this file to env.local.ps1, replace only the angle-bracket placeholders,
# then run: . .\env.local.ps1
# env.local.ps1 is ignored by Git. Never commit a secret value.

$env:HALLU_GATEWAY_URL = "https://<your-cloud-run-gateway-origin>"
$env:HALLU_GATEWAY_API_KEY = "<your-gateway-secret>"
$env:HALLU_TYPING_MODEL = "openai/gemini-2.5-flash"
$env:HALLU_HHEM_MODEL_PATH = (Join-Path $PSScriptRoot "local_resources\hhem-2.1-open")
