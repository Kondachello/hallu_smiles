#!/usr/bin/env bash
# Offline paired HalluGraph × GraphEval framework smoke test for DataSphere.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/hallu/client/bin/python}"
: "${RUN_ROOT:=$ROOT/outputs/experiment-framework-mock/local}"

test -x "$CLIENT_PYTHON" || { echo "client Python is missing: $CLIENT_PYTHON" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
cd "$ROOT"
export PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CLIENT_PYTHON" scripts/experiment_framework_datasphere_mock.py \
  --out "$RUN_ROOT/framework" --config "$ROOT/config.yaml"
"$CLIENT_PYTHON" - "$RUN_ROOT/framework/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("passed") is not True:
    raise SystemExit("framework mock summary did not pass")
PY
echo "[experiment-framework-mock] summary:"
cat "$RUN_ROOT/framework/summary.json"
