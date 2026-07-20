#!/usr/bin/env bash
# Work payload for the GraphEval DataSphere mock job (offline; fake backends only,
# no gateway/HHEM/secret).  RUN_ROOT is exported by the job cmd; defaults for a
# local run.  Runs the same probe that was validated locally.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${RUN_ROOT:=$ROOT/outputs/graph-eval-mock/local}"
mkdir -p "$RUN_ROOT"
python3 scripts/graph_eval_datasphere_mock.py --out "$RUN_ROOT"
echo "[graph-eval-mock] summary:"
cat "$RUN_ROOT/summary.json"
