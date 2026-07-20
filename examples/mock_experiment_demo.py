"""Run the framework's pretty offline demonstration without RAGTruth, secrets or network.

    python examples/mock_experiment_demo.py --output-root examples/mock_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.demo import run_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline mock GraphEval × HalluGraph archive demo")
    parser.add_argument("--output-root", default=str(ROOT / "examples" / "mock_output"))
    parser.add_argument("--run-id", default="mock-demo")
    args = parser.parse_args()
    run_demo(args.output_root, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
