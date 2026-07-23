"""Build a local dashboard and interactive case pages for a typing test run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallugraph_dynamic_typing.viewer import write_viewer_site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="run directory containing run_manifest.json, summary.json and case folders",
    )
    parser.add_argument(
        "--output",
        help="viewer directory; defaults to <run>/viewer",
    )
    args = parser.parse_args(argv)
    run = Path(args.run)
    index = write_viewer_site(run, args.output)
    print(
        json.dumps(
            {
                "run": str(run),
                "viewer": str(index),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
