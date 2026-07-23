"""Create an offline interactive HTML explorer for one typing-run artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallugraph_dynamic_typing.viewer import load_viewer_payload, write_viewer_html


def _snapshot_from_fixture(path: Path, case_id: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("case_id") != case_id:
            continue
        graphs = row["graphs"]
        return {
            "schema_version": "run-input-v1",
            "case_id": case_id,
            "source": {
                "source_id": row["source_id"], "context_raw": row["context"], "query_raw": row.get("query", ""),
                "context_graph": {"graph_id": f"{case_id}:context", "role": "context", **graphs["context"]},
                "query_graph": {"graph_id": f"{case_id}:query", "role": "query", **graphs["query"]},
            },
            "answer": {
                "source_id": row["source_id"], "response_id": case_id, "response_raw": row["response"],
                "answer_graph": {"graph_id": f"{case_id}:answer", "role": "answer", **graphs["answer"]},
            },
        }
    raise ValueError(f"case_id not found in fixture: {case_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing summary.json")
    parser.add_argument("--case", help="Case ID; required only if the run has several successful cases")
    parser.add_argument("--fixture", help="No-gold JSONL fallback for a run created before input_snapshot.json existed")
    parser.add_argument("--output", help="Destination HTML path; defaults to the selected case directory")
    args = parser.parse_args(argv)
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    successful = [item for item in summary if item.get("status") == "ok"]
    case_id = args.case or (successful[0].get("case_id") if len(successful) == 1 else None)
    if not case_id:
        raise ValueError("select one successful case with --case")
    case_dir = run / str(case_id)
    snapshot_path = case_dir / "input_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else _snapshot_from_fixture(Path(args.fixture), str(case_id)) if args.fixture else None
    if snapshot is None:
        raise ValueError("input_snapshot.json is absent; provide --fixture with the no-gold JSONL used for this run")
    output = Path(args.output) if args.output else case_dir / "typing-run-viewer.html"
    payload = load_viewer_payload(case_dir, snapshot)
    write_viewer_html(output, payload)
    print(json.dumps({"case_id": case_id, "viewer": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
