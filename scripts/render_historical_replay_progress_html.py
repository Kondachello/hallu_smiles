#!/usr/bin/env python3
"""Render a human-readable HTML view of a historical-qa-cache-replay run.

Local, read-only tooling: reads whatever has already been downloaded from a
DataSphere Job (complete or still in progress) and produces one static HTML
file. It never talks to a detector and never changes pipeline output --
purely a presentation layer over ``reports/progress.jsonl`` (+
``predictions/raw_predictions.jsonl`` and ``reports/historical_cache_discovery.json``
when present), so it is safe to re-run at any point, including against a
partial download of a still-EXECUTING Job.

Usage:
    python scripts/render_historical_replay_progress_html.py \\
        --run-dir outputs/datasphere-results/<RUN_ID>/<RUN_ID>/historical-cache-replay \\
        --output outputs/datasphere-results/<RUN_ID>/progress.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def build_html(run_dir: Path) -> str:
    progress = _read_jsonl(run_dir / "reports" / "progress.jsonl")
    predictions = _read_jsonl(run_dir / "predictions" / "raw_predictions.jsonl")
    discovery = _read_json(run_dir / "reports" / "historical_cache_discovery.json")
    seal = _read_json(run_dir / "prediction_seal.json")

    total_expected = 0
    for row in progress:
        if row.get("event") == "selection_complete":
            total_expected = row.get("requested_count", 0)
            break

    finished_ids = {row["response_id"] for row in progress if row.get("event") == "record_finished"}
    started_ids = {row["response_id"] for row in progress if row.get("event") == "record_started"}
    in_flight = sorted(started_ids - finished_ids)

    # response_id -> {method -> {score, status, ms}}
    by_response: dict[str, dict[str, dict[str, Any]]] = {}
    for p in predictions:
        rid = str(p.get("response_id"))
        by_response.setdefault(rid, {})[p.get("method", "?")] = {
            "score": p.get("raw_score"),
            "status": p.get("status"),
            "shared_graph_source": p.get("shared_graph_source"),
        }
    for row in progress:
        if row.get("event") == "detector_finished":
            rid = str(row["response_id"])
            entry = by_response.setdefault(rid, {}).setdefault(row["method"], {})
            entry.setdefault("score", row.get("score"))
            entry.setdefault("status", row.get("status"))
            entry["wall_time_ms"] = row.get("wall_time_ms")

    rows_html = []
    for rid in sorted(by_response, key=lambda x: (x not in finished_ids, x)):
        methods = by_response[rid]
        state = "done" if rid in finished_ids else ("running" if rid in started_ids else "?")
        cells = []
        for method in ("hallugraph", "grapheval"):
            m = methods.get(method)
            if not m:
                cells.append('<td class="pending">&mdash;</td>')
                continue
            score = m.get("score")
            score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else "&mdash;"
            status = m.get("status", "?")
            css = "ok" if status == "ok" else "bad"
            cells.append(f'<td class="{css}">{_esc(score_txt)} <span class="status">({_esc(status)})</span></td>')
        rows_html.append(
            f'<tr class="{state}"><td>{_esc(rid)}</td><td class="state-{state}">{_esc(state)}</td>'
            + "".join(cells) + "</tr>"
        )

    discovery_html = ""
    if discovery:
        cand_rows = "".join(
            f'<tr><td>{_esc(c["path"])}</td><td>{_esc(c["total"])}</td>'
            f'<td>{_esc(c["train"])}</td><td>{_esc(c["test"])}</td>'
            f'<td>{_esc(c["cv_folds"])}</td><td class="{"ok" if c["status"] == "valid" else "bad"}">{_esc(c["status"])}</td></tr>'
            for c in discovery.get("candidates", [])
        )
        discovery_html = f"""
        <h2>Historical checkpoint discovery</h2>
        <p>gateway_manifest_sha256: <code>{_esc(discovery.get("gateway_manifest_sha256"))}</code>
           &middot; requested qa_sample_size: <b>{_esc(discovery.get("requested_qa_sample_size"))}</b>
           &middot; valid matches: <b>{_esc(discovery.get("valid_count"))}</b></p>
        <table>
          <tr><th>path</th><th>total</th><th>train</th><th>test</th><th>cv_folds</th><th>status</th></tr>
          {cand_rows}
        </table>
        """

    seal_html = ""
    if seal:
        seal_html = f"<p>prediction_seal: <code>{_esc(json.dumps(seal, sort_keys=True))[:300]}</code></p>"

    progress_pct = round(100 * len(finished_ids) / total_expected, 1) if total_expected else 0.0

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>historical-qa-cache-replay progress</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 14px; }}
  th {{ background: #f4f4f4; }}
  tr.running {{ background: #fff8e1; }}
  .ok {{ color: #1a7f37; }}
  .bad {{ color: #cf222e; font-weight: bold; }}
  .pending {{ color: #999; }}
  .status {{ color: #888; font-size: 12px; }}
  .state-done {{ color: #1a7f37; }}
  .state-running {{ color: #b08800; }}
  .bar {{ background: #eee; border-radius: 4px; height: 18px; width: 100%; max-width: 400px; overflow: hidden; }}
  .bar-fill {{ background: #1a7f37; height: 100%; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
</style></head>
<body>
<h1>historical-qa-cache-replay &mdash; progress</h1>
<p><b>{len(finished_ids)}</b> / <b>{total_expected or "?"}</b> records finished
   ({progress_pct}%){f", {len(in_flight)} in flight: {', '.join(_esc(x) for x in in_flight)}" if in_flight else ""}.</p>
<div class="bar"><div class="bar-fill" style="width:{progress_pct}%"></div></div>
{discovery_html}
<h2>Per-response scores (higher = more likely hallucinated)</h2>
<table>
  <tr><th>response_id</th><th>state</th><th>HalluGraph</th><th>GraphEval</th></tr>
  {"".join(rows_html) if rows_html else '<tr><td colspan="4">no records yet</td></tr>'}
</table>
{seal_html}
<p style="color:#888;font-size:12px;">Generated locally from progress.jsonl / raw_predictions.jsonl.
Re-run this script after re-downloading to refresh, including on a still-EXECUTING Job.</p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="path to the extracted .../historical-cache-replay directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(Path(args.run_dir)), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
