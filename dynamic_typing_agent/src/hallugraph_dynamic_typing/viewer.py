"""Offline multi-run viewer for dynamic-typing artifacts."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping


ASSET_PACKAGE = "hallugraph_dynamic_typing.viewer_assets"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_cases(run_dir: Path) -> list[dict[str, Any]]:
    manifest = _read_json(run_dir / "run_manifest.json")
    if isinstance(manifest, dict) and isinstance(manifest.get("cases"), list):
        return [dict(item) for item in manifest["cases"]]
    summary = _read_json(run_dir / "summary.json", [])
    if isinstance(summary, dict):
        summary = summary.get("cases", [])
    if not isinstance(summary, list):
        raise ValueError("run summary must be a list or an object with a cases list")
    return [dict(item) for item in summary]


def _case_dir(run_dir: Path, case: Mapping[str, Any]) -> Path:
    relative = str(case.get("artifact_dir") or case["case_id"])
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return run_dir / candidate


def load_viewer_payload(
    case_dir: str | Path, snapshot: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Combine one case's immutable artifacts into a browser-oriented payload."""
    root = Path(case_dir)
    source_run = _read_json(root / "source_registry.json", {})
    answer_run = _read_json(root / "answer_annotations.json", {})
    snapshot_data = (
        dict(snapshot)
        if snapshot is not None
        else _read_json(root / "input_snapshot.json", {})
    )
    registry = source_run.get("registry") if isinstance(source_run, dict) else None
    annotations = (
        answer_run.get("annotations") if isinstance(answer_run, dict) else None
    )
    trace = _read_json(
        root / "execution_trace.json",
        {
            "schema_version": "execution-trace-v1",
            "input_events": [],
            "source_events": source_run.get("artifacts", [])
            if isinstance(source_run, dict)
            else [],
            "answer_events": answer_run.get("artifacts", [])
            if isinstance(answer_run, dict)
            else [],
        },
    )
    failure = _read_json(root / "failure.json")
    source = snapshot_data.get("source")
    if registry is not None and not isinstance(source, dict):
        raise ValueError("viewer input snapshot must contain a source object")
    return {
        "schema_version": "typing-viewer-case-v2",
        "case_id": str(snapshot_data.get("case_id", root.name)),
        "input_mode": snapshot_data.get("input_mode", "legacy"),
        "graph_provenance": snapshot_data.get("graph_provenance", {}),
        "metadata": snapshot_data.get("metadata", {}),
        "source": source,
        "answer": snapshot_data.get("answer"),
        "registry": registry,
        "annotations": annotations,
        "source_status": source_run.get("status")
        if isinstance(source_run, dict)
        else "failed",
        "answer_status": answer_run.get("status")
        if isinstance(answer_run, dict)
        else None,
        "manifest": _read_json(root / "manifest.json", {}),
        "trace": trace,
        "failure": failure,
    }


def load_run_payload(run_dir: str | Path) -> dict[str, Any]:
    """Load the canonical run manifest or adapt an older summary."""
    root = Path(run_dir)
    manifest = _read_json(root / "run_manifest.json")
    cases = _summary_cases(root)
    if not isinstance(manifest, dict):
        counts: dict[str, int] = {}
        for item in cases:
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        manifest = {
            "schema_version": "typing-test-run-legacy",
            "run_id": root.name,
            "case_count": len(cases),
            "status_counts": counts,
            "input": {"requested_mode": "legacy", "no_gold": True},
            "cases": cases,
        }
    else:
        manifest = {**manifest, "cases": cases}
    return manifest


def _safe_script_assignment(variable: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"window.{variable}={encoded};\n"


def _asset_text(name: str) -> str:
    return files(ASSET_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def write_viewer_site(
    run_dir: str | Path, destination: str | Path | None = None
) -> Path:
    """Generate a file:// compatible dashboard and all successful case pages."""
    root = Path(run_dir)
    target = Path(destination) if destination is not None else root / "viewer"
    assets = target / "assets"
    cases_root = target / "cases"
    assets.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)

    for name in ("styles.css", "dashboard.js", "case.js"):
        assets.joinpath(name).write_text(_asset_text(name), encoding="utf-8")
    target.joinpath("index.html").write_text(
        _asset_text("dashboard.html"), encoding="utf-8"
    )

    run_payload = load_run_payload(root)
    dashboard_cases: list[dict[str, Any]] = []
    for case in run_payload["cases"]:
        case_id = str(case["case_id"])
        case_target = cases_root / case_id
        case_target.mkdir(parents=True, exist_ok=True)
        artifact_dir = _case_dir(root, case)
        payload = load_viewer_payload(artifact_dir)
        case_target.joinpath("data.js").write_text(
            _safe_script_assignment("__TYPING_CASE__", payload), encoding="utf-8"
        )
        case_target.joinpath("index.html").write_text(
            _asset_text("case.html"), encoding="utf-8"
        )
        dashboard_cases.append(
            {
                **case,
                "viewer_path": f"cases/{case_id}/index.html",
            }
        )
    run_payload = {**run_payload, "cases": dashboard_cases}
    target.joinpath("run-data.js").write_text(
        _safe_script_assignment("__TYPING_RUN__", run_payload), encoding="utf-8"
    )
    return target / "index.html"


def write_viewer_html(destination: str | Path, payload: Mapping[str, Any]) -> Path:
    """Compatibility helper: write one fully self-contained case document."""
    html = _asset_text("case.html")
    html = html.replace(
        '<link rel="stylesheet" href="../../assets/styles.css">',
        f"<style>{_asset_text('styles.css')}</style>",
    )
    html = html.replace(
        '<script src="data.js"></script>',
        f"<script>{_safe_script_assignment('__TYPING_CASE__', payload)}</script>",
    )
    html = html.replace(
        '<script src="../../assets/case.js"></script>',
        f"<script>{_asset_text('case.js')}</script>",
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
