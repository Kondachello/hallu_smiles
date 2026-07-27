"""Local-only CLI for no-gold fixtures and future standalone inputs."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .agent import DynamicTypingAgent, graph_from_fixture
from .models import AnswerInput, SourceInput
from .persistence import ArtifactWriter
from .prompt_registry import PromptRegistry
from .kggen_pipeline import ragtruth_records, run_pipeline, text_records
from .test_framework import run_test_suite


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agent(args: argparse.Namespace) -> DynamicTypingAgent:
    if args.config:
        return DynamicTypingAgent.from_yaml(
            args.config,
            cache_root=args.cache_root,
            artifacts_root=getattr(args, "run_output", None) or args.output,
        )
    return DynamicTypingAgent(
        prompt_root=args.prompt_root,
        cache_root=args.cache_root or ".cache/dynamic-typing-agent",
        artifacts_root=getattr(args, "run_output", None) or args.output or "runs",
        backend="fake",
    )


def run_fixture(args: argparse.Namespace) -> int:
    agent = _agent(args)
    output = Path(getattr(args, "run_output", None) or args.output or agent.artifacts_root)
    summaries: list[dict] = []
    for row in _rows(Path(args.input))[: args.limit]:
        graphs = row["graphs"]
        source = SourceInput(
            source_id=row["source_id"],
            context_raw=row["context"],
            query_raw=row.get("query", ""),
            context_graph=graph_from_fixture(graph_id=f"{row['case_id']}:context", role="context", payload=graphs["context"]),
            query_graph=graph_from_fixture(graph_id=f"{row['case_id']}:query", role="query", payload=graphs["query"]),
        )
        source_run = agent.build_source_registry(source)
        if source_run.registry is None:
            summaries.append({"case_id": row["case_id"], "status": "failed", "failure": source_run.failure})
            continue
        answer = AnswerInput(
            source_id=row["source_id"],
            response_id=row["case_id"],
            response_raw=row["response"],
            answer_graph=graph_from_fixture(graph_id=f"{row['case_id']}:answer", role="answer", payload=graphs["answer"]),
            registry=source_run.registry,
        )
        answer_run = agent.annotate_answer(answer)
        path = agent.write_run_artifacts(run_id=row["case_id"], source_run=source_run, answer_run=answer_run)
        # The local viewer needs raw no-gold inputs as well as derived annotations.  Keep
        # the answer registry out of this snapshot because it is already immutable in
        # source_registry.json and must never be reconstructed from an answer artifact.
        ArtifactWriter(path).write_json(
            "input_snapshot.json",
            {
                "schema_version": "run-input-v1",
                "case_id": row["case_id"],
                "source": source.model_dump(mode="json"),
                "answer": answer.model_dump(mode="json", exclude={"registry"}),
            },
        )
        summaries.append({"case_id": row["case_id"], "status": answer_run.status.value, "artifact_dir": str(path)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(item["status"] != "ok" for item in summaries)
    print(json.dumps({"cases": len(summaries), "succeeded": len(summaries) - failed, "failed": failed, "output": str(output)}, ensure_ascii=False))
    return 1 if failed else 0


def validate(_: argparse.Namespace) -> int:
    registry = PromptRegistry()
    print(json.dumps({"prompt_manifest_sha256": registry.manifest_sha256, "prompt_count": len(registry.entries)}))
    return 0


def run_kggen_pipeline(args: argparse.Namespace) -> int:
    agent = _agent(args)
    output = Path(getattr(args, "run_output", None) or args.output or agent.artifacts_root)
    if args.ragtruth_source_info:
        records = ragtruth_records(args.ragtruth_source_info, args.ragtruth_response, limit=args.limit)
    else:
        records = text_records(args.input)[: args.limit]
    summary = run_pipeline(agent=agent, records=records, output=output, kggen_config=args.kggen_config, fake_kggen=args.fake_kggen, kggen_cache_root=args.kggen_cache_root)
    failed = sum(item["status"] != "ok" for item in summary)
    print(json.dumps({"cases": len(summary), "succeeded": len(summary) - failed, "failed": failed, "output": str(output)}, ensure_ascii=False))
    return 1 if failed else 0


def run_unified_test(args: argparse.Namespace) -> int:
    """Run text or graph fixtures into the same artifact and viewer layout."""
    agent = _agent(args)
    manifest = run_test_suite(
        agent=agent,
        input_path=args.input,
        output=args.run_output,
        limit=args.limit,
        input_mode=args.input_mode,
        kggen_mode=args.kggen,
        kggen_config=args.kggen_config,
        kggen_cache_root=args.kggen_cache_root,
        case_ids=args.case_id,
        render_viewer=not args.no_viewer,
    )
    failed = int(manifest["status_counts"].get("failed", 0))
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "cases": manifest["case_count"],
                "failed": failed,
                "output": str(args.run_output),
                "viewer": manifest.get("viewer"),
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hallugraph-type-agent")
    parser.add_argument("--config", help="YAML runtime configuration; live secrets are read only from environment")
    parser.add_argument("--prompt-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--output")
    commands = parser.add_subparsers(dest="command", required=True)
    unified = commands.add_parser(
        "test",
        help="run raw text, supplied graphs or a mixed no-gold JSONL into one browsable suite",
    )
    unified.add_argument("--input", required=True, help="no-gold JSONL test cases")
    unified.add_argument(
        "--input-mode",
        choices=("auto", "graphs", "text"),
        default="auto",
        help="auto detects a graphs object per case; text forces KGGen",
    )
    unified.add_argument(
        "--kggen",
        choices=("fake", "live"),
        default="fake",
        help="KGGen backend for text cases; ignored for supplied graphs",
    )
    unified.add_argument(
        "--kggen-config",
        help="HalluGraph KGGen YAML configuration; required for --kggen live",
    )
    unified.add_argument(
        "--kggen-cache-root",
        default=".cache/kggen-graphs",
        help="persistent immutable KGGen graph cache",
    )
    unified.add_argument("--limit", type=int, default=20)
    unified.add_argument(
        "--case-id",
        action="append",
        help="run only this case_id; repeat to select several immutable input rows",
    )
    unified.add_argument(
        "--output",
        dest="run_output",
        required=True,
        help="single output root for artifacts and the local viewer",
    )
    unified.add_argument(
        "--no-viewer",
        action="store_true",
        help="skip HTML generation while retaining the same artifact contract",
    )
    unified.set_defaults(handler=run_unified_test)
    run = commands.add_parser("run-fixture", help="run no-gold JSONL using the selected fake or live configuration")
    run.add_argument("--input", required=True)
    run.add_argument("--limit", type=int, default=20)
    run.add_argument("--output", dest="run_output", help="artifact directory; may be supplied after the command")
    run.set_defaults(handler=run_fixture)
    pipeline = commands.add_parser("run-kggen", help="extract KGGen graphs from text/RAGTruth, then run dynamic typing")
    pipeline.add_argument("--input", help="JSONL with no-gold context, query and optional response")
    pipeline.add_argument("--ragtruth-source-info", help="local RAGTruth source_info.jsonl")
    pipeline.add_argument("--ragtruth-response", help="local RAGTruth response.jsonl")
    pipeline.add_argument("--kggen-config", help="HalluGraph KGGen YAML config; required unless --fake-kggen")
    pipeline.add_argument("--kggen-cache-root", default=".cache/kggen-graphs", help="persistent immutable KGGen graph cache")
    pipeline.add_argument("--fake-kggen", action="store_true", help="offline KGGen-shaped extractor for plumbing tests only")
    pipeline.add_argument("--limit", type=int, default=20)
    pipeline.add_argument("--output", dest="run_output", help="artifact directory; may be supplied after the command")
    pipeline.set_defaults(handler=run_kggen_pipeline)
    verify = commands.add_parser("validate", help="validate prompt assets")
    verify.set_defaults(handler=validate)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
