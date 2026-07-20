"""Offline-safe command line interface for the first experiment-framework vertical slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import RunArchive, atomic_write_jsonl, read_jsonl
from .datasets.ragtruth import (
    audit_dataset,
    create_source_sample_manifest,
    fetch_dataset,
    materialize_subset,
    write_data_manifest,
    write_sample_manifest,
)
from .demo import run_demo
from .detectors import build_grapheval_fake, build_hallugraph_fake
from .evaluation import evaluate_joined_predictions, join_gold
from .runner import run_paired, seal_run


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_data_audit(args: argparse.Namespace) -> int:
    manifest = audit_dataset(args.source_info, args.responses, revision=args.revision)
    write_data_manifest(args.output, manifest)
    print(json.dumps({"output": args.output, "dataset_manifest_sha256": manifest["dataset_manifest_sha256"]}, ensure_ascii=False))
    return 0


def cmd_data_fetch(args: argparse.Namespace) -> int:
    manifest = fetch_dataset(data_root=args.data_root, revision=args.revision, raw_base=args.raw_base)
    print(json.dumps({"dataset_root": str(Path(args.data_root) / "raw" / args.revision), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"]}, ensure_ascii=False))
    return 0


def cmd_sample_create(args: argparse.Namespace) -> int:
    dataset_manifest = _load_json(args.data_manifest)
    manifest = create_source_sample_manifest(
        args.source_info,
        args.responses,
        dataset_manifest=dataset_manifest,
        split=args.split,
        seed=args.seed,
        n_sources=args.n_sources,
        tasks=args.task,
        models=args.model,
        include_all_responses_per_source=not args.one_response_per_source,
        purpose=args.purpose,
    )
    write_sample_manifest(args.output, manifest)
    print(json.dumps({"output": args.output, "sample_manifest_sha256": manifest["sample_manifest_sha256"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


def cmd_sample_materialize(args: argparse.Namespace) -> int:
    paths = materialize_subset(
        args.source_info,
        args.responses,
        dataset_manifest=_load_json(args.data_manifest),
        sample_manifest=_load_json(args.sample_manifest),
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    summary = run_demo(args.output_root, run_id=args.run_id)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_run_fake(args: argparse.Namespace) -> int:
    """Explicit fake-only execution path; no credentials or live models are used."""
    instances = read_jsonl(args.instances)
    archive = RunArchive.create(
        args.runs_root,
        run_id=args.run_id,
        manifest={"run_purpose": "offline_fake", "comparison_track": "exploratory", "network_access": False},
    )
    archived_instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(archived_instances, instances)
    detectors = {
        "hallugraph": build_hallugraph_fake(args.hallugraph_config),
        "grapheval": build_grapheval_fake(),
    }
    summary = run_paired(archive, instances_path=archived_instances, detectors=detectors)
    seal_run(archive, archived_instances)
    print(json.dumps({**summary, "archive": str(archive.path)}, ensure_ascii=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    archive = RunArchive(Path(args.runs_root), args.run_id)
    join_gold(archive, response_gold_path=args.response_gold)
    thresholds = {"hallugraph": args.hallugraph_threshold, "grapheval": args.grapheval_threshold}
    metrics = evaluate_joined_predictions(archive, thresholds=thresholds)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


def cmd_archive_validate(args: argparse.Namespace) -> int:
    archive = RunArchive(Path(args.runs_root), args.run_id)
    result = archive.validate()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments", description="Offline-safe GraphEval × HalluGraph experiment framework")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="RAGTruth data audit")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="explicitly download official RAGTruth files at a pinned commit")
    fetch.add_argument("--data-root", default="data/ragtruth")
    fetch.add_argument("--revision", required=True, help="exact 40-character Git commit SHA; floating branches are rejected")
    fetch.add_argument("--raw-base", default="https://raw.githubusercontent.com/ParticleMedia/RAGTruth")
    fetch.set_defaults(func=cmd_data_fetch)
    audit = data_sub.add_parser("audit", help="audit already available local JSONL files; never downloads")
    audit.add_argument("--source-info", required=True)
    audit.add_argument("--responses", required=True)
    audit.add_argument("--revision", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=cmd_data_audit)

    sample = sub.add_parser("sample", help="create or materialize deterministic no-gold samples")
    sample_sub = sample.add_subparsers(dest="sample_command", required=True)
    create = sample_sub.add_parser("create")
    create.add_argument("--source-info", required=True)
    create.add_argument("--responses", required=True)
    create.add_argument("--data-manifest", required=True)
    create.add_argument("--split", required=True, choices=("train", "test"))
    create.add_argument("--seed", required=True, type=int)
    create.add_argument("--n-sources", type=int)
    create.add_argument("--task", action="append", default=[])
    create.add_argument("--model", action="append", default=[])
    create.add_argument("--one-response-per-source", action="store_true")
    create.add_argument("--purpose", default="development")
    create.add_argument("--output", required=True)
    create.set_defaults(func=cmd_sample_create)
    materialize = sample_sub.add_parser("materialize")
    materialize.add_argument("--source-info", required=True)
    materialize.add_argument("--responses", required=True)
    materialize.add_argument("--data-manifest", required=True)
    materialize.add_argument("--sample-manifest", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.set_defaults(func=cmd_sample_materialize)

    demo = sub.add_parser("demo", help="run only deterministic mock detectors; no data download or credentials")
    demo.add_argument("--output-root", default="examples/mock_output")
    demo.add_argument("--run-id", default="mock-demo")
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="explicit execution modes")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    fake = run_sub.add_parser("fake", help="run existing detectors only with deterministic fake backends")
    fake.add_argument("--instances", required=True, help="materialized instances.no_gold.jsonl")
    fake.add_argument("--runs-root", default="runs")
    fake.add_argument("--run-id", required=True)
    fake.add_argument("--hallugraph-config", default="config.yaml")
    fake.set_defaults(func=cmd_run_fake)

    evaluate = sub.add_parser("evaluate", help="post-seal evaluation; never invokes a detector")
    evaluate.add_argument("--runs-root", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--response-gold", required=True)
    evaluate.add_argument("--hallugraph-threshold", type=float, required=True)
    evaluate.add_argument("--grapheval-threshold", type=float, required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    archive = sub.add_parser("archive", help="archive validation")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    validate = archive_sub.add_parser("validate")
    validate.add_argument("--runs-root", required=True)
    validate.add_argument("--run-id", required=True)
    validate.set_defaults(func=cmd_archive_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
