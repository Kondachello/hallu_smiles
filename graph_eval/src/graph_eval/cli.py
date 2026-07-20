"""Standalone GraphEval CLI: no-gold JSONL in, predictions JSONL out.

    python -m graph_eval.cli predict --config config.yaml \
        --input instances.no_gold.jsonl --output predictions.jsonl [--resume] [--limit N]

Reads only ``response_id, source_id, context, response, query?, metadata?`` from each
input line — any gold-like field present is ignored (never passed to the detector).
``--resume`` skips response_ids already present in the output and appends, so a run
interrupted mid-way continues cheaply (the extraction/NLI caches do the rest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import prediction_record
from .config import GraphEvalConfig, from_dict
from .detector import GraphEvalDetector
from .factory import build_extractor, build_nli
from .types import DetectionInput


def _load_config(path: str | None) -> GraphEvalConfig:
    if not path:
        return GraphEvalConfig()  # defaults => fake backends, fully offline
    import yaml  # optional dep; only needed when a config file is given

    with open(path, encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh) or {})


def _read_instances(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _done_ids(path: str) -> set[str]:
    ids: set[str] = set()
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(str(json.loads(line)["response_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def _to_input(rec: dict) -> DetectionInput:
    return DetectionInput(
        response_id=str(rec["response_id"]),
        source_id=str(rec["source_id"]),
        context=rec["context"],
        response=rec["response"],
        query=rec.get("query"),
        metadata=rec.get("metadata", {}),
    )


def cmd_predict(args) -> int:
    cfg = _load_config(args.config)
    detector = GraphEvalDetector(
        build_extractor(cfg, manifest_sha256=args.manifest_sha256),
        build_nli(cfg),
        paper_threshold=cfg.nli.paper_threshold,
        aggregation=cfg.nli.aggregation,
    )
    done = _done_ids(args.output) if args.resume else set()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    read = written = skipped = 0
    with out.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for rec in _read_instances(args.input):
            read += 1
            if str(rec.get("response_id")) in done:
                skipped += 1
                continue
            item = _to_input(rec)
            result = detector.predict(item)
            fh.write(json.dumps(prediction_record(result, item), ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            written += 1
            if args.limit and written >= args.limit:
                break
    print(json.dumps({"read": read, "written": written, "skipped": skipped, "output": str(out)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="run GraphEval on a no-gold JSONL")
    predict.add_argument("--config", default=None, help="YAML config (default: fake backends)")
    predict.add_argument("--input", required=True, help="no-gold instances JSONL")
    predict.add_argument("--output", required=True, help="predictions JSONL")
    predict.add_argument("--manifest-sha256", default=None, help="authenticated gateway manifest hash")
    predict.add_argument("--limit", type=int, default=None)
    predict.add_argument("--resume", action="store_true", help="skip response_ids already in --output")
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
