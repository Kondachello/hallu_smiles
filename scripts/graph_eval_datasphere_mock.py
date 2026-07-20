#!/usr/bin/env python3
"""Offline DataSphere mock-probe for GraphEval.

No gateway, no HHEM, no secret, no RAGTruth: it runs GraphEval end-to-end on the
deterministic fake backends and then a cache-only replay, asserting the contract
invariants.  This is the exact payload a DataSphere mock Job executes -- it proves
the GraphEval code imports and runs in the target environment without any live
backend.  Exits non-zero if any check fails; writes summary.json + predictions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graph_eval" / "src"))

from graph_eval import DetectionInput, GraphEvalDetector, from_dict  # noqa: E402
from graph_eval.artifacts import prediction_record  # noqa: E402
from graph_eval.factory import build_extractor, build_nli  # noqa: E402

INSTANCES = [
    {"response_id": "m1", "source_id": "s1", "query": "capital of France?",
     "context": "Paris is the capital of France.",
     "response": "Paris is the capital of France."},          # grounded -> ok
    {"response_id": "m2", "source_id": "s2", "query": "where is the tower?",
     "context": "The Eiffel Tower is in Paris.",
     "response": "The tower stands in Berlin."},               # ungrounded -> ok
    {"response_id": "m3", "source_id": "s3", "query": "boiling point?",
     "context": "Water boils at 100 degrees Celsius.",
     "response": "hi"},                                        # no triples -> empty_graph
]


def _detector(cache_dir: str, cache_only: bool) -> GraphEvalDetector:
    cfg = from_dict({
        "extractor": {"backend": "fake"},
        "nli": {"backend": "fake"},
        "cache_dir": cache_dir,
        "cache_only": cache_only,
    })
    return GraphEvalDetector(
        build_extractor(cfg), build_nli(cfg),
        paper_threshold=cfg.nli.paper_threshold, aggregation=cfg.nli.aggregation,
    )


def _predict_all(detector: GraphEvalDetector):
    results = []
    for row in INSTANCES:
        item = DetectionInput(
            row["response_id"], row["source_id"], row["context"], row["response"],
            query=row.get("query"),
        )
        results.append((item, detector.predict(item)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "graph-eval-mock" / "local"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = str(out / "cache")

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # 1. Cold run (writes caches).
    cold = _predict_all(_detector(cache_dir, cache_only=False))
    statuses = [r.status for _, r in cold]
    check("statuses_expected", statuses == ["ok", "ok", "empty_graph"], f"got {statuses}")
    check("empty_has_no_score", cold[2][1].raw_score is None)
    check("ok_has_float_score",
          all(isinstance(r.raw_score, float) for _, r in cold if r.status == "ok"))
    check("score_direction_bounded",
          all(0.0 <= r.raw_score <= 1.0 for _, r in cold if r.status == "ok"))

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for item, r in cold:
            fh.write(json.dumps(prediction_record(r, item), sort_keys=True, ensure_ascii=False) + "\n")

    # 2. Cache-only replay: identical results, zero model calls, miss would error.
    warm = _predict_all(_detector(cache_dir, cache_only=True))
    identical = all(
        c[1].status == w[1].status and c[1].raw_score == w[1].raw_score
        for c, w in zip(cold, warm)
    )
    check("replay_identical", identical)
    calls = sum(
        int(w[1].usage.get("extractor_calls", 0)) + int(w[1].usage.get("nli_calls", 0))
        for w in warm
    )
    check("replay_zero_model_calls", calls == 0, f"calls={calls}")

    passed = all(c["ok"] for c in checks)
    summary = {
        "probe": "graph-eval-datasphere-mock-v1",
        "passed": passed,
        "python": sys.version.split()[0],
        "n_instances": len(INSTANCES),
        "statuses": statuses,
        "checks": checks,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [c["name"] for c in checks if not c["ok"]]
    print(json.dumps({"passed": passed, "python": summary["python"],
                      "failed_checks": failed or "none"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
