#!/usr/bin/env python3
"""HalluGraph-KGGen entrypoint with strict and text-verified relation modes.

``--stage all`` has an intentionally leak-free order: extract graphs, tune only
on train rows, then score test rows once with frozen parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.audit import build_audit_record, write_audit
from src.cache import CacheOnlyMissError, config_value
from src.config import load_config
from src.data import Instance, load_instances, unique_sources
from src.evaluate import run_evaluation
from src.extract import FakeKGGen, Graph, KGExtractor, UsageLogger
from src.matching import DictEmbedder, Embedder, RefGraph, SBERTEmbedder
from src.metrics import ScoreResult, score_response
from src.sampling import (
    load_manifest_instances,
    qa_sample_quotas,
    select_qa_sample,
    write_manifest,
)
from src.tune import alpha_cv, h_array, prf_at_threshold, select_f1_threshold
from src.verifier import FakeRelationVerifier, RelationVerifier


def get_extractor(
    cfg, fake: bool, usage: UsageLogger, *, cache_only: bool = False
) -> KGExtractor:
    return KGExtractor(
        cfg, backend=FakeKGGen() if fake else None, usage=usage, cache_only=cache_only
    )


def get_embedder(cfg, fake: bool, *, cache_only: bool = False) -> Embedder:
    if fake:
        return DictEmbedder(dim=16)
    matching = cfg.matching
    return SBERTEmbedder(
        matching.embedding_model,
        model_revision=config_value(matching, "embedding_model_revision"),
        model_path=config_value(matching, "embedding_model_path"),
        # Cache-only replay may recompute CPU similarity scores, but it must
        # never claim vLLM's GPU or resolve an asset through the Hub even if a
        # stale runtime config says otherwise.
        device="cpu" if cache_only else str(config_value(matching, "embedding_device", "cpu")),
        local_files_only=(
            True if cache_only else bool(config_value(matching, "local_files_only", True))
        ),
    )


def get_verifier(
    cfg,
    fake: bool,
    usage: UsageLogger,
    relation_mode: str,
    *,
    cache_only: bool = False,
):
    if relation_mode == "strict":
        return None
    if fake:
        # Fake extraction is only a plumbing check; all text-grounded toy edges
        # receive a deterministic verdict without importing LiteLLM.
        return FakeRelationVerifier(default="entailed")
    if not cache_only:
        usage.try_hook_litellm()
    return RelationVerifier(cfg, usage=usage, cache_only=cache_only)


# --------------------------------------------------------------------------------------
# Stage: extract
# --------------------------------------------------------------------------------------
def extract_all(
    cfg, instances: list[Instance], extractor: KGExtractor, out_dir: Path
) -> tuple[dict[str, tuple[Graph, Graph]], dict[str, Graph], list[dict[str, Any]]]:
    """Return reference and answer graphs; references are built once per source."""
    failures: list[dict[str, Any]] = []
    ref_graphs: dict[str, tuple[Graph, Graph]] = {}
    resp_graphs: dict[str, Graph] = {}
    sources = unique_sources(instances)
    concurrency = max(1, int(cfg.llm.concurrency))

    def do_ref(item):
        source_id, inst = item
        print(f"[extract] reference:start source_id={source_id}", flush=True)
        try:
            graphs = extractor.extract_reference(inst.context, inst.query)
            print(f"[extract] reference:done source_id={source_id}", flush=True)
            return source_id, graphs, None
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[extract] reference:error source_id={source_id} error={exc!r}", flush=True)
            return source_id, None, {"stage": "reference", "source_id": source_id, "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for source_id, graphs, error in tqdm(
            pool.map(do_ref, sources.items()), total=len(sources), desc="extract G_c/G_q"
        ):
            if error:
                failures.append(error)
            else:
                ref_graphs[source_id] = graphs

    def do_response(inst: Instance):
        if inst.source_id not in ref_graphs:
            return inst.response_id, None, {
                "stage": "response(skipped: ref failed)", "response_id": inst.response_id,
                "source_id": inst.source_id, "error": "reference extraction failed",
            }
        print(
            f"[extract] response:start source_id={inst.source_id} response_id={inst.response_id}",
            flush=True,
        )
        try:
            graph = extractor.extract(inst.response, kind="response")
            print(
                f"[extract] response:done source_id={inst.source_id} response_id={inst.response_id}",
                flush=True,
            )
            return inst.response_id, graph, None
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"[extract] response:error source_id={inst.source_id} "
                f"response_id={inst.response_id} error={exc!r}",
                flush=True,
            )
            return inst.response_id, None, {
                "stage": "response", "response_id": inst.response_id,
                "source_id": inst.source_id, "error": repr(exc),
            }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(do_response, inst) for inst in instances]
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract G_a"):
            response_id, graph, error = future.result()
            if error:
                failures.append(error)
            else:
                resp_graphs[response_id] = graph

    failure_path = out_dir / "failed_extractions.jsonl"
    with open(failure_path, "w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")
    return ref_graphs, resp_graphs, failures


def _graph_summary(graph: Graph) -> dict[str, Any]:
    payload = graph.to_dict()
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def write_extraction_summary(
    instances: list[Instance],
    ref_graphs: dict[str, tuple[Graph, Graph]],
    resp_graphs: dict[str, Graph],
    failures: list[dict[str, Any]],
    extractor: KGExtractor,
    out_dir: Path,
) -> Path:
    """Write a machine-checkable proof of every selected reference/answer pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_records = [
        {
            "source_id": inst.source_id,
            "response_id": inst.response_id,
            "split": inst.split,
            "y": int(inst.y),
        }
        for inst in instances
    ]
    expected_sources = {inst.source_id for inst in instances}
    expected_responses = {inst.response_id for inst in instances}
    completed_records: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    cache_records: dict[str, dict[str, Any]] = {}

    def cache_record(text: str | None, kind: str) -> dict[str, Any] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None
        key = extractor._cache_key(normalized)
        path = extractor._cache_path(key)
        record = {
            "cache_key": key,
            "kind": kind,
            "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "cache_file": path.name,
            "cache_file_exists": path.is_file(),
        }
        cache_records.setdefault(key, record)
        return record

    for inst in instances:
        reference = ref_graphs.get(inst.source_id)
        answer = resp_graphs.get(inst.response_id)
        if reference is None or answer is None:
            continue
        context_graph, query_graph = reference
        identity = {
            "source_id": inst.source_id,
            "response_id": inst.response_id,
            "split": inst.split,
            "y": int(inst.y),
        }
        completed_records.append(identity)
        graph_records.append({
            **identity,
            "context": _graph_summary(context_graph),
            "query": _graph_summary(query_graph),
            "answer": _graph_summary(answer),
            "cache": {
                "context": cache_record(inst.context, "context"),
                "query": cache_record(inst.query, "query"),
                "answer": cache_record(inst.response, "response"),
            },
        })

    complete = (
        not failures
        and set(ref_graphs) == expected_sources
        and set(resp_graphs) == expected_responses
        and completed_records == expected_records
        and all(record["cache_file_exists"] for record in cache_records.values())
    )
    summary = {
        "protocol": "hallu-extraction-summary-v1",
        "status": "ready" if complete else "error",
        "expected_records": expected_records,
        "completed_records": completed_records,
        "expected_sources": len(expected_sources),
        "references_completed": len(ref_graphs),
        "responses_completed": len(resp_graphs),
        "pairs_completed": len(completed_records),
        "failures": failures,
        "graph_records": graph_records,
        "expected_cache_keys": sorted(cache_records),
        "cache_records": [cache_records[key] for key in sorted(cache_records)],
    }
    path = out_dir / "extraction_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Matching and scoring
# --------------------------------------------------------------------------------------
def build_refgraph(
    cfg, gc: Graph, gq: Graph, embedder: Embedder,
    tau_e: float | None = None, tau_r: float | None = None,
) -> RefGraph:
    matching = cfg.matching
    if tau_e is None and tau_r is None:
        return RefGraph(gc.entities | gq.entities, gc.relations | gq.relations, matching, embedder)

    class ThresholdView:
        entity_sim_threshold = tau_e if tau_e is not None else matching.entity_sim_threshold
        relation_sim_threshold = tau_r if tau_r is not None else matching.relation_sim_threshold
        allow_substring_match = matching.allow_substring_match
        direction_sensitive_edges = matching.direction_sensitive_edges
        inverse_edge_match = getattr(matching, "inverse_edge_match", False)
        min_substring_chars = getattr(matching, "min_substring_chars", 2)
        stopwords = getattr(matching, "stopwords", [])

    return RefGraph(gc.entities | gq.entities, gc.relations | gq.relations, ThresholdView, embedder)


def score_all(
    cfg,
    instances: list[Instance],
    ref_graphs: dict[str, tuple[Graph, Graph]],
    resp_graphs: dict[str, Graph],
    embedder: Embedder,
    *,
    tau_e: float | None = None,
    tau_r: float | None = None,
    relation_mode: str = "strict",
    verifier=None,
) -> dict[str, ScoreResult]:
    """Score all available rows; support mode verifies each grounded answer edge."""
    if relation_mode == "support" and verifier is None:
        raise ValueError("support scoring requires a relation verifier")
    results: dict[str, ScoreResult] = {}
    refgraph_cache: dict[str, RefGraph] = {}
    description = f"score {relation_mode} (tau_e={tau_e},tau_r={tau_r})"
    for inst in tqdm(instances, desc=description):
        if inst.source_id not in ref_graphs or inst.response_id not in resp_graphs:
            continue
        gc, gq = ref_graphs[inst.source_id]
        refgraph = refgraph_cache.get(inst.source_id)
        if refgraph is None:
            refgraph = build_refgraph(cfg, gc, gq, embedder, tau_e, tau_r)
            refgraph_cache[inst.source_id] = refgraph
        results[inst.response_id] = score_response(
            resp_graphs[inst.response_id], refgraph, gc, gq,
            context=inst.context, query=inst.query,
            verifier=verifier if relation_mode == "support" else None,
            verifier_matching_params={
                "tau_e": float(cfg.matching.entity_sim_threshold if tau_e is None else tau_e),
                "tau_r": float(cfg.matching.relation_sim_threshold if tau_r is None else tau_r),
                "allow_substring_match": bool(cfg.matching.allow_substring_match),
                "min_substring_chars": int(cfg.matching.min_substring_chars),
                "stopwords": list(cfg.matching.stopwords),
            },
        )
    return results


def persist_scored(
    path: Path, instances: list[Instance], results: dict[str, ScoreResult], relation_mode: str
) -> None:
    by_id = {inst.response_id: inst for inst in instances}
    with open(path, "w", encoding="utf-8") as handle:
        for response_id, result in results.items():
            inst = by_id[response_id]
            record = {
                "response_id": response_id, "source_id": inst.source_id, "task": inst.task,
                "gen_model": inst.gen_model, "split": inst.split, "y": inst.y,
                "context_len": len(inst.context), "gt_span_types": inst.gt_span_types,
                "relation_mode": relation_mode, "score": result.to_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_scored(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(
    scored: list[dict[str, Any]],
    alpha_strict: float,
    alpha_support: float,
    relation_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in scored:
        result = ScoreResult.from_dict(record["score"])
        h_strict = result.h_for_mode(alpha_strict, "strict")
        h_support = (
            result.h_for_mode(alpha_support, "support")
            if relation_mode == "support"
            else None
        )
        primary_h = h_support if relation_mode == "support" else h_strict
        primary_rp_h = result.support_rp_only_h() if relation_mode == "support" else result.rp_only_h()
        statuses = [entry.get("status") for entry in result.relation_audits if entry.get("status")]
        rows.append({
            "response_id": record["response_id"], "source_id": record["source_id"],
            "task": record["task"], "gen_model": record["gen_model"], "split": record["split"],
            "y": int(record["y"]), "context_len": int(record["context_len"]),
            "Vc": result.Vc, "Ec": result.Ec, "Vq": result.Vq, "Eq": result.Eq,
            "Va": result.Va, "Ea": result.Ea,
            "EG": result.EG,
            "RP": result.RP, "RP_defined": result.RP_defined,
            "RP_strict": result.RP_strict,
            "RP_grounded": result.RP_grounded,
            "RP_entailed_cond": result.RP_entailed_cond,
            "RP_support": result.RP_support,
            "RP_support_defined": result.RP_support_defined,
            "unscorable": result.unscorable, "ref_empty": result.ref_empty,
            "CFI_strict": result.cfi_for_mode(alpha_strict, "strict"),
            "H_strict": h_strict,
            "CFI_support": (
                result.cfi_for_mode(alpha_support, "support")
                if relation_mode == "support"
                else None
            ),
            "H_support": h_support,
            "H": primary_h,
            "H_eg": result.eg_only_h(), "H_rp": primary_rp_h,
            "H_rp_strict": result.rp_only_h(), "H_rp_support": result.support_rp_only_h(),
            "relation_statuses": json.dumps(statuses),
        })
    return rows


# --------------------------------------------------------------------------------------
# Train-only joint threshold/alpha tuning
# --------------------------------------------------------------------------------------
def tune_joint(
    cfg,
    instances: list[Instance],
    ref_graphs,
    resp_graphs,
    embedder,
    verifier,
    relation_mode: str,
    out_dir: Path,
) -> dict[str, Any]:
    train = [
        inst for inst in instances
        if inst.split == "train" and inst.source_id in ref_graphs and inst.response_id in resp_graphs
    ]
    if not train:
        raise ValueError("no train instances available for tuning")
    seed = int(cfg.eval.seed)
    alpha_grid = [float(value) for value in cfg.eval.alpha_grid]
    folds = int(cfg.eval.alpha_cv_folds)
    default_te = float(cfg.matching.entity_sim_threshold)
    default_tr = float(cfg.matching.relation_sim_threshold)
    rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[float, float], dict[str, ScoreResult]] = {}

    tau_r_candidates = [default_tr] if relation_mode == "support" else list(map(float, cfg.eval.tau_r_sweep))
    for tau_e in map(float, cfg.eval.tau_e_sweep):
        for tau_r in tau_r_candidates:
            results = score_all(
                cfg, train, ref_graphs, resp_graphs, embedder,
                tau_e=tau_e, tau_r=tau_r, relation_mode=relation_mode, verifier=verifier,
            )
            score_cache[(tau_e, tau_r)] = results
            scores = [results[i.response_id] for i in train if i.response_id in results]
            labels = [i.y for i in train if i.response_id in results]
            if cfg.metrics.alpha is not None:
                fixed = float(cfg.metrics.alpha)
                _, trace = alpha_cv(scores, labels, [fixed], folds, seed, mode=relation_mode)
            else:
                _, trace = alpha_cv(scores, labels, alpha_grid, folds, seed, mode=relation_mode)
            for alpha, cv_auc in trace.items():
                rows.append({
                    "tau_e": tau_e, "tau_r": tau_r, "alpha": float(alpha),
                    "n_train": len(scores), "cv_mean_auc": cv_auc,
                })

    valid = [row for row in rows if not math.isnan(float(row["cv_mean_auc"]))]
    if valid:
        best = max(
            valid,
            key=lambda row: (
                float(row["cv_mean_auc"]),
                -abs(float(row["tau_e"]) - default_te) - abs(float(row["tau_r"]) - default_tr)
                - abs(float(row["alpha"]) - 0.7),
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(float(row["tau_e"]) - default_te) + abs(float(row["tau_r"]) - default_tr),
                abs(float(row["alpha"]) - 0.7),
            ),
        )

    selected = score_cache[(float(best["tau_e"]), float(best["tau_r"]))]
    train_scores = [selected[i.response_id] for i in train if i.response_id in selected]
    train_y = np.array([i.y for i in train if i.response_id in selected])
    alpha = float(best["alpha"])
    H, mask = h_array(train_scores, alpha, mode=relation_mode)
    theta, train_f1 = select_f1_threshold(H[mask], train_y[mask])
    selected_trace = {
        str(row["alpha"]): row["cv_mean_auc"]
        for row in rows
        if row["tau_e"] == best["tau_e"] and row["tau_r"] == best["tau_r"]
    }
    info = {
        "relation_mode": relation_mode,
        "alpha": alpha,
        "theta": theta,
        "train_f1": train_f1,
        "tau_e": float(best["tau_e"]),
        "tau_r": float(best["tau_r"]),
        "selected_cv_auc": best["cv_mean_auc"],
        "alpha_cv": selected_trace,
        "joint_cv": rows,
    }
    (out_dir / "tuning.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def write_all_audits(
    instances: list[Instance],
    results: dict[str, ScoreResult],
    info: dict[str, Any],
    impute_h: float,
    out_dir: Path,
) -> None:
    by_id = {inst.response_id: inst for inst in instances}
    audit_dir = out_dir / "audit"
    relation_mode = str(info.get("relation_mode", "strict"))
    for response_id, result in tqdm(results.items(), desc="audit", total=len(results)):
        inst = by_id.get(response_id)
        if inst is None:
            continue
        record = build_audit_record(
            inst, result, float(info["alpha"]), alpha_support=float(info["alpha"]),
            relation_mode=relation_mode, impute_h=impute_h,
        )
        write_audit(record, audit_dir)


def _apply_cli_overrides(cfg, args) -> None:
    if args.data_dir:
        cfg.data._data["dir"] = args.data_dir  # noqa: SLF001
        cfg.data.dir = args.data_dir
    if args.cache_dir:
        cfg._data["cache_dir"] = args.cache_dir  # noqa: SLF001
        cfg.cache_dir = args.cache_dir
        verdict_dir = str(Path(args.cache_dir).parent / "verdicts")
        cfg.relation_verifier._data["cache_dir"] = verdict_dir  # noqa: SLF001
        cfg.relation_verifier.cache_dir = verdict_dir


def _select_instances(args, cfg, out_dir: Path) -> list[Instance]:
    all_instances = load_instances(
        cfg.data.dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true)
    )
    if args.qa_sample and args.qa_manifest:
        raise SystemExit("use either --qa-sample or --qa-manifest, not both")
    if args.qa_manifest:
        selected = load_manifest_instances(args.qa_manifest, all_instances)
        print(f"[qa-sample] loaded {len(selected)} fixed QA rows from {args.qa_manifest}")
        return _limit_qa_sample(selected, args.qa_sample_limit)
    if args.qa_sample:
        train_sources, test_sources = qa_sample_quotas(
            args.qa_sample_size, args.qa_test_fraction,
        )
        selected = select_qa_sample(
            all_instances, seed=args.sample_seed,
            train_sources=train_sources, test_sources=test_sources,
        )
        manifest_path = Path(args.qa_manifest_out or out_dir / "qa_sample_manifest.json")
        write_manifest(
            manifest_path, selected, seed=args.sample_seed,
            train_sources=train_sources, test_sources=test_sources,
        )
        print(
            f"[qa-sample] selected {len(selected)} QA rows "
            f"(train={train_sources}, test={test_sources}) -> {manifest_path}"
        )
        return _limit_qa_sample(selected, args.qa_sample_limit)
    if args.limit:
        return all_instances[: args.limit]
    return all_instances


def _limit_qa_sample(instances: list[Instance], limit: int | None) -> list[Instance]:
    """Cap a fixed QA sample only for a bounded runtime compatibility probe.

    The full manifest is written before this cap is applied, so the probe and
    the later full evaluation use exactly the same deterministic prefix.
    It is never used for metric tuning or evaluation.
    """
    if limit is None:
        return instances
    if limit <= 0:
        raise SystemExit("--qa-sample-limit must be positive")
    if limit > len(instances):
        raise SystemExit(f"--qa-sample-limit={limit} exceeds the fixed sample size {len(instances)}")
    print(f"[qa-sample] runtime probe cap: {limit}/{len(instances)} fixed QA rows")
    return instances[:limit]


def _assert_cache_only_no_live_calls(cache_only: bool, usage: UsageLogger) -> None:
    """Turn the replay's zero-live-call promise into a checked invariant."""
    if cache_only and usage.calls != 0:
        raise RuntimeError(
            f"cache-only replay recorded {usage.calls} live inference call(s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluGraph-KGGen on RAGTruth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", default="all", choices=["extract", "score", "tune", "evaluate", "all"])
    parser.add_argument("--data-dir", default=None, help="override data.dir")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    parser.add_argument("--cache-dir", default=None, help="override shared KG cache directory")
    parser.add_argument("--limit", type=int, default=None, help="cap #instances (smoke tests)")
    parser.add_argument("--relation-mode", choices=["strict", "support"], default="strict")
    parser.add_argument(
        "--qa-sample", "--qa-pilot", dest="qa_sample", action="store_true",
        help="create a deterministic, balanced QA sample (legacy alias: --qa-pilot)",
    )
    parser.add_argument(
        "--qa-manifest", "--qa-pilot-manifest", dest="qa_manifest", default=None,
        help="reuse an existing deterministic QA manifest",
    )
    parser.add_argument(
        "--qa-manifest-out", "--qa-pilot-manifest-out", dest="qa_manifest_out", default=None,
        help="where a newly selected QA manifest is written",
    )
    parser.add_argument(
        "--qa-sample-limit", "--qa-pilot-limit", dest="qa_sample_limit", type=int, default=None,
        help="extract only a deterministic prefix of a fixed QA sample (runtime probes only)",
    )
    parser.add_argument(
        "--qa-sample-size", type=int, default=20,
        help="total source-level QA records to select (default: 20)",
    )
    parser.add_argument(
        "--qa-test-fraction", default="0.2",
        help="held-out test fraction; must yield positive even train/test counts (default: 0.2)",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--fake-extractor", action="store_true", help="offline FakeKGGen/DictEmbedder/FakeVerifier")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="require warm KG/verifier caches and forbid every live inference call",
    )
    parser.add_argument(
        "--kg-cache-only",
        action="store_true",
        help="require warm KG caches while allowing live support-verifier calls",
    )
    parser.add_argument("--no-audit", action="store_true", help="skip writing per-response audit JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    out_dir = Path(args.output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(
        f"[cfg] model={cfg.llm.model} mode={args.relation_mode} "
        f"fake={args.fake_extractor} cache_only={args.cache_only} "
        f"kg_cache_only={args.kg_cache_only} stage={args.stage}"
    )
    if not args.fake_extractor and "PLACEHOLDER" in str(cfg.llm.model):
        print("[!] llm.model is PLACEHOLDER; configure it before a live run.")

    instances = _select_instances(args, cfg, out_dir)
    n_train = sum(inst.split == "train" for inst in instances)
    n_test = sum(inst.split == "test" for inst in instances)
    print(f"[data] {len(instances)} responses (train={n_train}, test={n_test})")
    usage = UsageLogger(out_dir / "usage.jsonl")
    extractor = get_extractor(
        cfg, args.fake_extractor, usage, cache_only=args.cache_only or args.kg_cache_only
    )
    embedder = get_embedder(
        cfg, args.fake_extractor, cache_only=args.cache_only or args.kg_cache_only
    )
    verifier = get_verifier(
        cfg, args.fake_extractor, usage, args.relation_mode, cache_only=args.cache_only
    )
    scored_path = out_dir / "scored.jsonl"
    tuning_path = out_dir / "tuning.json"

    ref_graphs: dict[str, tuple[Graph, Graph]] = {}
    resp_graphs: dict[str, Graph] = {}
    failures: list[dict[str, Any]] = []
    results: dict[str, ScoreResult] = {}

    if args.stage in {"extract", "score", "tune", "all"}:
        ref_graphs, resp_graphs, failures = extract_all(cfg, instances, extractor, out_dir)
        extraction_summary = write_extraction_summary(
            instances, ref_graphs, resp_graphs, failures, extractor, out_dir
        )
        print(f"[extract] refs={len(ref_graphs)} responses={len(resp_graphs)} failures={len(failures)}")
        print(f"[extract] summary={extraction_summary}")
    if args.stage == "extract":
        _assert_cache_only_no_live_calls(args.cache_only, usage)
        print(f"[done] extract only; elapsed={time.perf_counter() - start:.1f}s")
        return

    if args.stage == "score":
        results = score_all(
            cfg, instances, ref_graphs, resp_graphs, embedder,
            relation_mode=args.relation_mode, verifier=verifier,
        )
        persist_scored(scored_path, instances, results, args.relation_mode)
        _assert_cache_only_no_live_calls(args.cache_only, usage)
        print(f"[done] score only; wrote {len(results)} rows -> {scored_path}")
        return

    if args.stage in {"tune", "all"}:
        info = tune_joint(
            cfg, instances, ref_graphs, resp_graphs, embedder, verifier, args.relation_mode, out_dir
        )
        print(f"[tune] alpha={info['alpha']} tau_e={info['tau_e']} tau_r={info['tau_r']} theta={info['theta']:.4f}")
    else:
        if not tuning_path.exists():
            raise SystemExit(f"{tuning_path} missing; run --stage tune/all first")
        info = json.loads(tuning_path.read_text(encoding="utf-8"))
        if info.get("relation_mode") != args.relation_mode:
            raise SystemExit("tuning.json relation mode differs from --relation-mode")

    if args.stage == "tune":
        _assert_cache_only_no_live_calls(args.cache_only, usage)
        print(f"[done] tune only; elapsed={time.perf_counter() - start:.1f}s")
        return

    if args.stage == "all":
        # This is the only all-instance score after train-only parameter selection.
        results = score_all(
            cfg, instances, ref_graphs, resp_graphs, embedder,
            tau_e=float(info["tau_e"]), tau_r=float(info["tau_r"]),
            relation_mode=args.relation_mode, verifier=verifier,
        )
        persist_scored(scored_path, instances, results, args.relation_mode)
    if not scored_path.exists():
        raise SystemExit(f"{scored_path} missing; run --stage score/all first")
    scored = load_scored(scored_path)

    alpha = float(info["alpha"])
    rows = build_rows(scored, alpha, alpha, args.relation_mode)
    summary = run_evaluation(
        rows, alpha, float(info["theta"]), cfg, out_dir,
        tuning_info={"alpha_cv": info.get("alpha_cv", {}), "joint_cv": info.get("joint_cv", [])},
        usage_summary=usage.summary(), n_failed=len(failures), relation_mode=args.relation_mode,
        tau_e=float(info["tau_e"]), tau_r=float(info["tau_r"]),
    )
    print("[eval] summary:", json.dumps(_short(summary), indent=2))

    if not args.no_audit:
        if not results:
            results = {record["response_id"]: ScoreResult.from_dict(record["score"]) for record in scored}
        write_all_audits(instances, results, info, float(cfg.metrics.empty_response_impute_h), out_dir)
        print(f"[audit] wrote {len(results)} records -> {out_dir / 'audit'}")
    _assert_cache_only_no_live_calls(args.cache_only, usage)
    print(f"[done] stage={args.stage}; elapsed={time.perf_counter() - start:.1f}s usage={usage.summary()}")


def _short(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ["relation_mode", "alpha", "theta", "tau_e", "tau_r", "overall_AUC_exclude_unscorable", "overall_F1"]
    return {key: summary.get(key) for key in keys}


if __name__ == "__main__":
    main()
