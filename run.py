#!/usr/bin/env python3
"""HalluGraph-KGGen entrypoint with strict and text-verified relation modes.

``--stage all`` has an intentionally leak-free order: extract graphs, tune only
on train rows, then score test rows once with frozen parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.api_runtime import CacheOnlyMissError
from src.audit import build_audit_record, write_audit
from src.config import load_config
from src.data import Instance, load_instances, unique_sources
from src.evaluate import run_evaluation
from src.extract import FakeKGGen, Graph, KGExtractor, UsageLogger
from src.matching import DictEmbedder, Embedder, RefGraph, SBERTEmbedder
from src.metrics import ScoreResult, score_response
from src.sampling import load_manifest_instances, select_qa_pilot, write_manifest
from src.tune import alpha_cv, h_array, prf_at_threshold, select_f1_threshold
from src.verifier import FakeRelationVerifier, RelationVerifier


def get_extractor(
    cfg, fake: bool, usage: UsageLogger, *, cache_only: bool = False
) -> KGExtractor:
    return KGExtractor(
        cfg,
        backend=FakeKGGen() if fake else None,
        usage=usage,
        cache_only=cache_only,
    )


def get_embedder(cfg, fake: bool) -> Embedder:
    return DictEmbedder(dim=16) if fake else SBERTEmbedder(cfg.matching.embedding_model)


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
        try:
            return source_id, extractor.extract_reference(inst.context, inst.query), None
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001
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
        try:
            return inst.response_id, extractor.extract(inst.response, kind="response"), None
        except CacheOnlyMissError:
            raise
        except Exception as exc:  # noqa: BLE001
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
    if args.qa_pilot and args.qa_pilot_manifest:
        raise SystemExit("use either --qa-pilot or --qa-pilot-manifest, not both")
    if args.qa_pilot_manifest:
        selected = load_manifest_instances(args.qa_pilot_manifest, all_instances)
        print(f"[pilot] loaded {len(selected)} fixed QA rows from {args.qa_pilot_manifest}")
        return _limit_qa_pilot(selected, args.qa_pilot_limit)
    if args.qa_pilot:
        selected = select_qa_pilot(
            all_instances, seed=args.sample_seed,
            train_sources=args.qa_pilot_train_sources,
            test_sources=args.qa_pilot_test_sources,
        )
        manifest_path = Path(args.qa_pilot_manifest_out or out_dir / "qa_pilot_manifest.json")
        write_manifest(
            manifest_path, selected, seed=args.sample_seed,
            train_sources=args.qa_pilot_train_sources, test_sources=args.qa_pilot_test_sources,
        )
        print(f"[pilot] selected {len(selected)} QA rows -> {manifest_path}")
        return _limit_qa_pilot(selected, args.qa_pilot_limit)
    if args.limit:
        return all_instances[: args.limit]
    return all_instances


def _limit_qa_pilot(instances: list[Instance], limit: int | None) -> list[Instance]:
    """Use a deterministic prefix only for a bounded runtime probe.

    The complete 20-record manifest is written before applying this cap, so a
    successful probe and the later pilot share the exact same selection.
    """
    if limit is None:
        return instances
    if limit <= 0:
        raise SystemExit("--qa-pilot-limit must be positive")
    if limit > len(instances):
        raise SystemExit(
            f"--qa-pilot-limit={limit} exceeds fixed pilot size {len(instances)}"
        )
    print(f"[pilot] runtime probe cap: {limit}/{len(instances)} fixed QA rows")
    return instances[:limit]


def _assert_cache_only_no_live_calls(cache_only: bool, usage: UsageLogger) -> None:
    if cache_only and (usage.calls != 0 or usage.provider_calls != 0):
        raise RuntimeError(
            "cache-only replay reached a live inference boundary: "
            f"logical_calls={usage.calls}, provider_calls={usage.provider_calls}"
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
    parser.add_argument("--qa-pilot", action="store_true", help="create a fixed 20-source QA pilot")
    parser.add_argument("--qa-pilot-manifest", default=None, help="reuse an existing QA pilot manifest")
    parser.add_argument("--qa-pilot-manifest-out", default=None, help="where a new pilot manifest is written")
    parser.add_argument(
        "--qa-pilot-limit",
        type=int,
        default=None,
        help="process only a deterministic prefix of a full QA manifest (probes only)",
    )
    parser.add_argument("--qa-pilot-train-sources", type=int, default=16)
    parser.add_argument("--qa-pilot-test-sources", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--fake-extractor", action="store_true", help="offline FakeKGGen/DictEmbedder/FakeVerifier")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="require warm KG and verifier caches and forbid live inference",
    )
    parser.add_argument(
        "--kg-cache-only",
        action="store_true",
        help="require warm KG caches while allowing live verifier calls",
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
        cfg,
        args.fake_extractor,
        usage,
        cache_only=args.cache_only or args.kg_cache_only,
    )
    embedder = get_embedder(cfg, args.fake_extractor)
    verifier = get_verifier(
        cfg,
        args.fake_extractor,
        usage,
        args.relation_mode,
        cache_only=args.cache_only,
    )
    scored_path = out_dir / "scored.jsonl"
    tuning_path = out_dir / "tuning.json"

    ref_graphs: dict[str, tuple[Graph, Graph]] = {}
    resp_graphs: dict[str, Graph] = {}
    failures: list[dict[str, Any]] = []
    results: dict[str, ScoreResult] = {}

    if args.stage in {"extract", "score", "tune", "all"}:
        ref_graphs, resp_graphs, failures = extract_all(cfg, instances, extractor, out_dir)
        print(f"[extract] refs={len(ref_graphs)} responses={len(resp_graphs)} failures={len(failures)}")
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
