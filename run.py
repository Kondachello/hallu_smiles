#!/usr/bin/env python3
"""HalluGraph-KGGen single entrypoint.

    python run.py --config config.yaml --stage all

Stages:
    extract   : build G_c, G_q (once per source) and G_a (per response); fills the disk cache.
    score     : compute EG / RP / audit lists for every response -> results/scored.jsonl
    tune       : alpha (5-fold CV), theta (F1), tau_e x tau_r sweep -> results/tuning.json  [TRAIN ONLY]
    evaluate  : AUC/P/R/F1/ablation/bootstrap/diagnostics on TEST -> metrics.csv + report.md
    all       : the above in order.

Offline plumbing check (no API key, no torch):
    python run.py --stage all --fake-extractor --limit 40
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.data import Instance, load_instances, unique_sources
from src.extract import ExtractionError, FakeKGGen, Graph, KGExtractor, UsageLogger
from src.matching import DictEmbedder, Embedder, RefGraph, SBERTEmbedder
from src.metrics import ScoreResult, score_response
from src.tune import alpha_cv, h_array, prf_at_threshold, safe_auc, select_f1_threshold
from src.audit import build_audit_record, write_audit
from src.evaluate import run_evaluation


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
def get_extractor(cfg, fake: bool, usage: UsageLogger) -> KGExtractor:
    backend = FakeKGGen() if fake else None
    return KGExtractor(cfg, backend=backend, usage=usage)


def get_embedder(cfg, fake: bool) -> Embedder:
    if fake:
        return DictEmbedder(dim=16)  # deterministic, offline
    return SBERTEmbedder(cfg.matching.embedding_model)


# --------------------------------------------------------------------------------------
# Stage: extract
# --------------------------------------------------------------------------------------
def extract_all(
    cfg, instances: list[Instance], extractor: KGExtractor, out_dir: Path
) -> tuple[dict[str, tuple[Graph, Graph]], dict[str, Graph], list[dict[str, Any]]]:
    """Return (ref_graphs[source_id]=(G_c,G_q), resp_graphs[response_id]=G_a, failures)."""
    failures: list[dict[str, Any]] = []
    ref_graphs: dict[str, tuple[Graph, Graph]] = {}
    resp_graphs: dict[str, Graph] = {}

    sources = unique_sources(instances)  # G_c/G_q computed ONCE per source_id
    conc = max(1, int(cfg.llm.concurrency))

    # --- reference graphs (context + query), once per source ---
    def do_ref(sid_inst):
        sid, inst = sid_inst
        try:
            gc, gq = extractor.extract_reference(inst.context, inst.query)
            return sid, (gc, gq), None
        except Exception as e:  # noqa: BLE001
            return sid, None, {"stage": "reference", "source_id": sid, "error": repr(e)}

    with ThreadPoolExecutor(max_workers=conc) as ex:
        for sid, val, err in tqdm(
            ex.map(do_ref, sources.items()), total=len(sources), desc="extract G_c/G_q"
        ):
            if err:
                failures.append(err)
            else:
                ref_graphs[sid] = val

    # --- response graphs, per response ---
    def do_resp(inst: Instance):
        if inst.source_id not in ref_graphs:
            return inst.response_id, None, {
                "stage": "response(skipped: ref failed)",
                "response_id": inst.response_id, "source_id": inst.source_id,
                "error": "reference extraction failed",
            }
        try:
            ga = extractor.extract(inst.response, kind="response")
            return inst.response_id, ga, None
        except Exception as e:  # noqa: BLE001
            return inst.response_id, None, {
                "stage": "response", "response_id": inst.response_id,
                "source_id": inst.source_id, "error": repr(e),
            }

    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(do_resp, inst) for inst in instances]
        for f in tqdm(as_completed(futs), total=len(futs), desc="extract G_a"):
            rid, ga, err = f.result()
            if err:
                failures.append(err)
            else:
                resp_graphs[rid] = ga

    # persist failures
    fpath = out_dir / "failed_extractions.jsonl"
    with open(fpath, "w", encoding="utf-8") as fh:
        for r in failures:
            fh.write(json.dumps(r) + "\n")
    if failures:
        print(f"[warn] {len(failures)} extraction failures -> {fpath}")
    return ref_graphs, resp_graphs, failures


# --------------------------------------------------------------------------------------
# Stage: score
# --------------------------------------------------------------------------------------
def build_refgraph(cfg, gc: Graph, gq: Graph, embedder: Embedder,
                   tau_e: float | None = None, tau_r: float | None = None) -> RefGraph:
    m = cfg.matching
    # allow threshold overrides for the sweep without mutating global config
    if tau_e is None and tau_r is None:
        return RefGraph(gc.entities | gq.entities, gc.relations | gq.relations, m, embedder)

    class _M:  # lightweight override view
        entity_sim_threshold = tau_e if tau_e is not None else m.entity_sim_threshold
        relation_sim_threshold = tau_r if tau_r is not None else m.relation_sim_threshold
        allow_substring_match = m.allow_substring_match
        direction_sensitive_edges = m.direction_sensitive_edges
        inverse_edge_match = getattr(m, "inverse_edge_match", False)
        min_substring_chars = getattr(m, "min_substring_chars", 2)
        stopwords = getattr(m, "stopwords", [])

    return RefGraph(gc.entities | gq.entities, gc.relations | gq.relations, _M, embedder)


def score_all(
    cfg, instances: list[Instance],
    ref_graphs: dict[str, tuple[Graph, Graph]], resp_graphs: dict[str, Graph],
    embedder: Embedder, tau_e: float | None = None, tau_r: float | None = None,
) -> dict[str, ScoreResult]:
    """Score every response whose graphs are available. RefGraph is built once per source."""
    results: dict[str, ScoreResult] = {}
    refgraph_cache: dict[str, RefGraph] = {}
    for inst in tqdm(instances, desc=f"score (tau_e={tau_e},tau_r={tau_r})"):
        if inst.source_id not in ref_graphs or inst.response_id not in resp_graphs:
            continue
        gc, gq = ref_graphs[inst.source_id]
        rg = refgraph_cache.get(inst.source_id)
        if rg is None:
            rg = build_refgraph(cfg, gc, gq, embedder, tau_e, tau_r)
            refgraph_cache[inst.source_id] = rg
        results[inst.response_id] = score_response(resp_graphs[inst.response_id], rg, gc, gq)
    return results


def persist_scored(path: Path, instances: list[Instance], results: dict[str, ScoreResult]) -> None:
    by_id = {i.response_id: i for i in instances}
    with open(path, "w", encoding="utf-8") as f:
        for rid, res in results.items():
            inst = by_id[rid]
            rec = {
                "response_id": rid, "source_id": inst.source_id, "task": inst.task,
                "gen_model": inst.gen_model, "split": inst.split, "y": inst.y,
                "context_len": len(inst.context), "gt_span_types": inst.gt_span_types,
                "score": res.to_dict(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_scored(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------------------
# Rows for evaluation / tuning
# --------------------------------------------------------------------------------------
def build_rows(scored: list[dict[str, Any]], alpha: float, impute_h: float) -> list[dict[str, Any]]:
    rows = []
    for rec in scored:
        res = ScoreResult.from_dict(rec["score"])
        rows.append({
            "response_id": rec["response_id"], "source_id": rec["source_id"],
            "task": rec["task"], "gen_model": rec["gen_model"], "split": rec["split"],
            "y": int(rec["y"]), "context_len": int(rec["context_len"]),
            "Vc": res.Vc, "Ec": res.Ec, "Vq": res.Vq, "Eq": res.Eq, "Va": res.Va, "Ea": res.Ea,
            "EG": res.EG, "RP": res.RP, "RP_defined": res.RP_defined,
            "unscorable": res.unscorable, "ref_empty": res.ref_empty,
            "H": res.h(alpha),
            "H_eg": res.eg_only_h(),
            "H_rp": res.rp_only_h(),
        })
    return rows


def scored_to_results(scored: list[dict[str, Any]]) -> tuple[list[ScoreResult], list[dict[str, Any]]]:
    res = [ScoreResult.from_dict(r["score"]) for r in scored]
    return res, scored


# --------------------------------------------------------------------------------------
# Stage: tune (TRAIN ONLY)
# --------------------------------------------------------------------------------------
def tau_sweep(
    cfg, instances: list[Instance],
    ref_graphs, resp_graphs, embedder, alpha: float, seed: int, sweep_limit: int,
) -> list[dict[str, Any]]:
    """Re-score a sample of TRAIN responses for each (tau_e, tau_r); report train AUC."""
    train = [i for i in instances if i.split == "train"
             and i.source_id in ref_graphs and i.response_id in resp_graphs]
    if sweep_limit and len(train) > sweep_limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train), size=sweep_limit, replace=False)
        train = [train[i] for i in sorted(idx)]
    out = []
    for te in cfg.eval.tau_e_sweep:
        for tr in cfg.eval.tau_r_sweep:
            results = score_all(cfg, train, ref_graphs, resp_graphs, embedder, tau_e=te, tau_r=tr)
            reslist = [results[i.response_id] for i in train if i.response_id in results]
            y = np.array([i.y for i in train if i.response_id in results])
            H, mask = h_array(reslist, alpha)
            auc = safe_auc(H[mask], y[mask])
            out.append({"tau_e": te, "tau_r": tr, "n": int(mask.sum()), "train_AUC": auc})
    return out


def tune(
    cfg, instances: list[Instance], scored: list[dict[str, Any]],
    ref_graphs, resp_graphs, embedder, out_dir: Path,
) -> dict[str, Any]:
    seed = int(cfg.eval.seed)
    train_recs = [r for r in scored if r["split"] == "train"]
    train_res = [ScoreResult.from_dict(r["score"]) for r in train_recs]
    train_y = [int(r["y"]) for r in train_recs]

    # alpha via 5-fold CV
    grid = list(cfg.eval.alpha_grid)
    if cfg.metrics.alpha is not None:
        alpha = float(cfg.metrics.alpha)
        alpha_trace = {a: float("nan") for a in grid}
        print(f"[tune] alpha fixed by config = {alpha}")
    else:
        alpha, alpha_trace = alpha_cv(train_res, train_y, grid, int(cfg.eval.alpha_cv_folds), seed)
        print(f"[tune] alpha (CV) = {alpha}")

    # theta via F1 on train, using tuned alpha
    H, mask = h_array(train_res, alpha)
    theta, train_f1 = select_f1_threshold(H[mask], np.array(train_y)[mask])
    print(f"[tune] theta = {theta:.4f} (train F1 = {train_f1:.4f})")

    # tau sensitivity sweep (train only) -- needs graphs
    sweep = []
    if ref_graphs and resp_graphs:
        sweep = tau_sweep(cfg, instances, ref_graphs, resp_graphs, embedder, alpha, seed,
                          int(getattr(cfg.eval, "sweep_limit", 300) or 300))

    info = {"alpha": alpha, "theta": theta, "train_f1": train_f1,
            "alpha_cv": alpha_trace, "tau_sweep": sweep}
    (out_dir / "tuning.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------
def write_all_audits(instances, results: dict[str, ScoreResult], alpha, impute_h, out_dir: Path):
    by_id = {i.response_id: i for i in instances}
    audit_dir = out_dir / "audit"
    for rid, res in tqdm(results.items(), desc="audit", total=len(results)):
        inst = by_id.get(rid)
        if inst is None:
            continue
        rec = build_audit_record(inst, res, alpha, impute_h=impute_h)
        write_audit(rec, audit_dir)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HalluGraph-KGGen on RAGTruth")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["extract", "score", "tune", "evaluate", "all"])
    ap.add_argument("--data-dir", default=None, help="override data.dir")
    ap.add_argument("--output-dir", default=None, help="override output_dir")
    ap.add_argument("--limit", type=int, default=None, help="cap #instances (smoke tests)")
    ap.add_argument("--fake-extractor", action="store_true",
                    help="use offline FakeKGGen + DictEmbedder (no API, no torch)")
    ap.add_argument("--no-audit", action="store_true", help="skip writing per-response audits")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.data._data["dir"] = args.data_dir  # noqa: SLF001
        cfg.data.dir = args.data_dir
    out_dir = Path(args.output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[cfg] model={cfg.llm.model} fake={args.fake_extractor} stage={args.stage}")
    if not args.fake_extractor and "PLACEHOLDER" in str(cfg.llm.model):
        print("[!] llm.model is still 'PLACEHOLDER'. Set a real model in config.yaml, or "
              "use --fake-extractor for an offline plumbing run.")

    # --- data ---
    instances = load_instances(
        cfg.data.dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true)
    )
    if args.limit:
        # keep sources intact: take the first N responses but ensure their sources are present
        instances = instances[: args.limit]
    n_train = sum(i.split == "train" for i in instances)
    n_test = sum(i.split == "test" for i in instances)
    print(f"[data] {len(instances)} responses  (train={n_train}, test={n_test})")

    usage = UsageLogger(out_dir / "usage.jsonl")
    extractor = get_extractor(cfg, args.fake_extractor, usage)
    embedder = get_embedder(cfg, args.fake_extractor)

    scored_path = out_dir / "scored.jsonl"
    tuning_path = out_dir / "tuning.json"

    ref_graphs: dict[str, tuple[Graph, Graph]] = {}
    resp_graphs: dict[str, Graph] = {}
    results: dict[str, ScoreResult] = {}
    failures: list[dict[str, Any]] = []

    need_graphs = args.stage in ("extract", "score", "tune", "all")
    if need_graphs:
        ref_graphs, resp_graphs, failures = extract_all(cfg, instances, extractor, out_dir)
        print(f"[extract] refs={len(ref_graphs)} responses={len(resp_graphs)} "
              f"failures={len(failures)}  usage={usage.summary()}")

    if args.stage in ("score", "tune", "all"):
        results = score_all(cfg, instances, ref_graphs, resp_graphs, embedder)
        persist_scored(scored_path, instances, results)
        print(f"[score] scored {len(results)} responses -> {scored_path}")

    if args.stage == "extract":
        print(f"[done] extract only. elapsed={time.perf_counter()-t0:.1f}s")
        return

    # load scored (from disk if a prior stage produced it)
    if not scored_path.exists():
        raise SystemExit(f"{scored_path} missing; run --stage score/all first.")
    scored = load_scored(scored_path)

    # --- tune (train only) ---
    if args.stage in ("tune", "all"):
        info = tune(cfg, instances, scored, ref_graphs, resp_graphs, embedder, out_dir)
    else:
        if not tuning_path.exists():
            raise SystemExit(f"{tuning_path} missing; run --stage tune/all first.")
        info = json.loads(tuning_path.read_text(encoding="utf-8"))
    alpha, theta = info["alpha"], info["theta"]

    if args.stage == "tune":
        print(f"[done] tune only. alpha={alpha} theta={theta:.4f} "
              f"elapsed={time.perf_counter()-t0:.1f}s")
        return

    # --- evaluate (test scored once) ---
    impute_h = float(cfg.metrics.empty_response_impute_h)
    rows = build_rows(scored, alpha, impute_h)
    tuning_info = {"alpha_cv": info.get("alpha_cv", {}), "tau_sweep": info.get("tau_sweep", [])}
    summary = run_evaluation(
        rows, alpha, theta, cfg, out_dir,
        tuning_info=tuning_info, usage_summary=usage.summary(), n_failed=len(failures),
    )
    print("[eval] summary:", json.dumps(_short(summary), indent=2))

    # --- audit for every scored response ---
    if not args.no_audit:
        if not results:
            # reconstruct ScoreResults from disk if scoring happened in an earlier process
            by_id = {i.response_id: i for i in instances}
            results = {r["response_id"]: ScoreResult.from_dict(r["score"])
                       for r in scored if r["response_id"] in by_id}
        write_all_audits(instances, results, alpha, impute_h, out_dir)
        print(f"[audit] wrote {len(results)} records -> {out_dir/'audit'}")

    print(f"[done] stage={args.stage}. elapsed={time.perf_counter()-t0:.1f}s  usage={usage.summary()}")


def _short(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ["alpha", "theta", "tau_e", "tau_r",
            "overall_AUC_exclude_unscorable", "overall_F1", "overall_AUC_impute"]
    return {k: summary.get(k) for k in keys}


if __name__ == "__main__":
    main()
