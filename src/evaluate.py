"""Evaluation & reporting: AUC / P / R / F1, ablations, bootstrap CIs, diagnostics, plots.

All tuning is assumed already done (alpha, theta, tau_e, tau_r fixed on train). This module
scores the TEST split exactly once and writes metrics.csv + a markdown report + plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import config_value, evaluation_runtime_metadata
from .tune import prf_at_threshold, safe_auc


# --------------------------------------------------------------------------------------
# Bootstrap CIs
# --------------------------------------------------------------------------------------
def _bootstrap(
    metric_fn, h: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[float, float]:
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(h)
    h, y = h[finite], y[finite]
    if len(h) < 3 or len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(h))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        v = metric_fn(h[b], y[b])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def bootstrap_auc_ci(h, y, n, seed):
    return _bootstrap(lambda hh, yy: safe_auc(hh, yy), h, y, n, seed)


def bootstrap_f1_ci(h, y, theta, n, seed):
    def f1_fn(hh, yy):
        return prf_at_threshold(hh, yy, theta)[2]

    return _bootstrap(f1_fn, h, y, n, seed)


# --------------------------------------------------------------------------------------
# Grouped metric tables
# --------------------------------------------------------------------------------------
def _auc_row(df: pd.DataFrame, hcol: str) -> dict[str, Any]:
    sub = df[np.isfinite(df[hcol])]
    return {
        "n": int(len(sub)),
        "n_pos": int(sub["y"].sum()),
        "AUC": safe_auc(sub[hcol].to_numpy(), sub["y"].to_numpy()) if len(sub) else float("nan"),
    }


def auc_breakdown(df: pd.DataFrame, group: str, hcol: str = "H") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group):
        r = _auc_row(g, hcol)
        r[group] = key
        rows.append(r)
    overall = _auc_row(df, hcol)
    overall[group] = "OVERALL"
    rows.append(overall)
    return pd.DataFrame(rows)[[group, "n", "n_pos", "AUC"]]


def prf_breakdown(df: pd.DataFrame, theta: float, group: str, hcol: str = "H") -> pd.DataFrame:
    rows = []
    for key, g in list(df.groupby(group)) + [("OVERALL", df)]:
        sub = g[np.isfinite(g[hcol])]
        p, r, f1 = prf_at_threshold(sub[hcol].to_numpy(), sub["y"].to_numpy(), theta)
        rows.append({group: key, "n": int(len(sub)), "P": p, "R": r, "F1": f1})
    return pd.DataFrame(rows)


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """AUC for EG-alone, RP-alone, CFI (H) -- the paper's trio."""
    rows = []
    for name, col in [("EG-only", "H_eg"), ("RP-only", "H_rp"), ("CFI (H)", "H")]:
        sub = df[np.isfinite(df[col])]
        rows.append({
            "score": name,
            "n": int(len(sub)),
            "AUC": safe_auc(sub[col].to_numpy(), sub["y"].to_numpy()) if len(sub) else float("nan"),
        })
    return pd.DataFrame(rows)


def relation_score_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare strict and support hallucination scores when both are available."""
    rows = []
    for name, col in [("H_strict", "H_strict"), ("H_support", "H_support")]:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        sub = df[np.isfinite(values)].copy()
        h = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        rows.append({
            "detector": name,
            "n": int(len(sub)),
            "AUC": safe_auc(h, sub["y"].to_numpy()) if len(sub) else float("nan"),
        })
    return pd.DataFrame(rows)


def relation_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of per-edge audit statuses split by RAGTruth response label."""
    import json

    rows = []
    for _, record in df.iterrows():
        raw = record.get("relation_statuses", "[]")
        try:
            statuses = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except (TypeError, ValueError):
            statuses = []
        for status in statuses:
            rows.append({"y": int(record["y"]), "status": str(status)})
    if not rows:
        return pd.DataFrame(columns=["status", "y", "n"])
    return pd.DataFrame(rows).groupby(["status", "y"]).size().reset_index(name="n")


def context_length_buckets(df: pd.DataFrame, buckets: Sequence[float], hcol: str = "H") -> pd.DataFrame:
    edges = list(buckets)
    labels = [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]
    df = df.copy()
    df["clb"] = pd.cut(df["context_len"], bins=edges, labels=labels, right=False, include_lowest=True)
    rows = []
    for key, g in df.groupby("clb", observed=True):
        r = _auc_row(g, hcol)
        r["bucket"] = str(key)
        r["mean_ctx_len"] = float(g["context_len"].mean())
        rows.append(r)
    return pd.DataFrame(rows)[["bucket", "mean_ctx_len", "n", "n_pos", "AUC"]] if rows else pd.DataFrame()


def graph_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, g in list(df.groupby("task")) + [("OVERALL", df)]:
        rows.append({
            "task": task,
            "n": int(len(g)),
            "mean_Vc": float(g["Vc"].mean()),
            "mean_Ec": float(g["Ec"].mean()),
            "mean_Va": float(g["Va"].mean()),
            "mean_Ea": float(g["Ea"].mean()),
            "frac_empty_Ea": float((g["Ea"] == 0).mean()),
            "frac_unscorable_Va0": float(g["unscorable"].mean()),
        })
    return pd.DataFrame(rows)


def wilcoxon_factual_vs_hallucinated(df: pd.DataFrame, hcol: str = "H") -> dict[str, Any]:
    """Rank-based test that hallucinated responses score higher on H than factual ones.

    NOTE: the two groups are independent and unequal-length, so the correct statistic is the
    Wilcoxon rank-sum / Mann-Whitney U test (signed-rank requires *paired* data, which we do
    not have). We report Mann-Whitney U and label it accordingly.
    """
    from scipy.stats import mannwhitneyu

    sub = df[np.isfinite(df[hcol])]
    h_pos = sub[sub["y"] == 1][hcol].to_numpy()
    h_neg = sub[sub["y"] == 0][hcol].to_numpy()
    if len(h_pos) < 1 or len(h_neg) < 1:
        return {"test": "mann-whitney-u", "U": float("nan"), "p_value": float("nan"),
                "median_H_hallucinated": float("nan"), "median_H_factual": float("nan")}
    U, p = mannwhitneyu(h_pos, h_neg, alternative="greater")
    return {
        "test": "mann-whitney-u (rank-sum; groups unpaired)",
        "U": float(U), "p_value": float(p),
        "median_H_hallucinated": float(np.median(h_pos)),
        "median_H_factual": float(np.median(h_neg)),
    }


# --------------------------------------------------------------------------------------
# Plots (guarded -- skipped if matplotlib missing or disabled)
# --------------------------------------------------------------------------------------
def make_plots(df: pd.DataFrame, out_dir: Path, buckets: Sequence[float]) -> list[str]:
    made: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return made
    out_dir.mkdir(parents=True, exist_ok=True)

    # H histograms per task (y=0 vs y=1)
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        g = df[(df["task"] == task) & np.isfinite(df["H"])]
        ax.hist(g[g["y"] == 0]["H"], bins=20, alpha=0.6, label="factual (y=0)", density=True)
        ax.hist(g[g["y"] == 1]["H"], bins=20, alpha=0.6, label="hallucinated (y=1)", density=True)
        ax.set_title(f"H distribution — {task}")
        ax.set_xlabel("H = 1 - CFI"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout()
    p = out_dir / "h_distributions.png"; fig.savefig(p, dpi=120); plt.close(fig)
    made.append(str(p))

    # AUC vs context length
    clb = context_length_buckets(df, buckets)
    if len(clb):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(clb)), clb["AUC"], marker="o")
        ax.set_xticks(range(len(clb))); ax.set_xticklabels(clb["bucket"], rotation=45, ha="right")
        ax.set_ylabel("AUC"); ax.set_xlabel("context length bucket (chars)")
        ax.set_title("AUC vs context length"); ax.axhline(0.5, ls="--", color="grey")
        fig.tight_layout()
        p = out_dir / "auc_vs_context_length.png"; fig.savefig(p, dpi=120); plt.close(fig)
        made.append(str(p))
    return made


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame, floatfmt: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def run_evaluation(
    rows: list[dict[str, Any]],
    alpha: float,
    theta: float,
    cfg,
    out_dir: str | Path,
    tuning_info: dict[str, Any] | None = None,
    usage_summary: dict[str, Any] | None = None,
    n_failed: int = 0,
    relation_mode: str = "strict",
    tau_e: float | None = None,
    tau_r: float | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "metrics.csv", index=False)  # per-response scored rows (all splits)

    test = full[full["split"] == "test"].copy()
    # Policy (a): exclude unscorable ; (b): impute H for unscorable
    test_excl = test[~test["unscorable"].astype(bool)].copy()
    impute_h = float(cfg.metrics.empty_response_impute_h)
    test_imp = test.copy()
    test_imp["H"] = test_imp["H"].where(~test_imp["unscorable"].astype(bool), impute_h)

    seed = int(cfg.eval.seed)
    n_boot = int(cfg.eval.n_bootstrap)
    buckets = list(cfg.eval.context_length_buckets)

    # ---- primary tables (policy a: exclude unscorable) ----
    auc_task = auc_breakdown(test_excl, "task")
    auc_model = auc_breakdown(test_excl, "gen_model")
    prf_task = prf_breakdown(test_excl, theta, "task")
    ablation = ablation_table(test_excl)
    relation_comparison = relation_score_comparison(test_excl)
    status_breakdown = relation_status_breakdown(test)
    clb = context_length_buckets(test_excl, buckets)
    gstats = graph_stats(test)
    wil = wilcoxon_factual_vs_hallucinated(test_excl)

    H_all = test_excl["H"].to_numpy()
    y_all = test_excl["y"].to_numpy()
    overall_auc = safe_auc(H_all, y_all)
    auc_ci = bootstrap_auc_ci(H_all, y_all, n_boot, seed)
    p_o, r_o, f1_o = prf_at_threshold(H_all, y_all, theta)
    f1_ci = bootstrap_f1_ci(H_all, y_all, theta, n_boot, seed)

    # ---- policy (b): impute ----
    auc_imp = safe_auc(test_imp["H"].to_numpy(), test_imp["y"].to_numpy())
    p_i, r_i, f1_i = prf_at_threshold(test_imp["H"].to_numpy(), test_imp["y"].to_numpy(), theta)

    plots = make_plots(test_excl, plots_dir, buckets) if bool(cfg.eval.make_plots) else []

    # ---- degenerate-case counts ----
    degen = {
        "n_test": int(len(test)),
        "n_unscorable_Va0": int(test["unscorable"].sum()),
        "n_ref_empty": int(test["ref_empty"].sum()),
        "n_empty_Ea": int((test["Ea"] == 0).sum()),
        "n_failed_extractions": int(n_failed),
    }

    summary = {
        "relation_mode": relation_mode,
        "runtime": evaluation_runtime_metadata(cfg),
        "alpha": alpha, "theta": theta,
        "tau_e": float(cfg.matching.entity_sim_threshold if tau_e is None else tau_e),
        "tau_r": float(cfg.matching.relation_sim_threshold if tau_r is None else tau_r),
        "overall_AUC_exclude_unscorable": overall_auc,
        "overall_AUC_ci95": auc_ci,
        "overall_P": p_o, "overall_R": r_o, "overall_F1": f1_o,
        "overall_F1_ci95": f1_ci,
        "overall_AUC_impute": auc_imp,
        "overall_F1_impute": f1_i,
        "degenerate": degen,
    }
    pd.DataFrame([_flatten_summary(summary)]).to_csv(out_dir / "summary_metrics.csv", index=False)

    _write_report(
        out_dir, cfg, summary, auc_task, auc_model, prf_task, ablation, clb, gstats, wil,
        tuning_info or {}, usage_summary or {}, degen, plots,
        policy_b={"AUC": auc_imp, "P": p_i, "R": r_i, "F1": f1_i, "impute_h": impute_h},
        relation_comparison=relation_comparison, status_breakdown=status_breakdown,
    )
    return summary


def _flatten_summary(
    s: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flatten nested runtime/metric metadata into stable CSV columns."""
    out: dict[str, Any] = {}
    for k, v in s.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_summary(v, key))
        elif isinstance(v, (list, tuple)):
            seq = list(v) + [None, None]
            out[f"{key}_lo"], out[f"{key}_hi"] = seq[0], seq[1]
        else:
            out[key] = v
    return out


def _write_report(out_dir, cfg, summary, auc_task, auc_model, prf_task, ablation, clb,
                   gstats, wil, tuning_info, usage, degen, plots, policy_b,
                   relation_comparison, status_breakdown) -> None:
    L = []
    A = L.append
    A("# HalluGraph-KGGen — RAGTruth evaluation report\n")
    A(f"- **LLM model:** `{cfg.llm.model}`")
    A(f"- **LLM revision:** `{config_value(cfg.llm, 'model_revision') or 'unrecorded'}`")
    A(
        "- **Runtime fingerprint:** "
        f"`{config_value(cfg.llm, 'runtime_fingerprint') or 'unrecorded'}`"
    )
    A(
        "- **Structured output:** "
        f"`{config_value(cfg.llm, 'structured_output_transport', 'none')}` / "
        f"`{config_value(cfg.llm, 'structured_output_backend', 'none')}`"
    )
    A(f"- **relation detector mode:** `{summary['relation_mode']}`")
    A(f"- **Embedding model:** `{cfg.matching.embedding_model}`")
    A(
        "- **Embedding revision:** "
        f"`{config_value(cfg.matching, 'embedding_model_revision') or 'unrecorded'}`"
    )
    A(
        "- **Embedding runtime:** "
        f"path=`{config_value(cfg.matching, 'embedding_model_path') or 'Hub cache'}`, "
        f"device=`{config_value(cfg.matching, 'embedding_device', 'cpu')}`, "
        f"local-files-only=`{bool(config_value(cfg.matching, 'local_files_only', True))}`"
    )
    A(f"- **alpha (tuned on train):** {summary['alpha']}")
    A(f"- **theta / decision threshold (tuned on train F1):** {summary['theta']:.4f}")
    A(f"- **tau_e / tau_r:** {summary['tau_e']} / {summary['tau_r']}")
    A(f"- **empty-response-graph policy:** {cfg.metrics.empty_response_graph_policy}\n")

    A("## 1. Headline (test split, policy (a): exclude unscorable responses)\n")
    auc = summary["overall_AUC_exclude_unscorable"]; ci = summary["overall_AUC_ci95"]
    f1ci = summary["overall_F1_ci95"]
    A(f"- **ROC-AUC:** {auc:.4f}  (95% CI [{ci[0]:.4f}, {ci[1]:.4f}])")
    A(f"- **Precision / Recall / F1 @ theta:** {summary['overall_P']:.4f} / "
      f"{summary['overall_R']:.4f} / {summary['overall_F1']:.4f}  "
      f"(F1 95% CI [{f1ci[0]:.4f}, {f1ci[1]:.4f}])")
    A("\n> RAGTruth Table 5 reference points (response-level F1): "
      "Prompt-GPT-4-turbo = 63.4, fine-tuned Llama-2-13B = 78.7.\n")

    A("## 2. Policy (b): impute H=%.2f for unscorable (|V_a|=0)\n" % policy_b["impute_h"])
    A(f"- **ROC-AUC:** {policy_b['AUC']:.4f}")
    A(f"- **P / R / F1 @ theta:** {policy_b['P']:.4f} / {policy_b['R']:.4f} / {policy_b['F1']:.4f}\n")

    A("## 3. AUC by task\n"); A(_df_to_md(auc_task) + "\n")
    A("## 4. AUC by generator model\n"); A(_df_to_md(auc_model) + "\n")
    A("## 5. Precision / Recall / F1 by task (@ tuned theta)\n"); A(_df_to_md(prf_task) + "\n")
    A("## 6. Ablation — EG-only / RP-only / CFI(H) AUC\n"); A(_df_to_md(ablation) + "\n")
    A("## 6a. Strict vs. text-supported relation score (test)\n")
    A(_df_to_md(relation_comparison) + "\n" if len(relation_comparison) else "_support not scored_\n")
    A("## 6b. Relation audit statuses by response label\n")
    A(_df_to_md(status_breakdown) + "\n" if len(status_breakdown) else "_no relation audits_\n")
    A("## 7. AUC vs. context length (RAGTruth CLB-style buckets)\n")
    A(_df_to_md(clb) + "\n" if len(clb) else "_no data_\n")
    A("## 8. Graph statistics (mean |V|, |E|; empty-E_a fraction)\n"); A(_df_to_md(gstats) + "\n")
    A("## 9. Significance (hallucinated vs factual H)\n")
    A(f"- test: {wil['test']}")
    A(f"- U = {wil['U']:.1f}, p = {wil['p_value']:.3e}")
    A(f"- median H (hallucinated) = {wil['median_H_hallucinated']:.4f}, "
      f"median H (factual) = {wil['median_H_factual']:.4f}\n")

    A("## 10. Degenerate cases & failures\n")
    A(_df_to_md(pd.DataFrame([degen])) + "\n")

    if tuning_info:
        A("## 11. Tuning trace (train split only)\n")
        if "alpha_cv" in tuning_info:
            A("**alpha CV mean AUC:**\n")
            A(_df_to_md(pd.DataFrame(
                [{"alpha": a, "cv_mean_auc": v} for a, v in tuning_info["alpha_cv"].items()])) + "\n")
        if "tau_sweep" in tuning_info:
            A("**tau_e x tau_r sensitivity (train AUC):**\n")
            A(_df_to_md(pd.DataFrame(tuning_info["tau_sweep"])) + "\n")
        if "joint_cv" in tuning_info:
            A("**joint tau_e x tau_r x alpha CV:**\n")
            A(_df_to_md(pd.DataFrame(tuning_info["joint_cv"])) + "\n")

    if usage:
        A("## 12. API usage / cost\n")
        A(_df_to_md(pd.DataFrame([usage])) + "\n")

    if plots:
        A("## 13. Plots\n")
        for p in plots:
            rel = Path(p).relative_to(out_dir) if Path(p).is_relative_to(out_dir) else Path(p)
            A(f"![{Path(p).stem}]({rel})")
        A("")

    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
