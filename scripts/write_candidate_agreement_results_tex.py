#!/usr/bin/env python3
"""Write an English XeLaTeX report from a redacted terminal run archive."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def _read_archive_json(archive: Path, basename: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:*") as handle:
        matches = [member for member in handle.getmembers() if member.isfile() and PurePosixPath(member.name).name == basename]
        if len(matches) != 1:
            raise SystemExit(f"terminal archive needs exactly one {basename}")
        stream = handle.extractfile(matches[0])
        if stream is None:
            raise SystemExit(f"cannot read {basename} from terminal archive")
        value = json.loads(stream.read())
    if not isinstance(value, dict):
        raise SystemExit(f"{basename} is not a JSON object")
    return value


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}"


def _interval(value: list[Any]) -> str:
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    metrics = _read_archive_json(archive, "scientific_metrics.json")
    metadata = _read_archive_json(archive, "run_metadata.json")
    if metrics.get("protocol") != "ragtruth-qa-candidate-agreement-evaluation-v1":
        raise SystemExit("archive is not a terminal candidate-agreement run")
    paired = metrics.get("paired_evaluation")
    if not isinstance(paired, dict):
        raise SystemExit("candidate-agreement archive has no paired evaluation")
    heldout = paired["heldout_test"]
    methods = heldout["methods"]
    thresholds = paired["threshold_selection"]
    coverage = metrics["coverage"]
    rows = []
    labels = {
        "candidate_agreement": "candidate-agreement",
        "strict": "strict",
        "support": "support",
        "support_critical": "support-critical",
    }
    for key in ("candidate_agreement", "strict", "support", "support_critical"):
        item = methods[key]
        theta = (
            thresholds["candidate_agreement"]["theta"]
            if key == "candidate_agreement"
            else thresholds["graph_methods"]["thresholds"][key]
        )
        rows.append(
            f"{labels[key]} & {_fmt(theta)} & {_fmt(item['roc_auc'])} {_interval(item['roc_auc_ci95'])} & "
            f"{_fmt(item['precision'])} / {_fmt(item['recall'])} / {_fmt(item['f1'])} {_interval(item['f1_ci95'])} \\\\"
        )
    deltas = paired.get("paired_vs_support_critical", {})
    candidate_delta = deltas.get("candidate_agreement", {})
    auc_delta = candidate_delta.get("roc_auc", {})
    f1_delta = candidate_delta.get("f1", {})
    sample_contract = metadata.get("sample_cache_contract", {})
    tex = r"""% !TEX program = xelatex
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.25cm]{geometry}
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{english}
\setmainfont{Times New Roman}
\usepackage{microtype}
\usepackage{amsmath,amssymb,booktabs,tabularx}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\renewcommand{\arraystretch}{1.12}
\title{\vspace{-1.2em}\textbf{Likelihood-weighted Candidate Semantic Agreement for Response-level Hallucination Detection in RAGTruth QA}}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-1.7em}
\begin{abstract}
We evaluate a non-graph, response-conditioned baseline for RAGTruth QA.  Rather than measuring uncertainty of a prompt in isolation, the method measures whether the historical labelled response agrees semantically with likely Gemini answers sampled from the same context and question.  The comparison is paired with the archive-verified strict, support, and support-critical graph results on their common held-out denominator.  It is a baseline for candidate agreement, not direct factual verification.
\end{abstract}

\section{Construct}
For source prompt $x$, historical candidate answer $A$, and $15$ sampled answers $s_i$, let $p_i$ be the likelihood-normalized mass induced by the selected-token mean log likelihood.  We define
\begin{equation}
 M(A\mid x)=\sum_{i=1}^{15}p_i\,\mathbf{1}[A\equiv s_i], \qquad
 D(A\mid x)=1-M(A\mid x).
\end{equation}
Here $A\equiv s_i$ requires bidirectional non-contradictory NLI: neither direction may be contradiction and the pair may not be neutral in both directions.  Higher $D$ therefore means that the specific labelled candidate has less semantic support from likely generator answers to the same source prompt.  This differs from prompt-only semantic entropy, which clusters samples without comparing them to $A$.

\section{Protocol and cache provenance}
The deterministic RAGTruth manifest contains 750 rows (seed 42); source \texttt{12448} is quarantined, leaving 749 analysed sources.  Gemini was fixed to \texttt{openai/gemini-2.5-flash}, 15 samples, temperature 1.0, maximum output length 65,535, and selected-token mean log likelihood.  The gateway manifest was pinned before cache reuse.  Of 11,235 expected samples, """ + str(sample_contract.get("required_reused_samples", "?")) + r""" were cache-compatible read-through entries and """ + str(sample_contract.get("required_cold_samples", "?")) + r""" were serially generated only after preflight.  Candidate/sample comparisons were cached by candidate hash, sampled-answer identities, likelihoods, NLI identity, and protocol version; no candidate or completion text appears in the archive.

The response IDs and graph scalar scores were recovered read-only from the successful R12 archive.  The primary held-out comparison is exactly the graph-scorable $n=""" + str(heldout["n"]) + r"""$ responses (""" + str(heldout["n_hallucinated"]) + r""" hallucinated and """ + str(heldout["n_factual"]) + r""" factual), not an unpaired 150-versus-147 comparison.  Candidate-agreement threshold selection used only the corresponding """ + str(thresholds["candidate_agreement"]["n"]) + r""" graph-scored training responses.

\section{Held-out paired results}
\begin{table}[h]
\centering
\caption{Results on the common graph-scored held-out responses.  Thresholds are frozen from train only.  Intervals are nonparametric bootstrap 95\% intervals.}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & $\theta$ & ROC-AUC (95\% CI) & Precision / Recall / F1 (F1 95\% CI) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

The paired candidate-agreement minus support-critical difference is $\Delta$ROC-AUC = """ + _fmt(auc_delta.get("estimate", float("nan"))) + r""" (paired bootstrap 95\% interval """ + _interval(auc_delta.get("paired_bootstrap_ci95", [float("nan"), float("nan")])) + r"""), and $\Delta$F1 = """ + _fmt(f1_delta.get("estimate", float("nan"))) + r""" (""" + _interval(f1_delta.get("paired_bootstrap_ci95", [float("nan"), float("nan")])) + r""").  These are paired uncertainty intervals for the difference; they should not be replaced by an inference from overlap of separate method intervals.

\section{Interpretation and limitations}
Candidate agreement is appropriate for response-level labels because it evaluates the actual labelled answer rather than the prompt alone.  It may detect answers that are semantically atypical under plausible answers to the same source.  It is nevertheless not a factuality proof: a confidently repeated Gemini hallucination can assign high agreement to a bad candidate, while a correct but uncommon answer can receive low agreement.  The NLI relation is also semantic rather than evidence-grounded, and likelihood mass depends on the specified generator, decoding distribution, and local NLI model.

The graph methods remain fundamentally different: strict and support assess graph alignment or textual relation support, while support-critical includes atomic-claim and full-context review.  The paired comparison is therefore a family-diverse baseline comparison, not a claim that candidate agreement subsumes graph verification.  The result is one fixed-manifest estimate and requires independent replication before general performance conclusions.

\section{Acceptance}
The terminal run covered """ + str(coverage["eligible_sources"]) + r""" eligible sources: """ + str(coverage["scored"]) + r""" scored, """ + str(coverage["unscorable_output_length"]) + r""" output-length unscorable, and """ + str(coverage["unscorable_empty_candidate"]) + r""" empty-candidate unscorable.  A strict cache-only replay regenerated all score records without Gemini calls or NLI evaluations, produced byte-identical scientific outputs, and left the post-live cache inventory unchanged.

\end{document}
"""
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    print(f"[ok] wrote {output}")


if __name__ == "__main__":
    main()
