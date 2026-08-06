#!/usr/bin/env python3
"""Write a XeLaTeX result note from the redacted paired-subset archive only."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "ragtruth-candidate-agreement-r12-paired-entropy-subset-v1"
METHODS = ("candidate_agreement", "strict", "support", "support_critical")
METHOD_LABELS = {
    "candidate_agreement": "candidate-agreement",
    "strict": "strict",
    "support": "support",
    "support_critical": "support-critical",
}


def _read_json(archive: Path, name: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as handle:
        matches = [member for member in handle.getmembers() if member.isfile() and PurePosixPath(member.name).name == name]
        if len(matches) != 1:
            raise SystemExit(f"archive must contain exactly one {name}")
        stream = handle.extractfile(matches[0])
        if stream is None:
            raise SystemExit(f"cannot read {name} from archive")
        value = json.loads(stream.read())
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return value


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}"


def _interval(value: list[Any]) -> str:
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = _read_json(args.archive, "metrics.json")
    metadata = _read_json(args.archive, "run_metadata.json")
    preflight = _read_json(args.archive, "preflight.json")
    replay = _read_json(args.archive, "replay.json")
    if metrics.get("protocol") != PROTOCOL or metadata.get("state") != "completed":
        raise SystemExit("archive is not a completed paired candidate-agreement subset run")
    if preflight.get("gemini_api_calls") != 0 or replay.get("gemini_api_calls") != 0 or replay.get("nli_pair_evaluations") != 0:
        raise SystemExit("archive does not satisfy the zero-inference cache-only contract")

    coverage = metrics["coverage"]
    thresholds = metrics["threshold_selection"]
    heldout = metrics["heldout_test"]
    rows = []
    for method in METHODS:
        result = heldout[method]
        rows.append(
            f"{METHOD_LABELS[method]} & {_fmt(thresholds[method]['threshold'])} & "
            f"{_fmt(result['roc_auc'])} {_interval(result['roc_auc_ci95'])} & "
            f"{_fmt(result['precision'])} / {_fmt(result['recall'])} / {_fmt(result['f1'])} "
            f"{_interval(result['f1_ci95'])} " + r"\\"
        )
    delta = metrics["paired_difference_vs_support_critical"]["candidate_agreement"]
    execution = metrics["execution"]
    tex = r"""% !TEX program = xelatex
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.25cm]{geometry}
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{english}
\setmainfont{Times New Roman}
\usepackage{microtype}
\usepackage{amsmath,amssymb,booktabs}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\renewcommand{\arraystretch}{1.12}
\title{\vspace{-1.2em}\textbf{Likelihood-weighted Candidate Semantic Agreement: a Paired RAGTruth QA Baseline}}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-1.7em}
\begin{abstract}
We evaluate a non-graph, response-conditioned hallucination baseline on the exact intersection of a completed semantic-sampling run and archive-verified HalluGraph scores.  The construct measures semantic agreement between the labelled historical answer and likely sampled answers to the same context and question.  This is a cache-only paired retrospective comparison, not a new Gemini-generation experiment or a factual-verification method.
\end{abstract}

\section{Construct}
For source prompt $x$, historical candidate answer $A$, and $15$ cached Gemini samples $s_i$, let $p_i$ denote normalized likelihood mass derived from selected-token mean log likelihood.  Candidate agreement and disagreement are
\begin{equation}
 M(A\mid x)=\sum_{i=1}^{15}p_i\,\mathbf{1}[A\equiv s_i],\qquad D(A\mid x)=1-M(A\mid x).
\end{equation}
Here $A\equiv s_i$ is the same bidirectional non-contradictory NLI relation used by the semantic-entropy implementation: neither direction may be a contradiction and the two directions may not both be neutral.  Larger $D$ is interpreted as less semantic support for the particular labelled candidate under likely sampled answers.  Unlike prompt-only semantic entropy, this construct explicitly scores $A$.

\section{Paired cache-only protocol}
The analysis uses the exact response-ID intersection between the completed 559-source semantic-entropy run and the R12 graph-score archive: """ + str(coverage["paired"]) + r""" responses, comprising """ + str(coverage["train"]["n"]) + r""" paired training responses and """ + str(coverage["test"]["n"]) + r""" held-out responses (""" + str(coverage["test"]["positive"]) + r""" hallucinated and """ + str(coverage["test"]["negative"]) + r""" factual).  The 63 completed entropy responses without a graph score are excluded rather than compared against a different denominator.

All 7,440 required semantic samples were loaded from the pinned compatible cache.  Consequently this evaluation made zero Gemini calls.  It performed """ + str(execution["nli_pair_evaluations"]) + r""" local directional NLI evaluations to fill a separate content-addressed candidate-comparison cache.  Thresholds for every method were selected only on the 405 paired training responses.  A subsequent cache-only replay made zero Gemini calls and zero NLI evaluations, reproduced score records byte-for-byte, and left the candidate-comparison cache inventory unchanged.

\section{Held-out paired results}
\begin{table}[h]
\centering
\caption{Common $n=""" + str(coverage["test"]["n"]) + r"""$ held-out denominator.  Thresholds are fixed on the paired training split only.  Intervals are nonparametric bootstrap 95\% intervals.}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & $\theta$ & ROC-AUC (95\% CI) & Precision / Recall / F1 (F1 95\% CI) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

Relative to support-critical on the same held-out IDs, candidate-agreement has $\Delta$ROC-AUC = """ + _fmt(delta["roc_auc"]["estimate"]) + r""" (paired bootstrap 95\% interval """ + _interval(delta["roc_auc"]["paired_bootstrap_ci95"]) + r"""), and $\Delta$F1 = """ + _fmt(delta["f1"]["estimate"]) + r""" (""" + _interval(delta["f1"]["paired_bootstrap_ci95"]) + r""").  These are paired uncertainty summaries on one fixed manifest; they are not an independent replication or a formal significance claim.

\section{Interpretation and limitations}
Candidate agreement is a meaningful non-graph response-level baseline because it tests the labelled answer rather than generator uncertainty alone.  In this setting, however, it is much weaker at ranking hallucinated answers than support-critical (ROC-AUC """ + _fmt(heldout["candidate_agreement"]["roc_auc"]) + r""" versus """ + _fmt(heldout["support_critical"]["roc_auc"]) + r""").  Its train-selected threshold is effectively below the observed score range, yielding an all-positive held-out decision rule: recall is 1.000 but precision is limited by the 55/91 positive prevalence.  Thus its apparently moderate F1 should not be mistaken for discriminative evidence.

The construct is not direct factual verification.  A generator may consistently repeat a false claim, producing agreement with a hallucinated candidate; conversely, a correct but uncommon answer may obtain little likelihood mass.  It also inherits the sampled generator distribution and local NLI model.  Strict, support, and support-critical remain fundamentally different graph/evidence-based mechanisms.  This comparison establishes a family-diverse baseline on a shared denominator, not a replacement for evidence-grounded verification.

\end{document}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
