#!/usr/bin/env python3
"""Write an archive-backed XeLaTeX report for a completed DocRED run."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def _read_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"archive member is unreadable: {name}")
    payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"archive member is not a JSON object: {name}")
    return payload


def _root(archive: tarfile.TarFile) -> str:
    roots = {
        PurePosixPath(member.name).parts[0]
        for member in archive.getmembers()
        if member.name and not member.name.startswith("/")
    }
    if len(roots) != 1:
        raise ValueError("DocRED archive must contain exactly one top-level directory")
    root = roots.pop()
    if not root.startswith("vertex-cpu-docred-kg-artifacts"):
        raise ValueError("archive is not a DocRED KG evaluation artifact")
    return root


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}"


def _pct(value: Any) -> str:
    return f"{100 * float(value):.1f}\\%"


def _ci(values: Any) -> str:
    if not isinstance(values, list) or len(values) != 2:
        return "not available"
    return f"[{_fmt(values[0])}; {_fmt(values[1])}]"


def _validate(
    manifest: dict[str, Any], metadata: dict[str, Any], metrics: dict[str, Any],
    usage: dict[str, Any], before: dict[str, Any], after: dict[str, Any],
) -> tuple[int, int]:
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("artifact manifest has no documents")
    train = sum(item.get("split") == "train_annotated" for item in documents if isinstance(item, dict))
    dev = sum(item.get("split") == "dev" for item in documents if isinstance(item, dict))
    if train != 50 or dev != 200:
        raise ValueError("artifact is not the fixed 50/200 DocRED manifest")
    if metadata.get("state") != "completed":
        raise ValueError("artifact Job is not completed")
    if metrics.get("evaluation_split") != "held-out-development":
        raise ValueError("artifact does not mark held-out development evaluation")
    if metrics.get("documents") != 200 or float(metrics.get("extraction_coverage", 0.0)) != 1.0:
        raise ValueError("artifact has incomplete held-out extraction coverage")
    if usage.get("replay", {}).get("api_calls") != 0:
        raise ValueError("artifact replay made live inference calls")
    if before != after:
        raise ValueError("cache inventory changed during cache-only replay")
    return train, dev


def render(
    output: Path, *, manifest: dict[str, Any], metadata: dict[str, Any],
    metrics: dict[str, Any], tuning: dict[str, Any], usage: dict[str, Any],
    before: dict[str, Any], after: dict[str, Any],
) -> None:
    train, dev = _validate(manifest, metadata, metrics, usage, before, after)
    dataset = manifest["dataset"]
    bootstrap = metrics.get("bootstrap", {})
    alignment = metrics.get("alignment_diagnostics", {})
    budget = metrics.get("budget", {})
    live = usage.get("live", {})
    threshold = tuning.get("selected_threshold", metrics.get("selected_relation_threshold"))
    lines = [
        "% !TEX program = xelatex",
        "\\documentclass[11pt,a4paper]{article}",
        "\\usepackage[a4paper,margin=2.25cm]{geometry}",
        "\\usepackage{fontspec,microtype,booktabs,amsmath,amssymb,hyperref}",
        "\\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}",
        "\\setmainfont{Times New Roman}",
        "\\setlength{\\parindent}{0pt}",
        "\\setlength{\\parskip}{0.55em}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\title{\\vspace{-1.2em}\\textbf{KGGen Knowledge-Graph Extraction on DocRED:}\\\\[-0.25em] Archive-verified Held-out Development Results}",
        "\\author{}",
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        "\\vspace{-1.7em}",
        "",
        "\\begin{abstract}",
        "This study evaluates the repository's KGGen extraction path on DocRED using a fixed document-level protocol.",
        f"Agreement with DocRED annotations on 200 held-out public development documents is triple recall {_fmt(metrics['triple_recall'])}, gold-supported precision {_fmt(metrics['gold_supported_precision'])}, and micro F1 {_fmt(metrics['triple_f1'])}.",
        f"Their 95\\% document-bootstrap intervals are {_ci(bootstrap.get('triple_recall_ci95'))}, {_ci(bootstrap.get('gold_supported_precision_ci95'))}, and {_ci(bootstrap.get('triple_f1_ci95'))}.",
        "The archive verifies complete extraction coverage and a subsequent zero-network cache-only replay.",
        "\\end{abstract}",
        "",
        "\\section{Task and Alignment Policy}",
        "For each document, KGGen receives only source text and extracts a free-form graph; the extraction prompt is not given DocRED's relation inventory.",
        "Predicted endpoints are mapped to document-local \\texttt{vertexSet} IDs only through unambiguous normalized mention/title aliases; ambiguous endpoints are not guessed.",
        "A recovered triple must preserve its directed head--relation--tail tuple, and duplicate aligned predictions are removed after alignment.",
        f"KGGen predicates are matched locally against DocRED relation descriptions with image-local S-BERT. The threshold was selected only on the {train} annotated \\texttt{{train\\_annotated}} documents from $\\{{0.65,0.75,0.85\\}}$, then frozen at $\\tau_r={_fmt(threshold)}$ before held-out evaluation.",
        "",
        "\\section{Fixed Protocol and Provenance}",
        f"The dataset is \\texttt{{{dataset.get('repository', '')}}} at pinned revision \\texttt{{{dataset.get('revision', '')}}}. The deterministic manifest uses seed 42, {train} calibration documents (including a 10-document live smoke stage), and {dev} labelled public \\texttt{{dev}} documents.",
        "The blind official test split and \\texttt{train\\_distant} are not used; this is therefore held-out development evaluation, not a blind benchmark test.",
        "Extraction used one in-flight request, a 4096-token base cap, an 8192-token adaptive ceiling, four-second serial pacing, and the established 30-minute continuous-429 deadline.",
        "The artifact preserves provenance, redacted progress, aggregate cache inventory and metrics; it excludes document text, prompts, raw completions, and cache keys.",
        "",
        "\\section{Held-out Development Results}",
        f"\\begin{{table}}[h]\\centering\\caption{{Micro metrics; 95\\% intervals use {bootstrap.get('replicates', 'unknown')} document bootstrap resamples. Gold-supported precision is not absolute factual precision because DocRED annotations are incomplete.}}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        r"Metric & Estimate & 95\% CI & Count \\",
        "\\midrule",
        rf"Triple recall & {_fmt(metrics['triple_recall'])} & {_ci(bootstrap.get('triple_recall_ci95'))} & {metrics['matched_triples']} / {metrics['gold_triples']} gold \\",
        rf"Gold-supported precision & {_fmt(metrics['gold_supported_precision'])} & {_ci(bootstrap.get('gold_supported_precision_ci95'))} & {metrics['matched_triples']} / {metrics['predicted_triples']} predicted \\",
        rf"Triple F1 & {_fmt(metrics['triple_f1'])} & {_ci(bootstrap.get('triple_f1_ci95'))} & micro \\",
        rf"Entity-pair recall & {_fmt(metrics['entity_pair_recall'])} & -- & {metrics['matched_entity_pairs']} / {metrics['gold_entity_pairs']} gold \\",
        rf"Entity-pair gold-supported precision & {_fmt(metrics['entity_pair_gold_supported_precision'])} & -- & {metrics['matched_entity_pairs']} / {metrics['predicted_entity_pairs']} predicted \\",
        rf"Entity-pair F1 & {_fmt(metrics['entity_pair_f1'])} & -- & micro \\",
        "\\bottomrule",
        "\\end{tabular}\\end{table}",
        f"Extraction coverage is {_pct(metrics['extraction_coverage'])}: all {metrics['documents']} held-out documents produced a result and {metrics['extraction_failures']} extraction failures were recorded.",
        "Empty graphs and explicit failures remain zero-prediction documents in the denominator; they are not silently excluded.",
        "",
        "\\section{Diagnostics, Cost, and Cache Acceptance}",
        f"There were {alignment.get('raw_predicted_triples', 0)} raw predicted triples; {alignment.get('entity_aligned_predictions', 0)} had unambiguous endpoints and {alignment.get('relation_aligned_predictions', 0)} survived relation alignment.",
        f"Endpoint-unmatched, endpoint-ambiguous, relation-unmatched, and relation-ambiguous counts were {alignment.get('entity_unmatched', 0)}, {alignment.get('entity_ambiguous', 0)}, {alignment.get('relation_unmatched', 0)}, and {alignment.get('relation_ambiguous', 0)}.",
        f"The live stage recorded {live.get('api_calls', 0)} completed live calls, {live.get('cache_hits', 0)} cache hits, and {live.get('retries', 0)} retries.",
        f"The conservative cost cap was €{_fmt(budget.get('max_eur', 0.0))}; the archive estimate was €{_fmt(budget.get('estimated_spend_eur', 0.0))}.",
        f"The mandatory replay recorded {usage.get('replay', dict()).get('api_calls', 0)} live calls and did not change the cache inventory ({before.get('files', 0)} files; its aggregate digest is recorded in the terminal archive).",
        "",
        "\\section{Limitations}",
        "DocRED is not an exhaustive knowledge base: a predicted triple absent from annotation is not necessarily false. Reported precision is therefore explicitly gold-supported precision.",
        "The free-form-predicate relation matcher was calibrated on only 50 training documents. These results evaluate this exact extraction-and-alignment protocol on the public development split; they do not estimate open-world factual precision or guarantee transfer to another corpus.",
        "\\end{document}",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tarfile.open(args.artifact, "r:gz") as archive:
        root = _root(archive)
        render(
            Path(args.output),
            manifest=_read_json(archive, f"{root}/docred_manifest.json"),
            metadata=_read_json(archive, f"{root}/run_metadata.json"),
            metrics=_read_json(archive, f"{root}/docred-live/metrics.json"),
            tuning=_read_json(archive, f"{root}/docred-live/relation_alignment_tuning.json"),
            usage=_read_json(archive, f"{root}/usage-counts.json"),
            before=_read_json(archive, f"{root}/cache-before-replay.json"),
            after=_read_json(archive, f"{root}/cache-after-replay.json"),
        )


if __name__ == "__main__":
    main()
