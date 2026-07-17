#!/usr/bin/env python3
"""Create a Job-local config without modifying the repository config.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens for KGGen; must fit the local vLLM context window.",
    )
    parser.add_argument(
        "--cluster-max-items",
        type=int,
        default=None,
        help=(
            "Optional maximum entity/predicate candidates for KGGen LLM clustering. "
            "Omit it for the faithful, unbounded KGGen clustering path."
        ),
    )
    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        help=(
            "Keep raw KGGen entities/triples and skip KGGen's optional LLM clustering. "
            "Used only for the local DataSphere Llama runtime after a verified cluster-side Pydantic stall."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Response-level KGGen workers; local vLLM must use one shared DSPy client serially.",
    )
    parser.add_argument(
        "--serial-chunking",
        dest="serial_chunking",
        action="store_true",
        default=True,
        help="Disable KGGen's nested chunk ThreadPoolExecutor for local vLLM reliability.",
    )
    parser.add_argument(
        "--native-chunking",
        dest="serial_chunking",
        action="store_false",
        help="Use KGGen's native parallel chunking (not safe for the shared local vLLM client).",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Writable, job-local directory for caches and generated artifacts.",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.cluster_max_items is not None and args.cluster_max_items <= 0:
        raise ValueError("--cluster-max-items must be positive")

    with open(args.base_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    work_dir = Path(args.work_dir)
    config["llm"]["model"] = f"openai/{args.model_id}"
    config["llm"]["api_base"] = args.api_base
    config["llm"]["api_key_env"] = "OPENAI_API_KEY"
    # KGGen 0.4 otherwise defaults to 16k completion tokens.  The g1.1
    # Llama server deliberately has an 8k context window, so the upstream
    # default makes every extraction request fail before generation begins.
    config["llm"]["max_tokens"] = args.max_tokens
    config["llm"]["concurrency"] = args.concurrency
    config["extraction"]["serial_chunking"] = args.serial_chunking
    config["extraction"]["cluster_max_items"] = args.cluster_max_items
    if args.disable_clustering:
        config["extraction"]["cluster"] = False
    config["data"]["dir"] = args.data_dir
    # DataSphere mounts project storage read-only in Jobs.  Every mutable file
    # therefore belongs to the job-local output directory, never DS_PROJECT_HOME.
    config["cache_dir"] = str(work_dir / "cache" / "kg")
    config.setdefault("relation_verifier", {})["cache_dir"] = str(work_dir / "cache" / "verdicts")
    config["output_dir"] = str(work_dir / "results")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
