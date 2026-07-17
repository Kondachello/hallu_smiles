#!/usr/bin/env python3
"""Create a Job-local config without modifying the repository config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dspy_adapter import XGRAMMAR_STRICT_REQUEST_BACKEND


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
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
        "--explicit-clustering",
        action="store_true",
        help=(
            "Run KGGen generate(cluster=False) followed by the same KGGen.cluster() call. "
            "This preserves LLM clustering while emitting a precise local-runtime phase boundary."
        ),
    )
    parser.add_argument(
        "--vllm-guided-json",
        action="store_true",
        help=(
            "Use vLLM's native guided_json transport for DSPy typed outputs. "
            "Required by the pinned local vLLM 0.6.3 runtime; it does not change KGGen."
        ),
    )
    parser.add_argument(
        "--structured-output-transport",
        choices=("response_format", "guided_json", "none"),
        default="response_format",
    )
    parser.add_argument(
        "--structured-output-backend",
        choices=("xgrammar", "guidance"),
        default="xgrammar",
    )
    parser.add_argument(
        "--structured-output-request-backend",
        default=None,
        help=(
            "Exact request-level guided-decoding backend and options. The "
            "DataSphere XGrammar runtime pins disable-any-whitespace and no-fallback."
        ),
    )
    parser.add_argument(
        "--embedding-model-path",
        default="/opt/hallu/models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--embedding-model-revision",
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )
    parser.add_argument(
        "--cluster-min-retention-ratio",
        type=float,
        default=None,
        help="Probe-only clustering collapse threshold; omit for the full pilot.",
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
    if args.cluster_min_retention_ratio is not None and not (
        0.0 <= args.cluster_min_retention_ratio <= 1.0
    ):
        raise ValueError("--cluster-min-retention-ratio must be between 0 and 1")
    if args.vllm_guided_json and args.structured_output_transport != "guided_json":
        raise ValueError("--vllm-guided-json conflicts with --structured-output-transport")
    if (
        args.structured_output_transport == "response_format"
        and args.structured_output_backend == "xgrammar"
    ):
        args.structured_output_request_backend = (
            args.structured_output_request_backend
            or XGRAMMAR_STRICT_REQUEST_BACKEND
        )
        if args.structured_output_request_backend != XGRAMMAR_STRICT_REQUEST_BACKEND:
            raise ValueError(
                "DataSphere response_format requires strict bounded-whitespace XGrammar"
            )
    elif args.structured_output_request_backend is not None:
        raise ValueError(
            "--structured-output-request-backend is only supported for "
            "response_format with xgrammar"
        )

    with open(args.base_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    work_dir = Path(args.work_dir)
    config["llm"]["model"] = f"openai/{args.model_id}"
    config["llm"]["model_revision"] = args.model_revision
    config["llm"]["runtime_fingerprint"] = args.runtime_fingerprint
    config["llm"]["api_base"] = args.api_base
    config["llm"]["api_key_env"] = "OPENAI_API_KEY"
    # KGGen 0.4 otherwise defaults to 16k completion tokens.  The g1.1
    # Llama server deliberately has an 8k context window, so the upstream
    # default makes every extraction request fail before generation begins.
    config["llm"]["max_tokens"] = args.max_tokens
    config["llm"]["concurrency"] = args.concurrency
    config["llm"]["structured_output_transport"] = args.structured_output_transport
    config["llm"]["structured_output_backend"] = args.structured_output_backend
    config["llm"]["structured_output_request_backend"] = (
        args.structured_output_request_backend
    )
    config["llm"]["vllm_guided_json"] = args.vllm_guided_json
    config["extraction"]["serial_chunking"] = args.serial_chunking
    config["extraction"]["cluster_max_items"] = args.cluster_max_items
    config["extraction"]["cluster_min_retention_ratio"] = args.cluster_min_retention_ratio
    config["matching"]["embedding_model_path"] = args.embedding_model_path
    config["matching"]["embedding_model_revision"] = args.embedding_model_revision
    config["matching"]["embedding_device"] = "cpu"
    config["matching"]["local_files_only"] = True
    if args.disable_clustering:
        config["extraction"]["cluster"] = False
    if args.explicit_clustering:
        config["extraction"]["explicit_clustering"] = True
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
