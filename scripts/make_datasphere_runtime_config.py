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
        "--work-dir",
        required=True,
        help="Writable, job-local directory for caches and generated artifacts.",
    )
    args = parser.parse_args()

    with open(args.base_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    work_dir = Path(args.work_dir)
    config["llm"]["model"] = f"openai/{args.model_id}"
    config["llm"]["api_base"] = args.api_base
    config["llm"]["api_key_env"] = "OPENAI_API_KEY"
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
