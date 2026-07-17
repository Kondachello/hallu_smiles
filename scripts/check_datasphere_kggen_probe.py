#!/usr/bin/env python3
"""Exercise the real KGGen/DSPy structured-output path against local vLLM.

The usual OpenAI-compatible completion smoke test only proves a plain chat
request.  KGGen additionally goes through DSPy's typed-output adapter, which
is where an incompatible DSPy/LiteLLM combination can otherwise consume a GPU
Job without ever advancing the first graph.  This deliberately tiny graph
uses the same model name, API base, timeouts and raw-triple extraction mode as
the pilot, but runs before the 20 QA records are touched. Optional KGGen LLM
clustering can be requested explicitly for a separate diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROBE_TEXT = "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine."


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_probe(
    *,
    model_id: str,
    api_base: str,
    timeout_s: float,
    max_tokens: int,
    cluster: bool = False,
    vllm_guided_json: bool = False,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    from importlib import metadata

    from kg_gen import KGGen

    started = time.perf_counter()
    backend = KGGen(
        model=f"openai/{model_id}",
        max_tokens=max_tokens,
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY", "local-datasphere-key"),
        api_base=api_base,
    )
    # KGGen does not expose these DSPy controls in its constructor.  They must
    # match KGExtractor so the probe and pilot have the same failure boundary.
    backend.lm.kwargs["timeout"] = timeout_s
    backend.lm.num_retries = 0
    # The pilot deliberately keeps raw KGGen triples on the local Llama
    # profile. Exercise that exact code path here: clustering is an optional
    # post-processing LLM pass, not part of relation extraction.
    if vllm_guided_json:
        import dspy

        from src.dspy_adapter import vllm_guided_json_adapter

        with dspy.context(lm=backend.lm, adapter=vllm_guided_json_adapter()):
            graph = backend.generate(input_data=PROBE_TEXT, cluster=cluster)
    else:
        graph = backend.generate(input_data=PROBE_TEXT, cluster=cluster)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": f"openai/{model_id}",
        "api_base": api_base,
        "timeout_s": timeout_s,
        "max_tokens": max_tokens,
        "cluster": cluster,
        "vllm_guided_json": vllm_guided_json,
        "versions": {
            "kg-gen": metadata.version("kg-gen"),
            "dspy": metadata.version("dspy"),
            "litellm": metadata.version("litellm"),
        },
        "entities": len(getattr(graph, "entities", set())),
        "relations": len(getattr(graph, "relations", set())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also exercise optional KGGen LLM clustering (off for the local pilot).",
    )
    parser.add_argument(
        "--vllm-guided-json",
        action="store_true",
        help="Use vLLM's native guided_json schema transport, like the local pilot.",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.port <= 0:
        raise ValueError("--port must be positive")
    report = run_probe(
        model_id=args.model_id,
        api_base=f"http://127.0.0.1:{args.port}/v1",
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        cluster=args.cluster,
        vllm_guided_json=args.vllm_guided_json,
    )
    _atomic_json(Path(args.report), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
