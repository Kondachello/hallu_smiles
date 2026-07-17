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


PROBE_TEXT = (
    "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine. "
    "Marie Curie was born in Warsaw."
)


def _has_endpoints(relations: set[Any], left: str, right: str) -> bool:
    left = left.casefold()
    right = right.casefold()
    for relation in relations:
        if not isinstance(relation, (tuple, list)) or len(relation) != 3:
            continue
        subject = str(relation[0]).casefold()
        obj = str(relation[2]).casefold()
        if (left in subject and right in obj) or (left in obj and right in subject):
            return True
    return False


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
    structured_output_transport: str = "response_format",
    request_backend: str | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    from importlib import metadata

    from kg_gen import KGGen
    from src.dspy_adapter import XGRAMMAR_STRICT_REQUEST_BACKEND

    if structured_output_transport == "response_format":
        request_backend = request_backend or XGRAMMAR_STRICT_REQUEST_BACKEND
        if request_backend != XGRAMMAR_STRICT_REQUEST_BACKEND:
            raise ValueError(
                "response_format probe must use strict bounded-whitespace XGrammar"
            )

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
    from src.dspy_adapter import install_dspy_completion_guard

    install_dspy_completion_guard(backend.lm)
    # Exercise the same strict adapter as the pilot.  When ``cluster`` is set,
    # KGGen also runs its official LLM clustering pass under that adapter;
    # raw relation extraction remains the first observable failure boundary.
    if structured_output_transport != "none":
        import dspy

        from src.dspy_adapter import strict_json_schema_adapter, vllm_guided_json_adapter

        adapter = (
            strict_json_schema_adapter(request_backend=request_backend)
            if structured_output_transport == "response_format"
            else vllm_guided_json_adapter()
        )
        with dspy.context(lm=backend.lm, adapter=adapter):
            graph = backend.generate(input_data=PROBE_TEXT, cluster=cluster)
    else:
        graph = backend.generate(input_data=PROBE_TEXT, cluster=cluster)
    entities = set(getattr(graph, "entities", set()))
    relations = set(getattr(graph, "relations", set()))
    entities_count = len(entities)
    relations_count = len(relations)
    if entities_count < 4:
        raise RuntimeError(
            f"synthetic KGGen probe extracted only {entities_count} entities; expected at least four"
        )
    if relations_count < 2:
        raise RuntimeError(
            f"synthetic KGGen probe extracted only {relations_count} relations; expected at least two"
        )
    if not (
        _has_endpoints(relations, "Ada Lovelace", "Analytical Engine")
        or _has_endpoints(relations, "Ada Lovelace", "Charles Babbage")
    ):
        raise RuntimeError("synthetic KGGen probe omitted the Ada-Lovelace fact")
    if not _has_endpoints(relations, "Marie Curie", "Warsaw"):
        raise RuntimeError("synthetic KGGen probe omitted the Marie-Curie/Warsaw fact")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": f"openai/{model_id}",
        "api_base": api_base,
        "timeout_s": timeout_s,
        "max_tokens": max_tokens,
        "cluster": cluster,
        "structured_output_transport": structured_output_transport,
        "guided_decoding_request_backend": request_backend,
        "xgrammar_any_whitespace": (
            False if structured_output_transport == "response_format" else None
        ),
        "versions": {
            "kg-gen": metadata.version("kg-gen"),
            "dspy": metadata.version("dspy"),
            "litellm": metadata.version("litellm"),
        },
        "entities": entities_count,
        "relations": relations_count,
        "semantic_anchors": {
            "ada_lovelace": True,
            "marie_curie_warsaw": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also exercise KGGen's official LLM clustering pass.",
    )
    from src.dspy_adapter import XGRAMMAR_STRICT_REQUEST_BACKEND

    parser.add_argument(
        "--request-backend",
        choices=(XGRAMMAR_STRICT_REQUEST_BACKEND,),
        default=XGRAMMAR_STRICT_REQUEST_BACKEND,
    )
    parser.add_argument(
        "--structured-output-transport",
        choices=("none", "response_format", "guided_json"),
        default="response_format",
        help="Structured-output transport used by the target pilot runtime.",
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
        structured_output_transport=args.structured_output_transport,
        request_backend=args.request_backend,
    )
    _atomic_json(Path(args.report), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
