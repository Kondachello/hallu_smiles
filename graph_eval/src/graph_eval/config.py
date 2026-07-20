"""Typed GraphEval config (stdlib dataclasses; see DEVIATIONS.md #2)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .scoring import (
    AGGREGATION_MAX_UNSUPPORTED,
    DEFAULT_PAPER_THRESHOLD,
    VALID_AGGREGATIONS,
)


@dataclass(frozen=True)
class ExtractorConfig:
    backend: str = "fake"  # fake | gateway | vllm
    model: str = "openai/gemini-2.5-flash"
    prompt_profile: str = "grapheval_appendix_a_v1"
    output_mode: str = "paper_prompt"  # paper_prompt | structured_json
    temperature: float = 0.0
    max_tokens: int = 2048
    max_retries: int = 5
    api_base_env: str = "HALLU_GATEWAY_URL"
    api_key_env: str = "HALLU_GATEWAY_API_KEY"
    cache_namespace: str = "grapheval/extraction/v1"


@dataclass(frozen=True)
class NLIConfig:
    backend: str = "fake"  # fake | hhem
    model: str = "vectara/hallucination_evaluation_model"
    revision: str | None = None
    model_label: str = "hhem-2.1-open"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 8
    evidence_policy: str = "full_context_native"
    aggregation: str = AGGREGATION_MAX_UNSUPPORTED
    paper_threshold: float = DEFAULT_PAPER_THRESHOLD
    cache_namespace: str = "grapheval/nli/v1"


@dataclass(frozen=True)
class GraphEvalConfig:
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    cache_dir: str = ".cache/graph_eval"
    cache_only: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.nli.paper_threshold <= 1.0:
            raise ValueError("nli.paper_threshold must be in [0, 1]")
        if self.nli.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"unknown nli.aggregation: {self.nli.aggregation!r}")
        if self.extractor.output_mode not in ("paper_prompt", "structured_json"):
            raise ValueError(f"unknown extractor.output_mode: {self.extractor.output_mode!r}")


def _select(cls, values: dict) -> dict:
    return {k: v for k, v in (values or {}).items() if k in cls.__dataclass_fields__}


def from_dict(data: dict) -> GraphEvalConfig:
    data = data or {}
    cfg = GraphEvalConfig(
        extractor=ExtractorConfig(**_select(ExtractorConfig, data.get("extractor", {}))),
        nli=NLIConfig(**_select(NLIConfig, data.get("nli", {}))),
        cache_dir=data.get("cache_dir", ".cache/graph_eval"),
        cache_only=bool(data.get("cache_only", False)),
    )
    cfg.validate()
    return cfg
