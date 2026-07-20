"""GraphEval: a standalone answer-only KG + per-triple NLI hallucination detector."""
from __future__ import annotations

from .config import GraphEvalConfig, NLIConfig, ExtractorConfig, from_dict
from .detector import METHOD, GraphEvalDetector
from .types import (
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
    DetectionInput,
    DetectionResult,
    Triple,
)

__version__ = "0.1.0"
__all__ = [
    "GraphEvalDetector",
    "GraphEvalConfig",
    "ExtractorConfig",
    "NLIConfig",
    "from_dict",
    "DetectionInput",
    "DetectionResult",
    "Triple",
    "METHOD",
    "STATUS_OK",
    "STATUS_EMPTY_GRAPH",
    "STATUS_FAILED",
    "__version__",
]
